"""I/O Manager: every boundary with the outside world (web page, console).

It holds no business rules, AI or file code. For each ticket it calls:
    ai_manager.classify_ticket   security checks, redaction, Gemini, validation
    logic_manager.process_ticket priority, department, review flag
    data_manager.add_record      saves the decision
and for the queue view:
    data_manager.find_records + logic_manager.sort_queue
and for the helpdesk dashboard (resolve / reopen a ticket):
    data_manager.set_resolution
"""

import os
import re
import webbrowser
from datetime import datetime, timezone
from threading import Lock, Timer

from flask import Flask, Response, jsonify, render_template, request

import ai_manager
import data_manager
import logic_manager

# Look for index.html in the same folder as this file, wherever we are run from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=BASE_DIR)

# HTTP status for each "status" value classify_ticket can return.
STATUS_CODES = {
    "success": 200,
    "invalid": 400,
    "blocked": 403,
    "rate_limited": 429,
    "error": 503,
}


# Requests are handled on several threads; only one may read-modify-write
# the ticket file at a time, or two tickets could overwrite each other.
_data_lock = Lock()

# What a Flask route can return: a page, JSON, or JSON with a status code.
RouteResult = str | Response | tuple[Response, int]


def _error(reason: str, code: int) -> tuple[Response, int]:
    return jsonify({"status": "error", "reason": reason, "manual_review": True}), code


# Who raised the ticket. These are facts the user knows, so they come from
# the form. Category, severity and scope are NEVER asked for: the AI infers them.
MAX_NAME_LENGTH = 100
MAX_TITLE_LENGTH = 120
CONTACT_PATTERN = re.compile(r"^\+?[0-9][0-9 -]{6,18}[0-9]$")


def requester_details(form_data: dict) -> tuple[dict | None, str | None]:
    """Check and clean the name, contact number and title.
    Returns (details, None) if valid, or (None, reason) if not."""
    fields = {key: form_data.get(key) for key in ("name", "contact_number", "ticket_title")}
    if not all(isinstance(value, str) and value.strip() for value in fields.values()):
        return None, "Please fill in your name, contact number and a ticket title."

    name = ai_manager.sanitize(fields["name"], MAX_NAME_LENGTH)
    # Not shortened first: an over-long number must fail the check, not be cut.
    contact = ai_manager.sanitize(fields["contact_number"])
    title = ai_manager.sanitize(fields["ticket_title"], MAX_TITLE_LENGTH)

    if not CONTACT_PATTERN.match(contact):
        return None, "Contact number should be 8 to 15 digits, e.g. +65 9123 4567."

    # The title is stored and shown, so remove card numbers, passwords etc.
    title, _ = ai_manager.redact_sensitive(title)
    return {"name": name, "contact_number": contact, "ticket_title": title}, None


# ==========================================
# SIMPLE PRINT FUNCTION (all console output goes through here)
# ==========================================
def print_message(message: str) -> None:
    print(message)


# ==========================================
# FLASK WEB ROUTES (connecting HTML to Python)
# ==========================================
@app.route("/")
def home() -> str:
    return render_template("index.html")


@app.route("/submit-ticket", methods=["POST"])
def submit_ticket() -> RouteResult:
    form_data = request.get_json(silent=True)
    if not isinstance(form_data, dict):
        return jsonify({
            "status": "invalid",
            "reason": "Request must be a JSON object.",
            "manual_review": True,
        }), 400

    # Checked before the AI call, so a bad form never uses API quota.
    requester, problem = requester_details(form_data)
    if problem:
        return jsonify({"status": "invalid", "reason": problem, "manual_review": True}), 400

    # One rate-limit bucket per client address.
    user_id = request.remote_addr or "anonymous"

    # The whole AI pipeline lives in ai_manager. The AI sees the title and
    # description only: the name and contact number are never sent to Gemini.
    # The ticket text is not printed here, to keep it out of the console.
    description = form_data.get("problem_description")
    ticket_text = (f"{form_data['ticket_title']}\n{description}"
                   if isinstance(description, str) and description.strip() else "")
    result = ai_manager.classify_ticket(ticket_text, user_id=user_id)
    print_message(f"Ticket from {user_id}: {result['status']}")

    # Rejected, blocked, rate-limited or failed tickets are reported, not saved.
    if result["status"] != "success":
        return jsonify(result), STATUS_CODES.get(result["status"], 500)

    # Business rules: priority, department, human-review flag.
    decision = logic_manager.process_ticket(result)

    # Save. The AI's summary is stored, never the raw description, which may
    # contain details ai_manager redacted before the API call.
    with _data_lock:
        records = data_manager.load_records()
        if records is None:
            # Unreadable file: do NOT save, or the old data would be overwritten.
            print_message("tickets.json is unreadable; ticket was NOT saved.")
            return _error("Ticket data file could not be read, so the ticket was not saved.", 503)

        ticket = {
            "ticket_id": f"T-{len(records) + 1:04d}",
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **requester,
            **decision,
            "is_resolved": False,
            "resolved_at": None,
        }
        if not data_manager.add_record(records, ticket):
            print_message("Could not write tickets.json; ticket was NOT saved.")
            return _error("The ticket could not be saved. Please try again.", 503)

    print_message(f"Saved {ticket['ticket_id']}: {ticket['priority']} -> {ticket['department']}")

    result["ticket"] = ticket
    result["manual_review"] = bool(result.get("manual_review") or ticket["needs_review"])
    return jsonify(result), 200


@app.route("/queue")
def queue() -> RouteResult:
    """Saved tickets, most urgent first. Optional ?department=Cyber|HR|Finance|Infra."""
    department = request.args.get("department")
    if department and department not in logic_manager.DEPARTMENTS:
        return _error("Unknown department.", 400)

    with _data_lock:
        records = data_manager.load_records()
    if records is None:
        return _error("Ticket data file could not be read.", 503)

    if department:
        records = data_manager.find_records(records, "department", department)

    return jsonify({"status": "success", "tickets": logic_manager.sort_queue(records)})


@app.route("/helpdesk")
def helpdesk() -> str:
    """Dashboard for helpdesk staff: every department's queue."""
    return render_template("helpdesk.html", initial_department="all",
                           departments=logic_manager.DEPARTMENTS)


@app.route("/departments/<department>")
def department_page(department: str) -> RouteResult:
    """The same dashboard, opened on one department's queue."""
    if department not in logic_manager.DEPARTMENTS:
        return _error("Unknown department.", 404)
    return render_template("helpdesk.html", initial_department=department,
                           departments=logic_manager.DEPARTMENTS)


@app.route("/api/tickets/<ticket_id>/toggle-resolution", methods=["POST"])
def toggle_resolution(ticket_id: str) -> RouteResult:
    """Mark a ticket resolved, or reopen it. Body: {"is_resolved": true|false}."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("is_resolved"), bool):
        return _error('Send {"is_resolved": true} or {"is_resolved": false}.', 400)

    is_resolved = data["is_resolved"]
    resolved_at = (datetime.now(timezone.utc).isoformat(timespec="seconds")
                   if is_resolved else None)

    with _data_lock:
        records = data_manager.load_records()
        if records is None:
            # Unreadable file: do NOT save, or the old data would be overwritten.
            return _error("Ticket data file could not be read, so nothing was changed.", 503)
        outcome = data_manager.set_resolution(records, ticket_id, is_resolved, resolved_at)

    if outcome == "not_found":
        return _error("Ticket not found.", 404)
    if outcome == "save_failed":
        return _error("The change could not be saved. Please try again.", 503)

    print_message(f"{ticket_id} {'resolved' if is_resolved else 'reopened'}")
    return jsonify({"status": "success", "ticket_id": ticket_id,
                    "is_resolved": is_resolved, "resolved_at": resolved_at})


# ==========================================
# SERVER START-UP (called by main.py)
# ==========================================
def open_browser(host: str, port: int) -> None:
    """Open the ticket form and the helpdesk dashboard in two tabs."""
    webbrowser.open(f"http://{host}:{port}/")
    webbrowser.open(f"http://{host}:{port}/helpdesk")


def run_server(host: str = "127.0.0.1", port: int = 5000, auto_open: bool = True) -> None:
    print_message("Starting web server... Press CTRL+C in the terminal to stop it.")
    if auto_open:
        Timer(1, open_browser, args=(host, port)).start()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":  # still works on its own, but main.py is the entry point
    run_server()