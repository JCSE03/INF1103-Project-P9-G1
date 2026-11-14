import os
import re
import webbrowser
from datetime import datetime, timezone
from threading import Lock, Timer

from flask import Flask, Response, jsonify, render_template, request

import ai_manager
import ai_summary
import ai_troubleshooting
import analytics
import data_manager
import logic_manager
import pending_tickets


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


# Prevents race conditions during concurrent write operations to tickets.json
#like two tickets writing at the same time
_data_lock = Lock()

# The different returns of the Flask
RouteResult = str | Response | tuple[Response, int]

# Self-service: writes troubleshooting steps for low-urgency Infra tickets, and
# holds those tickets (not saved) while the requester tries the steps.
# Both are used only through their abstract types.
_advisor: ai_troubleshoot.TicketAdvisor = ai_troubleshooting.TroubleshootingAdvisor.with_gemini()
_pending: pending_tickets.PendingTicketStore = pending_tickets.InMemoryPendingTickets()


def _error(reason: str, code: int) -> tuple[Response, int]:
    return jsonify({"status": "error", "reason": reason, "manual_review": True}), code


MAX_NAME_LENGTH = 100
MAX_TITLE_LENGTH = 120
# sg num: optional +65, then 8 digits starting with 3, 6, 8 or 9
# (a space or dash in the middle is allowed), e.g. 91234567, +65 9123 4567.
CONTACT_PATTERN = re.compile(r"^(\+65[ -]?)?[3689][0-9]{3}[ -]?[0-9]{4}$")


def requester_details(form_data: dict) -> tuple[dict | None, str | None]:
    """Validate user input fields in the index form"""
    fields = {key: form_data.get(key) for key in ("name", "contact_number", "ticket_title")}
    if not all(isinstance(value, str) and value.strip() for value in fields.values()):
        return None, "Please fill in your name, contact number and a ticket title."

    name = ai_manager.sanitize(fields["name"], MAX_NAME_LENGTH)
    contact = ai_manager.sanitize(fields["contact_number"])
    title = ai_manager.sanitize(fields["ticket_title"], MAX_TITLE_LENGTH)

    if not CONTACT_PATTERN.match(contact):
        return None, "Please enter a valid 8-digit Singapore number, e.g. 9123 4567."

    title, _ = ai_manager.redact_sensitive(title)
    return {"name": name, "contact_number": contact, "ticket_title": title}, None



def print_message(message: str) -> None:
    print(message)


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

    # Validates the input fields early to avoid wasting API quote on invalid inputs
    requester, problem = requester_details(form_data)
    if problem:
        return jsonify({"status": "invalid", "reason": problem, "manual_review": True}), 400

    user_id = request.remote_addr or "anonymous"

    #Only send title and description to Gemini API
    description = form_data.get("problem_description")
    ticket_text = (f"{form_data['ticket_title']}\n{description}"
                   if isinstance(description, str) and description.strip() else "")
    result = ai_manager.classify_ticket(ticket_text, user_id=user_id)
    print_message(f"Ticket from {user_id}: {result['status']}")

    
    if result["status"] != "success":
        return jsonify(result), STATUS_CODES.get(result["status"], 500)

    decision = logic_manager.process_ticket(result)

    # Low-urgency Infra tickets: offer AI troubleshooting steps first. The
    # ticket is held, not saved, until the requester says the steps did not help.
    if logic_manager.offers_self_service(decision):
        steps = _advisor.advise(decision["category"], decision["summary"],
                                ticket_text, user_id=user_id)
        if steps.is_success:
            token = _pending.hold({"requester": requester, "decision": decision})
            print_message(f"Held {decision['priority']} -> {decision['department']} "
                          "ticket: troubleshooting steps shown, not saved yet.")
            return jsonify({"status": "troubleshooting", "pending_id": token,
                            "troubleshooting": steps.to_dict(),
                            "manual_review": False}), 200
        # The steps are only a convenience: if the AI cannot write them, fall
        # through and save the ticket as normal rather than leave it unsent.
        print_message(f"Troubleshooting unavailable ({steps.status}); saving the ticket.")

    saved = _save_ticket(requester, decision)
    if isinstance(saved, tuple):
        return saved  # an error response
    result["ticket"] = saved
    result["manual_review"] = bool(result.get("manual_review") or saved["needs_review"])
    return jsonify(result), 200


def _save_ticket(requester: dict, decision: dict) -> dict | tuple[Response, int]:
    """Saves one processed ticket to tickets.json. Returns the saved ticket, or
    an error response (a tuple) if it could not be saved."""
    with _data_lock:
        records = data_manager.load_records()
        if records is None:
            # Abort saving if the file reading fails
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
    return ticket


@app.route("/api/pending-tickets/<pending_id>/answer", methods=["POST"])
def answer_troubleshooting(pending_id: str) -> RouteResult:
    """The requester's answer to "did the steps fix it?". Send {"solved": true}
    to drop the held ticket, or {"solved": false} to save it for the IT team."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("solved"), bool):
        return _error('Send {"solved": true} or {"solved": false}.', 400)

    held = _pending.take(pending_id)
    if held is None:
        return _error("This request has expired or was already answered. "
                      "Please submit it again if you still need help.", 410)

    if data["solved"]:
        print_message("Troubleshooting solved the issue; the ticket was dropped, not saved.")
        return jsonify({"status": "closed"})

    saved = _save_ticket(held["requester"], held["decision"])
    if isinstance(saved, tuple):
        _pending.put_back(pending_id, held)  # let the requester try again
        return saved
    return jsonify({"status": "success", "ticket": saved,
                    "manual_review": bool(saved["needs_review"])})


@app.route("/queue")
def queue() -> RouteResult:
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


@app.route("/analytics")
def analytics_page() -> str:
    """Charts and statistics for IT managers."""
    return render_template("analytics.html", departments=logic_manager.DEPARTMENTS)


# Asks the AI for a short summary of the analytics. One shared instance, so its
# cache and its request limit apply to every visitor.
_summarizer = ai_summary.ReportSummarizer.with_gemini()


def _report_for_request() -> tuple[dict | None, tuple[Response, int] | None]:
    """The analytics report for ?department= (all departments by default), as
    (report, None), or (None, error_response) if the request or the data is bad."""
    department = request.args.get("department")
    if department and department not in logic_manager.DEPARTMENTS:
        return None, _error("Unknown department.", 400)

    with _data_lock:
        records = data_manager.load_records()
    if records is None:
        return None, _error("Ticket data file could not be read.", 503)

    if department:
        records = data_manager.find_records(records, "department", department)
    return analytics.AnalyticsReport().build(records), None


@app.route("/api/analytics")
def analytics_report() -> RouteResult:
    report, error = _report_for_request()
    return error or jsonify({"status": "success", "report": report})


@app.route("/api/analytics/summary", methods=["POST"])
def analytics_summary() -> RouteResult:
    """An AI-written summary of the same report. POST, because it costs an AI request."""
    report, error = _report_for_request()
    if error:
        return error

    result = _summarizer.summarize(report, user_id=request.remote_addr or "anonymous")
    code = 200 if result.status == "no_data" else STATUS_CODES.get(result.status, 500)
    return jsonify(result.to_dict()), code


@app.route("/helpdesk")
def helpdesk() -> str:
    """Dashboard for helpdesk staff, every department's queue."""
    return render_template("helpdesk.html", initial_department="all",
                           departments=logic_manager.DEPARTMENTS)


@app.route("/departments/<department>")
def department_page(department: str) -> RouteResult:
    if department not in logic_manager.DEPARTMENTS:
        return _error("Unknown department.", 404)
    return render_template("helpdesk.html", initial_department=department,
                           departments=logic_manager.DEPARTMENTS)


@app.route("/api/tickets/<ticket_id>/toggle-resolution", methods=["POST"])
def toggle_resolution(ticket_id: str) -> RouteResult:
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



def open_browser(host: str, port: int) -> None:
    """Open the ticket form and the helpdesk dashboard in two tabs."""
    webbrowser.open(f"http://{host}:{port}/")
    webbrowser.open(f"http://{host}:{port}/helpdesk")


def run_server(host: str = "127.0.0.1", port: int = 5000, auto_open: bool = True) -> None:
    print_message("Starting web server... Press CTRL+C in the terminal to stop it.")
    if auto_open:
        Timer(1, open_browser, args=(host, port)).start()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":  
    run_server()