"""I/O Manager: every boundary with the outside world (web page, console).

It holds no business rules, AI or file code. For each ticket it calls:
    ai_manager.classify_ticket   security checks, redaction, Gemini, validation
    logic_manager.process_ticket priority, department, review flag
    data_manager.add_record      saves the decision
and for the queue view:
    data_manager.find_records + logic_manager.sort_queue
"""

import os
import webbrowser
from datetime import datetime, timezone
from threading import Lock, Timer

from flask import Flask, jsonify, render_template, request

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


def _error(reason, code):
    return jsonify({"status": "error", "reason": reason, "manual_review": True}), code


# ==========================================
# SIMPLE PRINT FUNCTION (all console output goes through here)
# ==========================================
def print_message(message):
    print(message)


# ==========================================
# FLASK WEB ROUTES (connecting HTML to Python)
# ==========================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/submit-ticket", methods=["POST"])
def submit_ticket():
    form_data = request.get_json(silent=True)
    if not isinstance(form_data, dict):
        return jsonify({
            "status": "invalid",
            "reason": "Request must be a JSON object.",
            "manual_review": True,
        }), 400

    # One rate-limit bucket per client address.
    user_id = request.remote_addr or "anonymous"

    # The whole AI pipeline lives in ai_manager. The ticket text is not
    # printed here, to keep it out of the console.
    result = ai_manager.classify_ticket(form_data, user_id=user_id)
    print_message(f"Ticket from {user_id}: {result['status']}")

    # Rejected, blocked, rate-limited or failed tickets are reported, not saved.
    if result["status"] != "success":
        return jsonify(result), STATUS_CODES.get(result["status"], 500)

    # Business rules: priority, department, human-review flag.
    decision = logic_manager.process_ticket(result)

    # Save. Only the AI's summary is stored, never the raw ticket text,
    # which may contain details ai_manager redacted before the API call.
    with _data_lock:
        records = data_manager.load_records()
        if records is None:
            # Unreadable file: do NOT save, or the old data would be overwritten.
            print_message("tickets.json is unreadable; ticket was NOT saved.")
            return _error("Ticket data file could not be read, so the ticket was not saved.", 503)

        ticket = {
            "ticket_id": f"T-{len(records) + 1:04d}",
            "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **decision,
        }
        if not data_manager.add_record(records, ticket):
            print_message("Could not write tickets.json; ticket was NOT saved.")
            return _error("The ticket could not be saved. Please try again.", 503)

    print_message(f"Saved {ticket['ticket_id']}: {ticket['priority']} -> {ticket['department']}")

    result["ticket"] = ticket
    result["manual_review"] = bool(result.get("manual_review") or ticket["needs_review"])
    return jsonify(result), 200


@app.route("/queue")
def queue():
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


# ==========================================
# SERVER START-UP (called by main.py)
# ==========================================
def open_browser(host, port):
    webbrowser.open(f"http://{host}:{port}")


def run_server(host="127.0.0.1", port=5000, auto_open=True):
    print_message("Starting web server... Press CTRL+C in the terminal to stop it.")
    if auto_open:
        Timer(1, open_browser, args=(host, port)).start()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":  # still works on its own, but main.py is the entry point
    run_server()
