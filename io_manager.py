from datetime import datetime, timezone
from uuid import uuid4
import webbrowser
from threading import Timer

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

load_dotenv()

import ai_manager
import data_manager
import logic_manager

app = Flask(__name__, template_folder=".")


def print_message(message):
    print(message)


def get_records():
    records = data_manager.load_records()
    return records if records is not None else []


def queue_summary(records):
    departments = {}

    for department in logic_manager.VALID_DEPARTMENTS:
        department_records = [
            ticket for ticket in records
            if ticket.get("result", {}).get("department") == department
        ]

        departments[department] = {
            "total": len(department_records),
            "queued": len([
                ticket for ticket in department_records
                if ticket.get("result", {}).get("queue_status") == "Queued"
            ]),
        }

    return {
        "total": len(records),
        "queued": len([
            ticket for ticket in records
            if ticket.get("result", {}).get("queue_status") == "Queued"
        ]),
        "departments": departments,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/helpdesk")
def helpdesk():
    return render_template(
        "helpdesk.html",
        initial_department="all",
        departments=logic_manager.VALID_DEPARTMENTS,
    )


@app.route("/departments/<department>")
def department_page(department):
    if department not in logic_manager.VALID_DEPARTMENTS:
        return jsonify({"error": "Unknown department."}), 404

    return render_template(
        "helpdesk.html",
        initial_department=department,
        departments=logic_manager.VALID_DEPARTMENTS,
    )


@app.route("/api/queue-status")
def ticket_queue_status():
    return jsonify(queue_summary(get_records()))


@app.route("/api/tickets")
def tickets():
    department = request.args.get("department", "all")
    records = get_records()

    if department != "all":
        if department not in logic_manager.VALID_DEPARTMENTS:
            return jsonify({"error": "Unknown department."}), 400

        records = [
            ticket for ticket in records
            if ticket.get("result", {}).get("department") == department
        ]

    # Newest tickets first.
    records = list(reversed(records))

    return jsonify({
        "tickets": records,
        "queue": queue_summary(get_records()),
    })


@app.route("/submit-ticket", methods=["POST"])
def submit_ticket():
    form_data = request.get_json(silent=True) or {}

    required_fields = (
        "ticket_title",
        "name",
        "contact_number",
        "problem_description",
        "affected_users",
        "work_blocked",
        "workaround_available",
    )

    if any(not str(form_data.get(field, "")).strip() for field in required_fields):
        return jsonify({
            "success": False,
            "error": "Please complete all required fields."
        }), 400

    print_message("Processing new ticket: " + str(form_data.get("ticket_title")))

    classification = ai_manager.classify_ticket(form_data)

    if classification["status"] != "success":
        return jsonify({
            "success": False,
            "error": classification.get(
                "reason",
                "AI processing failed. Please try again."
            )
        }), 400

    parsed_result = classification["data"]
    routing = logic_manager.route_ticket(parsed_result)

    result = {
        **parsed_result,
        **routing,
        "manual_review": classification.get("manual_review", False),
        "flag_reason": classification.get("flag_reason"),
    }

    records = get_records()
    department_tickets_before_submission = [
        ticket for ticket in records
        if ticket.get("result", {}).get("department") == result["department"]
    ]

    ticket = {
        "ticket_id": f"TKT-{uuid4().hex[:8].upper()}",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "ticket_title": form_data["ticket_title"].strip(),
        "name": form_data["name"].strip(),
        "contact_number": form_data["contact_number"].strip(),
        "problem_description": form_data["problem_description"].strip(),
        "affected_users": form_data["affected_users"],
        "work_blocked": form_data["work_blocked"],
        "workaround_available": form_data["workaround_available"],
        "result": result,
        "queue_position": len(department_tickets_before_submission) + 1,
    }

    if not data_manager.add_record(records, ticket):
        return jsonify({
            "success": False,
            "error": "The ticket could not be saved. Please try again."
        }), 500

    return jsonify({
        "success": True,
        "message": (
            f"Ticket {ticket['ticket_id']} has been submitted to "
            f"{result['department']}."
        ),
        "ticket_id": ticket["ticket_id"],
        "helpdesk_url": f"/departments/{result['department']}",
    })


def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")
    webbrowser.open("http://127.0.0.1:5000/helpdesk")


if __name__ == "__main__":
    print_message("Starting web server... Press CTRL+C in the terminal to stop it.")
    Timer(1, open_browser).start()
    app.run(port=5000, debug=False)