from flask import Flask, render_template, request, jsonify
import webbrowser
from threading import Timer
from dotenv import load_dotenv

load_dotenv()

import ai_manager

# Initialize Flask app to look for index.html in the same folder
app = Flask(__name__, template_folder='.')

# Stores the latest successfully processed ticket
latest_ticket = None


# ==========================================
# SIMPLE PRINT FUNCTION
# ==========================================
def print_message(message):
    print(message)


# ==========================================
# FLASK WEB ROUTES
# ==========================================

@app.route("/")
def home():
    # Show the user ticket submission page
    return render_template("index.html")


@app.route("/helpdesk")
def helpdesk():
    # Show the helpdesk dashboard
    return render_template("helpdesk.html")


@app.route("/ticket-status")
def ticket_status():
    # Check whether a ticket has been submitted
    if latest_ticket is None:
        return jsonify({
            "available": False
        })

    return jsonify({
        "available": True,
        "ticket": latest_ticket
    })


@app.route("/submit-ticket", methods=["POST"])
def submit_ticket():
    global latest_ticket

    # 1. Grab the data sent from the webpage form
    form_data = request.json

    print_message(
        "Processing new ticket: "
        + str(form_data.get("ticket_title"))
    )

    # 2. Send the ticket through the complete AI classification pipeline
    #
    # classify_ticket() handles:
    # - Security checks
    # - Prompt creation
    # - Gemini API call
    # - Response parsing
    # - Response validation
    # - Retries if Gemini fails or returns invalid data
    classification = ai_manager.classify_ticket(form_data)

    # ==========================================
    # AI PROCESSING SUCCESSFUL
    # ==========================================

    if classification["status"] == "success":

        print_message("AI processing successful.")

        parsed_json = classification["data"]

        # Store the latest ticket for the helpdesk page
        latest_ticket = {
            "ticket_title": form_data.get("ticket_title"),
            "problem_description": form_data.get("problem_description"),
            "affected_service": form_data.get("affected_service"),
            "result": parsed_json
        }

        # Send confirmation back to the user
        return jsonify({
            "success": True,
            "message": "Thank you for submitting your ticket. We will be in touch."
        })


    # ==========================================
    # INVALID TICKET
    # ==========================================

    elif classification["status"] == "invalid":

        print_message(
            "Ticket rejected: "
            + classification["reason"]
        )

        return jsonify({
            "success": False,
            "error": classification["reason"]
        }), 400


    # ==========================================
    # BLOCKED TICKET
    # ==========================================

    elif classification["status"] == "blocked":

        print_message(
            "Ticket blocked: "
            + classification["reason"]
        )

        return jsonify({
            "success": False,
            "error": classification["reason"]
        }), 400


    # ==========================================
    # AI PROCESSING FAILED
    # ==========================================

    else:

        print_message(
            "AI processing failed: "
            + classification.get(
                "reason",
                "Unknown AI processing error."
            )
        )

        return jsonify({
            "success": False,
            "error": classification.get(
                "reason",
                "AI processing failed."
            )
        }), 500


# ==========================================
# OPEN BROWSER
# ==========================================

def open_browser():

    # Open the user ticket submission page
    webbrowser.open("http://127.0.0.1:5000/")

    # Open the helpdesk dashboard
    webbrowser.open("http://127.0.0.1:5000/helpdesk")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print_message(
        "Starting web server... Press CTRL+C in the terminal to stop it."
    )

    # Automatically open both pages after 1 second
    Timer(1, open_browser).start()

    # Run the Flask server
    app.run(port=5000, debug=False)