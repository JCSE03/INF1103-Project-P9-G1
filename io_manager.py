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

    # 2. Pass the data to your AI Manager
    prompt = ai_manager.build_prompt(form_data)
    raw_response = ai_manager.call_api(prompt)
    parsed_json = ai_manager.parse_response(raw_response)

    # 3. Validate the AI response
    if ai_manager.validate_response(parsed_json):

        print_message("AI processing successful.")

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

    else:

        print_message("AI validation failed.")

        return jsonify({
            "success": False,
            "error": "AI validation failed",
            "raw_data": parsed_json
        }), 400


# ==========================================
# MAIN EXECUTION
# ==========================================

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print_message(
        "Starting web server... Press CTRL+C in the terminal to stop it."
    )

    # Automatically open the user page after 1 second
    Timer(1, open_browser).start()

    # Run the Flask server
    app.run(port=5000, debug=False)