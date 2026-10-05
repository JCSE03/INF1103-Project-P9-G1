"""I/O Manager: every boundary with the outside world (web page, console).

It holds no AI logic: tickets are handed to ai_manager.classify_ticket, which
does the security checks, redaction, API call, parsing and validation.
"""

import os
import webbrowser
from threading import Timer

from flask import Flask, jsonify, render_template, request

import ai_manager

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

    return jsonify(result), STATUS_CODES.get(result["status"], 500)


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
