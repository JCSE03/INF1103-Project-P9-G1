from flask import Flask, render_template, request, jsonify
import webbrowser
from threading import Timer
import ai_manager  # Your AI logic file

# Initialize Flask app to look for index.html in the same folder
app = Flask(__name__, template_folder='.') 

# ==========================================
# SIMPLE PRINT FUNCTION
# ==========================================
def print_message(message):
    print(message)

# ==========================================
# FLASK WEB ROUTES (Connecting HTML to Python)
# ==========================================
@app.route("/")
def home():
    # Show the web page when someone visits the site
    return render_template("index.html")

@app.route("/submit-ticket", methods=["POST"])
def submit_ticket():
    # 1. Grab the data sent from the webpage form
    form_data = request.json
    
    print_message("Processing new ticket: " + str(form_data.get('ticket_title')))

    # 2. Pass the data to your AI Manager[cite: 1]
    prompt = ai_manager.build_prompt(form_data)
    raw_response = ai_manager.call_api(prompt)
    parsed_json = ai_manager.parse_response(raw_response)
    
    # 3. Send the AI's JSON answer back to the webpage
    if ai_manager.validate_response(parsed_json):
        print_message("AI processing successful.")
        return jsonify(parsed_json)
    else:
        print_message("AI validation failed.")
        return jsonify({"error": "AI validation failed", "raw_data": parsed_json}), 400

# ==========================================
# MAIN EXECUTION
# ==========================================
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    print_message("Starting web server... Press CTRL+C in the terminal to stop it.")
    
    # Automatically open the browser after 1 second
    Timer(1, open_browser).start()
    
    # Run the Flask server
    app.run(port=5000, debug=False)