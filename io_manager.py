from flask import Flask, render_template, request, jsonify
import webbrowser
from threading import Timer
import ai_manager      # Your AI logic file
import logic_manager   # <-- NEW: Import your Logic Manager

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

    # 2. Pass the data to your AI Manager
    prompt = ai_manager.build_prompt(form_data)
    raw_response = ai_manager.call_api(prompt)
    parsed_json = ai_manager.parse_response(raw_response)
    
    # 3. Validate and Route
    if ai_manager.validate_response(parsed_json):
        print_message("AI processing successful.")
        
        # --- NEW: Run the Logic Manager ---
        # This is where we translate the AI text into priority scores and departments
        final_result = logic_manager.calculate_priority(parsed_json)
        
        # Log the decision to the terminal so you can see it working
        print_message(f"-> Logic Calculated: {final_result['priority']} (Score: {final_result['priority_score']}) routed to {final_result['assigned_department']}")
        
        # Send the enriched result to the webpage
        return jsonify(final_result)
        
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