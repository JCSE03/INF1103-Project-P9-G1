from google import genai
import json
import logging
logging.basicConfig(level=logging.ERROR)

client = genai.Client(api_key="")


def build_prompt(record):
    return f"""
You are an AI classification component for an internal bank IT helpdesk.
Analyse the following IT support ticket.
Ticket title:
{record["ticket_title"]}
Problem description:
{record["problem_description"]}
Affected service/device:
{record["affected_service"]}
Number of affected users:
{record["affected_users"]}
Work blocked:
{record["work_blocked"]}
Workaround available:
{record["workaround_available"]}
Your job is ONLY to classify and extract information.
Do NOT:
- assign final priority
- assign SLA
- decide escalation
- decide routing
- recommend actions
Return ONLY valid JSON in this format:
{{
    "category": "",
    "issue_type": "",
    "technical_severity": "",
    "affected_scope": "",
    "summary": "",
    "confidence_score": 0.0
}}
Category must be one of:
hardware, software, network, account_access, cybersecurity,
email, printer, banking_application, other
Technical severity must be one of:
low, medium, high, critical
Affected scope must be one of:
individual, multiple_users, department, organisation
confidence_score must be between 0.0 and 1.0.
Do not include Markdown.
Do not include explanations.
Return JSON only.
"""


import time

def call_api(prompt):
    # Try up to 3 times before giving up
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            return response.text

        except Exception as error:
            error_message = str(error)
            logging.error(f"Gemini API error (Attempt {attempt + 1}): {error_message}")
            
            # If it's a 503 busy error, wait 5 seconds and try again
            if "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < 2:
                    time.sleep(5)
                    continue
            
            # If it's a different error or we ran out of tries, return None
            return None


def parse_response(raw):
    if raw is None:
        logging.error("No response received from Gemini.")
        return None

    try:
        return json.loads(raw)

    except (json.JSONDecodeError, TypeError):
        logging.error("Invalid JSON returned by Gemini.")
        return None

def validate_response(data):

    if not isinstance(data, dict):
        return False

    required_keys = [
        "category",
        "issue_type",
        "technical_severity",
        "affected_scope",
        "summary",
        "confidence_score"
    ]

    # Check that all required fields exist
    for key in required_keys:
        if key not in data:
            return False

    valid_categories = [
        "hardware",
        "software",
        "network",
        "account_access",
        "cybersecurity",
        "email",
        "printer",
        "banking_application",
        "other"
    ]

    valid_severities = [
        "low",
        "medium",
        "high",
        "critical"
    ]

    valid_scopes = [
        "individual",
        "multiple_users",
        "department",
        "organisation"
    ]

    if data["category"] not in valid_categories:
        return False

    if data["technical_severity"] not in valid_severities:
        return False

    if data["affected_scope"] not in valid_scopes:
        return False

    if not isinstance(data["issue_type"], str):
        return False

    if not isinstance(data["summary"], str):
        return False

    if not isinstance(data["confidence_score"], (int, float)):
        return False

    if not 0 <= data["confidence_score"] <= 1:
        return False

    return True