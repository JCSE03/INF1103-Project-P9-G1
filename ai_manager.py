
import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# CONFIGURATION
# ==========================================

logging.basicConfig(level=logging.ERROR)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None

MAX_RETRIES = 2
RETRY_DELAY = 5


# ==========================================
# GEMINI CLIENT
# ==========================================

def get_client():
    global client

    if client is None:

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "API_KEY environment variable is not set."
            )

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    return client


# ==========================================
# PROMPT CREATION
# ==========================================

def build_prompt(record):

    return f"""
You are an AI classification component for an internal bank IT helpdesk.

Analyse the following IT support ticket.

Ticket title:
{record.get("ticket_title", "")}

Problem description:
{record.get("problem_description", "")}

Affected service:
{record.get("affected_service", "")}

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{{
    "category": "",
    "issue_type": "",
    "technical_severity": "",
    "affected_scope": "",
    "summary": "",
    "confidence_score": 0.0
}}

Field requirements:

category:
Classify the general technical area of the issue.

Examples:
- Account Access
- Hardware
- Network
- Software
- Security
- Email
- Cloud
- Other

issue_type:
Describe the specific type of technical issue.

technical_severity:
Describe the technical impact of the problem using only:
- Low
- Medium
- High
- Critical

affected_scope:
Describe how broadly the issue appears to affect users or systems.

Examples:
- Single User
- Multiple Users
- Department
- Organisation
- Unknown

summary:
Provide a concise technical summary of the reported issue.

confidence_score:
A number between 0.0 and 1.0 representing your confidence
in the classification.

Important restrictions:

Do NOT determine:
- final ticket priority
- SLA
- escalation
- routing
- business impact
- business criticality
- remediation steps
- recommended actions

Those decisions are handled separately by the IT helpdesk system.

Do not include markdown.
Do not include ```json.
Return only the JSON object.
"""


# ==========================================
# GEMINI API CALL
# ==========================================

def call_api(prompt):

    api_client = get_client()

    response = api_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    if not response or not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


# ==========================================
# RESPONSE PARSING
# ==========================================

def parse_response(response_text):

    if not response_text:
        raise ValueError(
            "Empty AI response."
        )

    cleaned = response_text.strip()

    # Remove accidental markdown code fences.
    if cleaned.startswith("```"):

        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:

        parsed = json.loads(cleaned)

    except json.JSONDecodeError as error:

        raise ValueError(
            f"AI returned invalid JSON: {error}"
        ) from error

    if not isinstance(parsed, dict):

        raise ValueError(
            "AI response must be a JSON object."
        )

    return parsed


# ==========================================
# RESPONSE VALIDATION
# ==========================================

def validate_response(data):

    required_fields = {
        "category",
        "issue_type",
        "technical_severity",
        "affected_scope",
        "summary",
        "confidence_score"
    }

    if not isinstance(data, dict):

        return False, (
            "AI response is not a JSON object."
        )

    missing_fields = (
        required_fields - set(data.keys())
    )

    if missing_fields:

        return False, (
            "AI response is missing required fields: "
            + ", ".join(sorted(missing_fields))
        )

    text_fields = [
        "category",
        "issue_type",
        "technical_severity",
        "affected_scope",
        "summary"
    ]

    for field in text_fields:

        if not isinstance(data[field], str):

            return False, (
                f"Field '{field}' must be a string."
            )

        if not data[field].strip():

            return False, (
                f"Field '{field}' cannot be empty."
            )

    # ------------------------------------------
    # TECHNICAL SEVERITY
    # ------------------------------------------

    valid_severities = {
        "Low",
        "Medium",
        "High",
        "Critical"
    }

    if data["technical_severity"] not in valid_severities:

        return False, (
            "Invalid technical_severity. "
            "Expected Low, Medium, High, or Critical."
        )

    # ------------------------------------------
    # AFFECTED SCOPE
    # ------------------------------------------

    valid_scopes = {
        "Single User",
        "Multiple Users",
        "Department",
        "Organisation",
        "Unknown"
    }

    if data["affected_scope"] not in valid_scopes:

        return False, (
            "Invalid affected_scope. "
            "Expected Single User, Multiple Users, "
            "Department, Organisation, or Unknown."
        )

    # ------------------------------------------
    # CONFIDENCE SCORE
    # ------------------------------------------

    confidence = data["confidence_score"]

    if isinstance(confidence, bool):

        return False, (
            "confidence_score must be a number."
        )

    if not isinstance(confidence, (int, float)):

        return False, (
            "confidence_score must be a number."
        )

    if not 0.0 <= float(confidence) <= 1.0:

        return False, (
            "confidence_score must be between 0.0 and 1.0."
        )

    data["confidence_score"] = float(
        confidence
    )

    return True, None


# ==========================================
# SECURITY CHECK
# ==========================================

def security_check(record):

    if not isinstance(record, dict):

        return False, (
            "Invalid ticket data."
        )

    required_fields = [
        "ticket_title",
        "problem_description"
    ]

    for field in required_fields:

        value = record.get(field)

        if not isinstance(value, str):

            return False, (
                f"Missing or invalid field: {field}"
            )

        if not value.strip():

            return False, (
                f"Field cannot be empty: {field}"
            )

    # ------------------------------------------
    # LENGTH PROTECTION
    # ------------------------------------------

    if len(record["ticket_title"]) > 500:

        return False, (
            "Ticket title is too long."
        )

    if len(record["problem_description"]) > 10000:

        return False, (
            "Problem description is too long."
        )

    # ------------------------------------------
    # BASIC PROMPT-INJECTION CHECK
    # ------------------------------------------

    suspicious_phrases = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard previous instructions",
        "system prompt",
        "reveal your instructions",
        "show your prompt"
    ]

    combined_text = (
        record.get("ticket_title", "")
        + " "
        + record.get("problem_description", "")
    ).lower()

    for phrase in suspicious_phrases:

        if phrase in combined_text:

            return False, (
                "Ticket contains content that "
                "cannot be processed."
            )

    return True, None


# ==========================================
# CLASSIFICATION PIPELINE
# ==========================================

def classify_ticket(record):

    # ------------------------------------------
    # 1. SECURITY CHECK
    # ------------------------------------------

    security_valid, security_reason = (
        security_check(record)
    )

    if not security_valid:

        return {
            "status": "blocked",
            "reason": security_reason
        }

    # ------------------------------------------
    # 2. BUILD PROMPT
    # ------------------------------------------

    try:

        prompt = build_prompt(record)

    except Exception as error:

        logging.error(
            "Prompt creation failed: %s",
            error
        )

        return {
            "status": "failed",
            "reason": (
                "Unable to create AI "
                "classification prompt."
            )
        }

    # ------------------------------------------
    # 3. GEMINI API + PARSING + VALIDATION
    # ------------------------------------------

    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        try:

            response_text = call_api(
                prompt
            )

            parsed_response = parse_response(
                response_text
            )

            valid, validation_reason = (
                validate_response(
                    parsed_response
                )
            )

            if not valid:

                raise ValueError(
                    validation_reason
                )

            # ----------------------------------
            # SUCCESS
            # ----------------------------------

            return {
                "status": "success",
                "data": parsed_response
            }

        except Exception as error:

            last_error = error

            logging.error(
                "AI classification attempt %d/%d failed: %s",
                attempt + 1,
                MAX_RETRIES + 1,
                error
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

    # ------------------------------------------
    # ALL RETRIES FAILED
    # ------------------------------------------

    return {
        "status": "failed",
        "reason": (
            "AI classification failed after "
            f"{MAX_RETRIES + 1} attempts: "
            f"{last_error}"
        )
    }
