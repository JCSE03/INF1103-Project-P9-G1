# logic_manager.py
"""
Logic Manager: Takes AI classification JSON, calculates priority,
assigns a department, and flags low-confidence tickets.
"""

# ------------------------------------------------------------
# MAPPING TABLES (text -> number)
# ------------------------------------------------------------
SEVERITY_MAP = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

SCOPE_MAP = {
    "individual": 1,
    "team": 2,
    "multiple_users": 2,     # alias for "team"
    "department": 3,
    "organisation": 4,
    "organization": 4,       # alias
}

# ------------------------------------------------------------
# DEPARTMENT ROUTING (category -> department)
# ------------------------------------------------------------
DEPARTMENT_MAP = {
    # Infra
    "network": "Infra",
    "infra": "Infra",
    "hardware": "Infra",
    "wifi": "Infra",
    "account_access": "Infra",   # login issues usually hit Infra

    # Cyber
    "security": "Cyber",
    "cyber": "Cyber",
    "malware": "Cyber",
    "phishing": "Cyber",

    # HR
    "hr": "HR",
    "payroll": "HR",
    "onboarding": "HR",

    # Finance
    "finance": "Finance",
    "billing": "Finance",
    "invoice": "Finance",
}

CONFIDENCE_THRESHOLD = 0.7


# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------
def calculate_priority(ai_response: dict) -> dict:
    """
    Takes the AI's JSON dict, returns an enriched dict with
    priority, department, and routing info.
    """

    # --- Extract values (with safe defaults) ---
    severity_text = str(ai_response.get("technical_severity", "low")).lower()
    scope_text    = str(ai_response.get("affected_scope", "individual")).lower()
    category      = str(ai_response.get("category", "unknown")).lower()
    confidence    = float(ai_response.get("confidence_score", 0.0))

    # --- Text -> Number ---
    severity_score = SEVERITY_MAP.get(severity_text, 1)
    scope_score    = SCOPE_MAP.get(scope_text, 1)

    # --- Priority Calculation ---
    priority_score = severity_score * scope_score

    # --- Map to P1-P5 ---
    if   15 <= priority_score <= 16: priority_label = "P1"
    elif 13 <= priority_score <= 14: priority_label = "P2"
    elif  8 <= priority_score <= 12: priority_label = "P3"
    elif  4 <= priority_score <=  7: priority_label = "P4"
    elif  1 <= priority_score <=  3: priority_label = "P5"
    else:                            priority_label = "UNKNOWN"

    # --- Assign Department ---
    department = DEPARTMENT_MAP.get(category, "Infra")  # default fallback

    # --- Confidence Check ---
    if confidence < CONFIDENCE_THRESHOLD:
        routing_message = (
            f"[{department}] Confidence score low! "
            f"({confidence:.2f}) Please reroute if needed."
        )
        needs_review = True
    else:
        routing_message = f"Auto-routed to {department}."
        needs_review = False

    # --- Build enriched response ---
    return {
        # Original AI fields (unchanged)
        "category":         ai_response.get("category"),
        "issue_type":       ai_response.get("issue_type"),
        "technical_severity": ai_response.get("technical_severity"),
        "affected_scope":   ai_response.get("affected_scope"),
        "summary":          ai_response.get("summary"),
        "confidence_score": confidence,

        # New fields from Logic Manager
        "priority":              priority_label,
        "priority_score":        priority_score,
        "assigned_department":   department,
        "routing_message":       routing_message,
        "requires_human_review": needs_review,

        # Debug info (great for troubleshooting)
        "debug": {
            "severity_score": severity_score,
            "scope_score":    scope_score,
        },
    }