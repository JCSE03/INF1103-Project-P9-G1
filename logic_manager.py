
# logic_manager.py - business rules for the IT helpdesk.

# Takes the AI's classification of a ticket and decides:
# 1. its priority score
# 2. its priority (P1 = most urgent | P5 = least urgent)
# 3. which department handles it
# 4. whether a human should double-check the AI

# Also holds the rule for ordering a department's queue.

# Rules for this file: functions only (no classes), no print(),
# no file access, no API calls. Input in, output out.


# SCORING RUBRIC (change the numbers here, nowhere else)

# Points for how bad the problem is
SEVERITY_POINTS = {
    "low": 10,
    "medium": 25,
    "high": 40,
    "critical": 55,
}

# Points for how many people are affected
SCOPE_POINTS = {
    "individual": 0,
    "multiple_users": 10,
    "department": 20,
    "organisation": 30,
}

# Extra points for security issues
CYBERSECURITY_BONUS = 10

# Minimum score needed for each priority, most urgent first.
# Anything below the last one is P5.
PRIORITY_THRESHOLDS = [
    ("P1", 70),
    ("P2", 50),
    ("P3", 35),
    ("P4", 25),
]
LOWEST_PRIORITY = "P5"


# ROUTING (AI category -> department)
DEPARTMENTS = ["Cyber", "HR", "Finance", "Infra"]

ROUTING = {
    "cybersecurity": "Cyber",
    "hr": "HR",
    "finance": "Finance",
    "hardware": "Infra",
    "software": "Infra",
    "network": "Infra",
    "account_access": "Infra",
    "other": "Infra",
}

# REVIEW RULE
# Below this confidence, a human should check the AI's answer
LOW_CONFIDENCE_THRESHOLD = 0.6


# BUSINESS RULES (one rule per function)
def calculate_priority_score(severity, scope, category):
    """Add up the points for severity, scope and the cyber bonus."""
    score = SEVERITY_POINTS[severity] + SCOPE_POINTS[scope]
    if category == "cybersecurity":
        score += CYBERSECURITY_BONUS
    return score


def score_to_priority(score):
    """Turn a score into P1 ... P5 using PRIORITY_THRESHOLDS."""
    for priority, minimum_score in PRIORITY_THRESHOLDS:
        if score >= minimum_score:
            return priority
    return LOWEST_PRIORITY


def apply_cyber_override(priority, category, scope):
    """
    Multi-condition rule: a security issue affecting more than one
    person is always P1, whatever its score.
    """
    if category == "cybersecurity" and scope != "individual":
        return "P1"
    return priority


def get_department(category):
    """Look up which department handles this category."""
    return ROUTING[category]


def needs_human_review(confidence, ai_flagged):
    """A human checks the ticket if the AI is unsure or flagged it."""
    return confidence < LOW_CONFIDENCE_THRESHOLD or ai_flagged


def get_priority_number(priority):
    """'P1' -> 1, ... 'P5' -> 5 (easier to sort and filter on)."""
    return int(priority[1:])


# MAIN ENTRY POINT (io_manager calls this)
def process_ticket(ai_result):
    """
    Input:  what ai_manager.classify_ticket() returned, e.g.
            {"status": "success", "data": {...}, "flagged": False}
    Output: one decision dict, ready for data_manager to save.
    """
    ai_data = ai_result["data"]
    category = ai_data["category"]
    severity = ai_data["severity"]
    scope = ai_data["affected_scope"]
    confidence = ai_data["confidence_score"]

    score = calculate_priority_score(severity, scope, category)
    priority = score_to_priority(score)
    priority = apply_cyber_override(priority, category, scope)

    return {
        "priority": priority,
        "priority_number": get_priority_number(priority),
        "priority_score": score,
        "department": get_department(category),
        "needs_review": needs_human_review(confidence, ai_result.get("flagged", False)),
        "category": category,
        "severity": severity,
        "affected_scope": scope,
        "summary": ai_data["summary"],
        "confidence_score": confidence,
    }


# QUEUE ORDER
def queue_sort_key(ticket):
    """
    Order of a ticket in the queue:
        1. lower priority number first (P1 before P2)
        2. then higher score first
        3. then older ticket first
    """
    return (ticket["priority_number"], -ticket["priority_score"], ticket["submitted_at"])


def sort_queue(tickets):
    """Return a new list of tickets, most urgent first."""
    return sorted(tickets, key=queue_sort_key)