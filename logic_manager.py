"""
logic_manager.py - the "domain brain" of the IT helpdesk triage system.

ROLE IN THE PIPELINE
    io_manager -> ai_manager -> logic_manager -> data_manager

    ai_manager only CLASSIFIES a ticket (category, severity, scope, summary,
    confidence). It deliberately makes no business decisions.
    THIS module turns that classification into a business decision:
        - which department gets the ticket        (routing)
        - how urgent it is                        (priority P1-P4)
        - how fast it must be handled             (SLA)
        - whether management must be told         (escalation)
        - whether a human must double-check it    (review flag)
        - final outcome: routed / escalated / flagged / rejected

RULES OF THIS FILE
    - 100% procedural: functions only, NO classes (hard project constraint).
    - No print() and no input(): all display lives in io_manager.
    - No API calls and no import of ai_manager, so the tests can run
      without an API key or an internet connection.
    - Never crashes: bad or missing AI data is turned into a safe outcome.
    - Deterministic: the same input always gives the same decision.
"""

import re
from typing import Optional, Tuple


# ============================================================
# CONFIGURATION (business rules live here so they are easy to tune)
# ============================================================

# Which team handles which category.
# IMPORTANT: these keys must match ai_manager.VALID_CATEGORIES exactly.
ROUTING = {
    "hardware": "Desktop & Hardware Support",
    "software": "Application Support",
    "network": "Network Operations",
    "account_access": "Identity & Access Management",
    "cybersecurity": "Security Operations Centre (SOC)",
    "hr": "HR Systems Support",
    "finance": "Finance Systems Support",
    "other": "General IT Helpdesk",
}

# Fallback team if a category is ever unknown, or the AI was unavailable.
GENERAL_DEPARTMENT = ROUTING["other"]

# --- Priority scoring: every factor adds points, total is capped at 100 ---
SEVERITY_POINTS = {"low": 10, "medium": 20, "high": 30, "critical": 40}
SCOPE_POINTS = {
    "individual": 0,
    "multiple_users": 5,
    "department": 10,
    "organisation": 20,
}
# Scopes from smallest to largest. Used to compare two scopes with each other.
SCOPE_ORDER = ["individual", "multiple_users", "department", "organisation"]

WORK_BLOCKED_POINTS = 10       # user cannot do their job
NO_WORKAROUND_POINTS = 5       # and there is no way around it
CRITICAL_SERVICE_POINTS = 15   # a core banking service is seriously affected
CYBERSECURITY_POINTS = 15      # security incidents are weighted up
MAX_SCORE = 100

# Score -> priority. Checked from most to least urgent. AnytWhing lower is P4.
PRIORITY_THRESHOLDS = [("P1", 70), ("P2", 50), ("P3", 30)]
LOWEST_PRIORITY = "P4"

# Most urgent first. Used to compare priorities with each other.
PRIORITY_ORDER = ["P1", "P2", "P3", "P4"]

# Service Level Agreement targets for each priority.
SLA_TABLE = {
    "P1": {"response_minutes": 15, "resolution_hours": 4},
    "P2": {"response_minutes": 30, "resolution_hours": 8},
    "P3": {"response_minutes": 240, "resolution_hours": 24},
    "P4": {"response_minutes": 480, "resolution_hours": 72},
}

# Words that suggest a core banking / money-moving service is affected.
CRITICAL_SERVICE_KEYWORDS = [
    "core banking", "online banking", "mobile banking", "banking app",
    "payment", "swift", "atm", "trading", "settlement", "clearing",
]

# AI confidence below this value means "a human should double-check".
LOW_CONFIDENCE_THRESHOLD = 0.6

# If the AI is down we cannot assess a ticket. Err on the side of caution
# so a real outage is not buried: park it at P2 for manual triage.
AI_FAILURE_PRIORITY = "P2"

# Fields that must be present in a successful AI result.
REQUIRED_AI_FIELDS = [
    "category", "severity", "affected_scope", "summary", "confidence_score",
]


# ============================================================
# SMALL HELPERS (turn messy user input into clean values)
# ============================================================

def parse_yes_no(value) -> bool:
    """Turn a form answer such as 'Yes', 'y', 'true' or True into a bool."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("yes", "y", "true", "1")


def parse_reported_scope(value) -> Optional[str]:
    """
    Work out the scope the USER reported in the 'affected users' answer.
    The web form sends a scope word ("department"); other callers may send
    a number (5, "50+"). Both are understood:
        "department"  -> "department"
        1             -> "individual"
        2 or more     -> "multiple_users" (a count alone cannot prove a
                         whole department, so we do not guess higher)
    Returns None if the answer is empty or not understood.
    """
    text = str(value).strip().lower()
    if text in SCOPE_ORDER:
        return text
    match = re.search(r"\d+", text)
    if match is None:
        return None
    count = int(match.group())
    if count <= 0:
        return None
    return "individual" if count == 1 else "multiple_users"


def more_urgent(priority_a: str, priority_b: str) -> str:
    """Return whichever of two priorities (e.g. 'P2' and 'P1') is more urgent."""
    if PRIORITY_ORDER.index(priority_a) <= PRIORITY_ORDER.index(priority_b):
        return priority_a
    return priority_b


# ============================================================
# BUSINESS RULES
# ============================================================

def reconcile_scope(ai_scope: str, reported_scope: Optional[str]) -> Tuple[str, bool]:
    """
    MULTI-CONDITION RULE (AI output + user input).
    Cross-check the AI's scope against the scope the user reported.
    If the user reported a BIGGER scope than the AI picked, the AI has
    under-estimated the impact, so we raise it to the user's value.
    (We never lower it: under-triage is the dangerous direction.)

    Returns (final_scope, was_adjusted).
    """
    if reported_scope is None or reported_scope not in SCOPE_ORDER:
        return ai_scope, False
    if ai_scope not in SCOPE_ORDER:
        return reported_scope, True
    if SCOPE_ORDER.index(reported_scope) > SCOPE_ORDER.index(ai_scope):
        return reported_scope, True
    return ai_scope, False


def is_critical_service_hit(affected_service: str, summary: str, severity: str) -> bool:
    """
    MULTI-CONDITION RULE (AI output + user input).
    True only when BOTH are true:
        1. severity is high or critical, and
        2. the service or summary mentions a core banking keyword.
    A minor issue on a banking-related screen should not jump the queue.
    """
    if severity not in ("high", "critical"):
        return False
    text = f"{affected_service} {summary}".lower()
    return any(keyword in text for keyword in CRITICAL_SERVICE_KEYWORDS)


def calculate_priority_score(
    severity: str,
    scope: str,
    work_blocked: bool,
    workaround_available: bool,
    category: str,
    critical_service: bool,
) -> int:
    """Add up the points for every factor and cap the total at MAX_SCORE."""
    score = SEVERITY_POINTS.get(severity, 0)
    score += SCOPE_POINTS.get(scope, 0)

    if work_blocked:
        score += WORK_BLOCKED_POINTS
    if not workaround_available:
        score += NO_WORKAROUND_POINTS
    if critical_service:
        score += CRITICAL_SERVICE_POINTS
    if category == "cybersecurity":
        score += CYBERSECURITY_POINTS

    return min(score, MAX_SCORE)


def score_to_priority(score: int) -> str:
    """Convert a numeric score into P1 (most urgent) ... P4 (least urgent)."""
    for priority, minimum_score in PRIORITY_THRESHOLDS:
        if score >= minimum_score:
            return priority
    return LOWEST_PRIORITY


def apply_priority_overrides(
    priority: str, category: str, scope: str
) -> Tuple[str, Optional[str]]:
    """
    MULTI-CONDITION RULE (two AI output fields).
    Security incidents must never be under-prioritised, however low the
    raw score is:
        - cybersecurity affecting more than one user -> at least P1
        - any other cybersecurity ticket             -> at least P2

    Returns (new_priority, reason). reason is None if nothing changed.
    """
    if category != "cybersecurity":
        return priority, None

    if scope != "individual":
        minimum, reason = "P1", "Cybersecurity issue beyond one user: minimum P1"
    else:
        minimum, reason = "P2", "Cybersecurity issue: minimum P2"

    new_priority = more_urgent(priority, minimum)
    if new_priority == priority:
        return priority, None  # already urgent enough, no override needed
    return new_priority, reason


def check_escalation(priority: str, severity: str, scope: str) -> Tuple[bool, Optional[str]]:
    """
    MULTI-CONDITION RULE (two AI output fields).
    Decide if management / a major-incident process must be notified.
    Escalate when:
        - priority is P1, OR
        - severity is high/critical AND scope is department/organisation.

    Returns (should_escalate, reason).
    """
    if priority == "P1":
        return True, "Priority P1 requires escalation"
    if severity in ("high", "critical") and scope in ("department", "organisation"):
        return True, f"{severity} severity affecting {scope} requires escalation"
    return False, None


def check_needs_review(confidence: float, scope_adjusted: bool) -> Tuple[bool, Optional[str]]:
    """
    Flag the ticket for a human to double-check when the AI may be wrong:
        - confidence is below LOW_CONFIDENCE_THRESHOLD, OR
        - the AI's scope disagreed with what the user reported.
    """
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return True, f"Low AI confidence ({confidence:.2f}): human review needed"
    if scope_adjusted:
        return True, "AI scope was lower than the user reported: human review needed"
    return False, None


def decide_outcome(escalate: bool, needs_review: bool) -> str:
    """
    Pick the final outcome (urgency beats everything else):
        escalated -> management must be told
        flagged   -> a human must double-check before work starts
        routed    -> straight to the right team automatically
    """
    if escalate:
        return "escalated"
    if needs_review:
        return "flagged"
    return "routed"


def get_department(category: str) -> str:
    """Look up the team for a category; unknown categories go to the general desk."""
    return ROUTING.get(category, GENERAL_DEPARTMENT)


def get_sla(priority: str) -> dict:
    """Return a COPY of the SLA targets so callers cannot change the table."""
    return dict(SLA_TABLE[priority])


# ============================================================
# BUILDING THE RESULT (one consistent, flat dict for data_manager)
# ============================================================

def build_decision(
    record: dict,
    outcome: str,
    reasons: list,
    department: Optional[str] = None,
    priority: Optional[str] = None,
    priority_score: Optional[int] = None,
    escalate: bool = False,
    needs_review: bool = False,
    security_alert: bool = False,
    ai_data: Optional[dict] = None,
) -> dict:
    """
    Build the decision dict. EVERY decision has the same keys, whether the
    ticket was routed or rejected, so data_manager can write one CSV layout.
    Values are flat (text, numbers, True/False, None) so they save cleanly.
    """
    ai_data = ai_data or {}
    sla = get_sla(priority) if priority in SLA_TABLE else {}

    return {
        "ticket_title": record.get("ticket_title", ""),
        "outcome": outcome,
        "assigned_department": department,
        "priority": priority,
        "priority_score": priority_score,
        "sla_response_minutes": sla.get("response_minutes"),
        "sla_resolution_hours": sla.get("resolution_hours"),
        "escalate": escalate,
        "requires_human_review": needs_review,
        "security_alert": security_alert,
        "category": ai_data.get("category"),
        "severity": ai_data.get("severity"),
        "affected_scope": ai_data.get("affected_scope"),
        "summary": ai_data.get("summary"),
        "confidence_score": ai_data.get("confidence_score"),
        "decision_reasons": "; ".join(reasons),
    }


def reject_ticket(record: dict, ai_result: dict) -> dict:
    """
    ai_manager's security check refused the ticket ('invalid' or 'blocked').
    Outcome is 'rejected'. A 'blocked' ticket (attempt to get privileged
    credentials) also raises a security alert for later review.
    """
    reason = ai_result.get("reason", "Ticket rejected by security check.")
    is_blocked = ai_result.get("status") == "blocked"
    return build_decision(
        record,
        outcome="rejected",
        reasons=[reason],
        security_alert=is_blocked,
    )


def route_to_manual_triage(record: dict, reason: str) -> dict:
    """
    The AI could not classify this ticket (API down, bad response, missing
    fields). We never drop a real ticket: send it to the general desk at a
    cautious priority and flag it for a human.
    """
    return build_decision(
        record,
        outcome="flagged",
        reasons=[reason, f"Manual triage at provisional {AI_FAILURE_PRIORITY}"],
        department=GENERAL_DEPARTMENT,
        priority=AI_FAILURE_PRIORITY,
        needs_review=True,
    )


# ============================================================
# THE CORE DECISION PIPELINE
# ============================================================

def find_missing_ai_fields(ai_data) -> list:
    """List any required AI fields that are missing (or all of them if not a dict)."""
    if not isinstance(ai_data, dict):
        return list(REQUIRED_AI_FIELDS)
    return [field for field in REQUIRED_AI_FIELDS if field not in ai_data]


def score_and_route(record: dict, ai_data: dict) -> dict:
    """
    Apply every business rule to one AI-classified ticket, in order:
        1. cross-check scope   2. score   3. priority   4. overrides
        5. escalation          6. review  7. outcome    8. routing + SLA
    Each rule that fires adds a plain-English line to `reasons`, so every
    decision can be explained (useful for the report and the demo).
    """
    reasons = []

    category = ai_data["category"]
    severity = ai_data["severity"]
    confidence = float(ai_data["confidence_score"])

    # 1. Cross-check the AI's scope against the user's own numbers.
    reported_scope = parse_reported_scope(record.get("affected_users"))
    scope, scope_adjusted = reconcile_scope(ai_data["affected_scope"], reported_scope)
    if scope_adjusted:
        reasons.append(
            f"Scope raised from {ai_data['affected_scope']} to {scope} "
            f"(matches what the user reported)"
        )

    # 2. Gather the facts the user supplied on the form.
    work_blocked = parse_yes_no(record.get("work_blocked"))
    workaround = parse_yes_no(record.get("workaround_available"))
    critical_service = is_critical_service_hit(
        str(record.get("affected_service") or ""), ai_data["summary"], severity
    )
    if critical_service:
        reasons.append("Core banking service affected")

    # 3. Score the ticket and convert the score into a priority.
    score = calculate_priority_score(
        severity, scope, work_blocked, workaround, category, critical_service
    )
    priority = score_to_priority(score)
    reasons.append(f"Score {score} gives {priority}")

    # 4. Safety overrides (e.g. security issues are never low priority).
    priority, override_reason = apply_priority_overrides(priority, category, scope)
    if override_reason:
        reasons.append(override_reason)

    # 5. Should management be told?
    escalate, escalate_reason = check_escalation(priority, severity, scope)
    if escalate_reason:
        reasons.append(escalate_reason)

    # 6. Should a human double-check the AI?
    needs_review, review_reason = check_needs_review(confidence, scope_adjusted)
    if review_reason:
        reasons.append(review_reason)

    # 7 + 8. Final outcome, team and SLA.
    outcome = decide_outcome(escalate, needs_review)
    department = get_department(category)
    reasons.append(f"Routed to {department}")

    # Store the scope we actually used (it may differ from the AI's).
    final_ai_data = dict(ai_data)
    final_ai_data["affected_scope"] = scope

    return build_decision(
        record,
        outcome=outcome,
        reasons=reasons,
        department=department,
        priority=priority,
        priority_score=score,
        escalate=escalate,
        needs_review=needs_review,
        ai_data=final_ai_data,
    )


def process_ticket(record: dict, ai_result: dict) -> dict:
    """
    SINGLE ENTRY POINT for the rest of the program.

    Takes:
        record    - the validated form data from io_manager
        ai_result - what ai_manager.classify_ticket(record) returned, e.g.
                    {"status": "success", "data": {...}}
                    {"status": "invalid" | "blocked" | "error", "reason": "..."}

    Returns a flat decision dict (see build_decision). Never raises.
    """
    # Defensive: ai_result should always be a dict, but never trust it blindly.
    if not isinstance(ai_result, dict):
        return route_to_manual_triage(record, "No AI result received")

    status = ai_result.get("status")

    # The security check refused this ticket: reject it.
    if status in ("invalid", "blocked"):
        return reject_ticket(record, ai_result)

    # Anything other than success means the AI failed: manual triage.
    if status != "success":
        reason = ai_result.get("reason", "AI classification unavailable")
        return route_to_manual_triage(record, reason)

    # Success claimed, but make sure the fields we depend on are really there.
    ai_data = ai_result.get("data")
    missing = find_missing_ai_fields(ai_data)
    if missing:
        return route_to_manual_triage(
            record, f"AI result missing fields: {', '.join(missing)}"
        )

    return score_and_route(record, ai_data)