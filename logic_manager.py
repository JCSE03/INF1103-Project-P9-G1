
from typing import Dict, Any, Tuple, List

sample_ticket_1 = {
    "ticket_id": "TICKET-1103",
    "summary": "Core ATM network latency spike",
    "severity": "Critical",
    "affected_scope": "Organization",
    "confidence_score": 0.95,
    "department": "Infra"
}

SEVERITY_MAP = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

SCOPE_MAP = {
    "individual": 1,
    "multiple people": 2,
    "multiple_people": 2, # In case the AI manager spits out a spacing or underscore
    "department": 3,
    "organization": 4,
}

VALID_DEPARTMENTS = {"Cyber", "HR", "Finance", "Infra"}
CONFIDENCE_THRESHOLD = 0.5


def _normalize_severity(val: Any) -> int:
    # 1. If it's already an integer (1 to 4), use it as is
    if isinstance(val, int) and 1 <= val <= 4:
        return val
    
    # 2. Clean up text: convert to string, remove spaces, make lowercase
    cleaned_val = str(val).strip().lower()
    
    # 3. Look up the score; if not found, safely fallback to 1 (Low)
    return SEVERITY_MAP.get(cleaned_val, 1)

def _normalize_scope(val: Any) -> int:
    if isinstance(val, int) and 1 <= val <= 4:
        return val
    return SCOPE_MAP.get(str(val).strip().lower(), 1)

def score(record: Dict[str, Any]) -> Tuple[int, str]:
    """
    Produces a numeric priority score (1-16) and priority tier (P1-P5).
    Formula: Severity Score * Affected Scope Score = Priority Score
    """
    sev_score = _normalize_severity(record.get("severity", 1))
    scope_score = _normalize_scope(record.get("affected_scope", 1))
    
    priority_score = sev_score * scope_score
    if 15 <= priority_score <= 16:
        priority_level = "P1"
    elif 13 <= priority_score <= 14:
        priority_level = "P2"
    elif 8 <= priority_score <= 12:
        priority_level = "P3"
    elif 4 <= priority_score <= 7:
        priority_level = "P4"
    else:
        priority_level = "P5"
    return priority_score, priority_level