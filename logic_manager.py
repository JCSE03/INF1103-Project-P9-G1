
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