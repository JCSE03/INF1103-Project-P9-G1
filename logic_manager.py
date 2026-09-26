

sample_ticket_1 = {
    "ticket_id": "TICKET-1103",
    "summary": "Core ATM network latency spike",
    "severity": "Critical",
    "affected_scope": "Organization",
    "confidence_score": 0.95,
    "department": "Infra"
}

severity_map = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

scope_map = {
    "individual": 1,
    "multiple people": 2,
    "multiple_people": 2,
    "department": 3,
    "organization": 4,
}

VALID_DEPARTMENTS = {"Cyber", "HR", "Finance", "Infra"}
CONFIDENCE_THRESHOLD = 0.5