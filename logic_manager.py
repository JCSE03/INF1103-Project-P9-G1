CATEGORY_DEPARTMENTS = {
    "cybersecurity": "Cyber",
    "hr": "HR",
    "finance": "Finance",
    "hardware": "Infra",
    "software": "Infra",
    "network": "Infra",
    "account_access": "Infra",
    "other": "Infra",
}

VALID_DEPARTMENTS = tuple(sorted(set(CATEGORY_DEPARTMENTS.values())))


def route_ticket(classification):
    """Add operational routing details to a validated AI classification."""
    category = classification.get("category", "other")
    department = CATEGORY_DEPARTMENTS.get(category, "Infra")

    return {
        "department": department,
        "queue_status": "Queued",
    }