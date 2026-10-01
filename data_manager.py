import json
import os

DATA_FILE = "data/tickets.json"


# Load Tickets
def load_records():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r") as file:
            records = json.load(file)

        if not isinstance(records, list):
            return None

        return records

    except (json.JSONDecodeError, OSError):
        return None


# Save Tickets
def save_records(records):
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

        text = json.dumps(records, indent=4)

        with open(DATA_FILE, "w") as file:
            file.write(text)

        return True

    except (OSError, TypeError):
        return False


# Generate Ticket ID
def generate_ticket_id(records):
    highest_number = 0

    for record in records:
        ticket_id = record.get("ticket_id", "")

        # Ignore invalid ticket IDs
        if not isinstance(ticket_id, str):
            continue

        if ticket_id.startswith("TICKET-"):
            try:
                number = int(ticket_id.split("-")[1])

                if number > highest_number:
                    highest_number = number

            except ValueError:
                continue

    next_number = highest_number + 1

    return f"TICKET-{next_number:04d}"


# Add Ticket
def add_record(records, record):
    if records is None:
        return False

    record["ticket_id"] = generate_ticket_id(records)

    records.append(record)

    if save_records(records):
        return True

    # Remove ticket if saving fails
    records.pop()
    return False


# Find Ticket
def find_by_ticket_id(records, ticket_id):
    if records is None:
        return None

    for record in records:
        if record.get("ticket_id") == ticket_id:
            return record

    return None


# Filter by Category
def filter_by_category(records, category):
    if records is None:
        return []

    results = []

    for record in records:
        if record.get("category") == category:
            results.append(record)

    return results


# Filter by Department
def filter_by_department(records, department):
    if records is None:
        return []

    results = []

    for record in records:
        if record.get("department") == department:
            results.append(record)

    return results


# Update Ticket Status
def update_status(records, ticket_id, new_status):
    if records is None:
        return False

    for record in records:

        if record.get("ticket_id") == ticket_id:

            old_status = record.get("status")
            record["status"] = new_status

            if save_records(records):
                return True

            # Restore old status if saving fails
            if old_status is None:
                record.pop("status", None)
            else:
                record["status"] = old_status

            return False

    return False


# Get Active Ticket Queue
def get_active_tickets(records):
    if records is None:
        return []

    results = []

    for record in records:
        if record.get("status") != "resolved":
            results.append(record)

    return results