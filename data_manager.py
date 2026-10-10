import json
import os

# Absolute path, so the file is found wherever the program is launched from.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tickets.json")


def load_records():
    """Return the saved records, [] if there is no file yet, None if unreadable."""
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, list):
            return None

        return records

    except (json.JSONDecodeError, OSError):
        return None


def save_records(records):
    """Write all records. Returns True on success, False on failure."""
    temp_file = DATA_FILE + ".tmp"

    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

        # Write to a temp file first, then swap it in, so a crash half-way
        # through cannot leave a corrupt tickets.json behind.
        with open(temp_file, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=4)
        os.replace(temp_file, DATA_FILE)

        return True

    except OSError:
        return False


def add_record(records, record):
    records.append(record)

    if save_records(records):
        return True

    records.pop()
    return False


def find_records(records, field, value):
    """Query: every record whose `field` equals `value`
    e.g. find_records(records, "department", "Cyber")."""
    return [record for record in records if record.get(field) == value]

def resolve_ticket(records, ticket_id, resolved_at_timestamp):
    """Find a ticket by ID, mark it resolved, and save."""
    for ticket in records:
        if ticket.get("ticket_id") == ticket_id:
            ticket["is_resolved"] = True
            ticket["resolved_at"] = resolved_at_timestamp
            return save_records(records)
    return False

def toggle_ticket_resolution(records, ticket_id, is_resolved, resolved_at_timestamp=None):
    """Toggle a ticket's resolution status and save."""
    for ticket in records:
        if ticket.get("ticket_id") == ticket_id:
            ticket["is_resolved"] = is_resolved
            ticket["resolved_at"] = resolved_at_timestamp if is_resolved else None
            return save_records(records)
    return False