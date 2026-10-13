import json
import logging
import os
import time

# Absolute path, so the file is found wherever the program is launched from.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tickets.json")

# On Windows, OneDrive sync or antivirus can lock tickets.json for a moment,
# so a locked file is retried a few times before the save is given up.
SAVE_ATTEMPTS = 5
SAVE_RETRY_DELAY_SECONDS = 0.2


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

        for attempt in range(SAVE_ATTEMPTS):
            try:
                os.replace(temp_file, DATA_FILE)
                return True
            except PermissionError:  # file briefly locked by another program
                if attempt == SAVE_ATTEMPTS - 1:
                    raise
                time.sleep(SAVE_RETRY_DELAY_SECONDS)

    except OSError as error:
        logging.error(f"Could not save {DATA_FILE}: {type(error).__name__}: {error}")
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


def set_resolution(records, ticket_id, is_resolved, resolved_at=None):
    """Mark a ticket resolved (with a timestamp) or reopen it, then save.

    Returns "updated", "not_found", or "save_failed". If saving fails the
    ticket is put back the way it was, so memory and file stay the same.
    """
    for ticket in records:
        if ticket.get("ticket_id") != ticket_id:
            continue

        before = (ticket.get("is_resolved", False), ticket.get("resolved_at"))
        ticket["is_resolved"] = is_resolved
        ticket["resolved_at"] = resolved_at if is_resolved else None

        if save_records(records):
            return "updated"

        ticket["is_resolved"], ticket["resolved_at"] = before
        return "save_failed"

    return "not_found"