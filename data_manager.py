import json
import os

DATA_FILE = "data/tickets.json"


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

def save_records(records):
    os.makedirs("data", exist_ok=True)

    try:
        with open(DATA_FILE, "w") as file:
            json.dump(records, file, indent=4)

        return True

    except OSError:
        return False


def add_record(records, record):
    records.append(record)

    if save_records(records):
        return True

    records.pop()
    return False