"""Data Manager: saves, loads and queries helpdesk tickets in data/tickets.json.

Classes: StorageError -> DataReadError / DataSaveError (errors the IO Manager
catches), JsonFile (reads and writes one JSON file), TicketRepository
(<<abstract>> ticket storage) and JsonTicketRepository (the JSON implementation,
which owns a JsonFile).

Other managers use it only through the abstract type, e.g.
    repository: TicketRepository = JsonTicketRepository()

The module-level functions at the bottom are a TEMPORARY compatibility layer
for io_manager.py and the existing tests; new code should use the classes.
"""
import json
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from typing import Any

# Absolute paths, so the files are found wherever the program is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "tickets.json")
SAMPLE_FILE = os.path.join(BASE_DIR, "sampleTickets.json")

# On Windows, OneDrive sync or antivirus can lock tickets.json for a moment,
# so a locked file is retried a few times before the save is given up.
SAVE_ATTEMPTS = 5
SAVE_RETRY_DELAY_SECONDS = 0.2


# --- errors -----------------------------------------------------------------

class StorageError(Exception):
    """Base class for every ticket storage problem, so callers can catch any
    of them with one except."""


class DataReadError(StorageError):
    """tickets.json exists but is malformed or unreadable. It must not be
    overwritten, as the data may still be recoverable."""


class DataSaveError(StorageError):
    """tickets.json could not be written. The old file is left unchanged."""


# --- JSON file (owned by JsonTicketRepository) ------------------------------

class JsonFile:
    """One JSON file on disk. Knows about files only, not about tickets."""

    def __init__(self, path: str, save_attempts: int = SAVE_ATTEMPTS,
                 retry_delay: float = SAVE_RETRY_DELAY_SECONDS) -> None:
        self._path = path
        self._save_attempts = save_attempts
        self._retry_delay = retry_delay

    @property
    def path(self) -> str:
        return self._path  # read-only: no setter

    def exists(self) -> bool:
        return os.path.exists(self._path)

    def copy_from(self, source_path: str) -> bool:
        """Creates this file as a copy of another. False if that is not possible."""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            shutil.copyfile(source_path, self._path)
        except OSError:
            return False
        return True

    def read(self) -> Any:
        try:
            with open(self._path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError) as error:
            raise DataReadError(f"Could not read {self._path}: {error}") from error

    def write(self, data: Any) -> None:
        """Writes to a temp file first, then swaps it in, so a crash half-way
        through cannot leave a corrupt file behind."""
        temp_file = self._path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(temp_file, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)
            self._replace_with_retries(temp_file)
        except (OSError, TypeError, ValueError) as error:  # TypeError/ValueError: not JSON-serialisable
            self._remove(temp_file)
            logging.error(f"Could not save {self._path}: {type(error).__name__}: {error}")
            raise DataSaveError(f"Could not save {self._path}: {error}") from error

    def _replace_with_retries(self, temp_file: str) -> None:
        for attempt in range(self._save_attempts):
            try:
                os.replace(temp_file, self._path)
                return
            except PermissionError:  # file briefly locked by another program
                if attempt == self._save_attempts - 1:
                    raise
                time.sleep(self._retry_delay)

    @staticmethod
    def _remove(path: str) -> None:
        # Clean up after a failed save; the real file was never touched.
        try:
            os.remove(path)
        except OSError:
            pass


# --- ticket storage: abstract class + JSON implementation -------------------

class TicketRepository(ABC):
    """<<abstract>> What the rest of the program may do with stored tickets.
    A subclass decides how they are stored; code that uses a TicketRepository
    works with any of them."""

    @abstractmethod
    def load_all(self) -> list[dict]:
        """Every stored ticket. Raises DataReadError if the data is unreadable."""

    @abstractmethod
    def add(self, ticket: dict) -> None:
        """Stores a new ticket. Raises DataSaveError if it cannot be saved."""

    @abstractmethod
    def set_resolution(self, ticket_id: str, is_resolved: bool,
                       resolved_at: str | None = None) -> bool:
        """Marks a ticket resolved (with a timestamp) or reopens it.
        False if there is no such ticket."""

    def find(self, field: str, value: Any) -> list[dict]:
        """Every ticket whose `field` equals `value`, e.g. find("department", "Cyber").
        Shared by every subclass."""
        return [ticket for ticket in self.load_all() if ticket.get(field) == value]


class JsonTicketRepository(TicketRepository):
    """Tickets as a list of dicts in one JSON file. Every change re-reads the
    file and writes the whole updated list back, so unchanged tickets are kept
    and a failed save leaves the file as it was.

    Created here (lives and dies with the repository): the JsonFile."""

    def __init__(self, path: str = DATA_FILE, sample_path: str = SAMPLE_FILE) -> None:
        self._file = JsonFile(path)
        self._sample_path = sample_path

    @property
    def path(self) -> str:
        return self._file.path

    def load_all(self) -> list[dict]:
        # A missing tickets.json starts as a copy of the sample tickets.
        if not self._file.exists() and not self._file.copy_from(self._sample_path):
            return []

        records = self._file.read()
        if not isinstance(records, list) or not all(isinstance(ticket, dict) for ticket in records):
            raise DataReadError(f"{self._file.path} does not contain a list of tickets.")
        return records

    def add(self, ticket: dict) -> None:
        records = self.load_all()
        records.append(ticket)
        self._file.write(records)

    def set_resolution(self, ticket_id: str, is_resolved: bool,
                       resolved_at: str | None = None) -> bool:
        records = self.load_all()
        for ticket in records:
            if ticket.get("ticket_id") == ticket_id:
                ticket["is_resolved"] = is_resolved
                ticket["resolved_at"] = resolved_at if is_resolved else None
                self._file.write(records)
                return True
        return False


# --- TEMPORARY compatibility layer ------------------------------------------
# io_manager.py and the existing tests call (and patch) these Phase 1 names and
# expect their old return values (None / False / status strings) instead of
# exceptions. Each builds the classes from the CURRENT module values, so
# patches of DATA_FILE, SAMPLE_FILE etc. keep working. Delete once io_manager
# uses JsonTicketRepository directly.

def load_records() -> list[dict] | None:
    try:
        return JsonTicketRepository(DATA_FILE, SAMPLE_FILE).load_all()
    except DataReadError:
        return None


def save_records(records: list[dict]) -> bool:
    try:
        JsonFile(DATA_FILE, SAVE_ATTEMPTS, SAVE_RETRY_DELAY_SECONDS).write(records)
    except DataSaveError:
        return False
    return True


def add_record(records: list[dict], record: dict) -> bool:
    records.append(record)
    if save_records(records):
        return True
    records.pop()  # undo, so memory matches the file
    return False


def find_records(records: list[dict], field: str, value: Any) -> list[dict]:
    return [record for record in records if record.get(field) == value]


def set_resolution(records: list[dict], ticket_id: str, is_resolved: bool,
                   resolved_at: str | None = None) -> str:
    """Returns "updated", "not_found" or "save_failed"; a failed save puts the
    ticket back the way it was."""
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
