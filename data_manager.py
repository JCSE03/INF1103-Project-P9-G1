"""Data Manager: saves, loads and queries helpdesk tickets in data/tickets.json."""

import json
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "tickets.json")
SAMPLE_FILE = os.path.join(BASE_DIR, "sampleTickets.json")

SAVE_ATTEMPTS = 5
SAVE_RETRY_DELAY_SECONDS = 0.2


class StorageError(Exception):
    """Base class for every ticket storage problem."""


class DataReadError(StorageError):
    """tickets.json exists but is malformed or unreadable."""


class DataSaveError(StorageError):
    """tickets.json could not be written."""


class JsonFile:
    """One JSON file on disk."""

    def __init__(self, path: str, save_attempts: int = SAVE_ATTEMPTS,
                 retry_delay: float = SAVE_RETRY_DELAY_SECONDS) -> None:
        self._path = path
        self._save_attempts = save_attempts
        self._retry_delay = retry_delay

    @property
    def path(self) -> str:
        return self._path

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
        """Writes to a temp file first, then swaps it in."""
        temp_file = self._path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(temp_file, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)
            self._replace_with_retries(temp_file)
        except (OSError, TypeError, ValueError) as error:
            self._remove(temp_file)
            logging.error(f"Could not save {self._path}: {type(error).__name__}: {error}")
            raise DataSaveError(f"Could not save {self._path}: {error}") from error

    def _replace_with_retries(self, temp_file: str) -> None:
        for attempt in range(self._save_attempts):
            try:
                os.replace(temp_file, self._path)
                return
            except PermissionError:
                if attempt == self._save_attempts - 1:
                    raise
                time.sleep(self._retry_delay)

    @staticmethod
    def _remove(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass


class TicketRepository(ABC):
    """Abstract ticket storage contract."""

    @abstractmethod
    def load_all(self) -> list[dict]:
        """Every stored ticket. Raises DataReadError if the data is unreadable."""

    @abstractmethod
    def add(self, ticket: dict) -> None:
        """Stores a new ticket. Raises DataSaveError if it cannot be saved."""

    @abstractmethod
    def set_resolution(self, ticket_id: str, is_resolved: bool,
                       resolved_at: str | None = None) -> bool:
        """Marks a ticket resolved or reopens it. False if not found."""

    def find(self, field: str, value: Any) -> list[dict]:
        """Every ticket whose field equals value."""
        return [ticket for ticket in self.load_all() if ticket.get(field) == value]


class JsonTicketRepository(TicketRepository):
    """Tickets stored as a list of dicts in a JSON file."""

    def __init__(self, path: str = DATA_FILE, sample_path: str = SAMPLE_FILE) -> None:
        self._file = JsonFile(path)
        self._sample_path = sample_path

    @property
    def path(self) -> str:
        return self._file.path

    def load_all(self) -> list[dict]:
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
    records.pop()
    return False


def find_records(records: list[dict], field: str, value: Any) -> list[dict]:
    return [record for record in records if record.get(field) == value]


def set_resolution(records: list[dict], ticket_id: str, is_resolved: bool,
                   resolved_at: str | None = None) -> str:
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