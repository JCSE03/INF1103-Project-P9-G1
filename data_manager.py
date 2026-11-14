
import json
import logging
import os
import shutil
import sqlite3
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import Any

# Absolute paths, so the files are found wherever the program is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "tickets.json")
DATABASE_FILE = os.path.join(BASE_DIR, "data", "tickets.db")
SAMPLE_FILE = os.path.join(BASE_DIR, "sampleTickets.json")

# On Windows, OneDrive sync or antivirus can lock tickets.json for a moment,
# so a locked file is retried a few times before the save is given up.
SAVE_ATTEMPTS = 5
SAVE_RETRY_DELAY_SECONDS = 0.2

STORAGE_SETTING = "HTPRS_STORAGE"  # in .env: json (default) or sqlite
DATABASE_LOCK_TIMEOUT_SECONDS = 5


# --- errors -----------------------------------------------------------------

class StorageError(Exception):
    """Base class for every ticket storage problem, so callers can catch any
    of them with one except."""


class DataReadError(StorageError):
    """The stored tickets (tickets.json or tickets.db) exist but are malformed
    or unreadable. They must not be overwritten, as the data may still be
    recoverable."""


class DataSaveError(StorageError):
    """The tickets could not be saved. The stored data is left unchanged."""


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


# --- ticket storage: abstract class + JSON and SQLite implementations ------

class TicketRepository(ABC):
    """<<abstract>> What the rest of the program may do with stored tickets.
    A subclass decides how they are stored; code that uses a TicketRepository
    works with any of them. CRUD: add (create), load_all / get / find (read),
    set_resolution (update), delete (delete)."""

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

    @abstractmethod
    def delete(self, ticket_id: str) -> bool:
        """Removes one ticket. False if there is no such ticket."""

    def get(self, ticket_id: str) -> dict | None:
        """One ticket, or None. Shared; a subclass may override it with a faster lookup."""
        matches = self.find("ticket_id", ticket_id)
        return matches[0] if matches else None

    def find(self, field: str, value: Any) -> list[dict]:
        """Every ticket whose `field` equals `value`, e.g. find("department", "Cyber").
        Shared by every subclass."""
        return [ticket for ticket in self.load_all() if ticket.get(field) == value]


class JsonTicketRepository(TicketRepository):
    """Tickets as a list of dicts in one JSON file. Every change re-reads the
    file and writes the whole updated list back, so unchanged tickets are kept
    and a failed save leaves the file as it was.

    Created here (lives and dies with the repository): the JsonFile."""

    def __init__(self, path: str | None = None, sample_path: str | None = None) -> None:
        # None means the module setting, read when the object is created
        self._file = JsonFile(path or DATA_FILE)
        self._sample_path = sample_path or SAMPLE_FILE

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

    def delete(self, ticket_id: str) -> bool:
        records = self.load_all()
        remaining = [ticket for ticket in records if ticket.get("ticket_id") != ticket_id]
        if len(remaining) == len(records):
            return False
        self._file.write(remaining)
        return True


class SqliteTicketRepository(TicketRepository):
    """Tickets in a SQLite database, one row per ticket. The whole ticket is
    kept as JSON in `data`; ticket_id and department are also columns so they
    can be looked up directly. Unlike the JSON file, a change only touches its
    own row, and every change is a transaction: it is saved completely or not
    at all.

    A new database starts with the tickets already in tickets.json (so
    switching storage keeps them), or the sample tickets if there are none."""

    _SCHEMA = """CREATE TABLE IF NOT EXISTS tickets (
                     ticket_id  TEXT PRIMARY KEY,
                     department TEXT,
                     data       TEXT NOT NULL)"""
    _COLUMNS = ("ticket_id", "department")  # fields find() can query directly

    def __init__(self, path: str | None = None, sample_path: str | None = None,
                 json_path: str | None = None) -> None:
        self._path = path or DATABASE_FILE
        self._sample_path = sample_path or SAMPLE_FILE
        self._json_path = json_path or DATA_FILE
        self._lock_timeout = DATABASE_LOCK_TIMEOUT_SECONDS

    @property
    def path(self) -> str:
        return self._path

    def load_all(self) -> list[dict]:
        with self._connection(DataReadError) as conn:
            rows = conn.execute("SELECT data FROM tickets ORDER BY rowid").fetchall()
        return [self._decode(data) for (data,) in rows]

    def get(self, ticket_id: str) -> dict | None:
        # Overrides the shared version with a direct lookup.
        with self._connection(DataReadError) as conn:
            row = conn.execute("SELECT data FROM tickets WHERE ticket_id = ?",
                               (ticket_id,)).fetchone()
        return self._decode(row[0]) if row else None

    def find(self, field: str, value: Any) -> list[dict]:
        # Overrides the shared version: indexed columns are queried directly.
        if field not in self._COLUMNS:
            return super().find(field, value)
        # field was checked against _COLUMNS, so it is safe inside the SQL text
        with self._connection(DataReadError) as conn:
            rows = conn.execute(f"SELECT data FROM tickets WHERE {field} = ? ORDER BY rowid",
                                (value,)).fetchall()
        return [self._decode(data) for (data,) in rows]

    def add(self, ticket: dict) -> None:
        with self._connection(DataSaveError) as conn:
            self._insert(conn, ticket)

    def set_resolution(self, ticket_id: str, is_resolved: bool,
                       resolved_at: str | None = None) -> bool:
        with self._connection(DataSaveError) as conn:
            row = conn.execute("SELECT data FROM tickets WHERE ticket_id = ?",
                               (ticket_id,)).fetchone()
            if row is None:
                return False
            ticket = self._decode(row[0])
            ticket["is_resolved"] = is_resolved
            ticket["resolved_at"] = resolved_at if is_resolved else None
            conn.execute("UPDATE tickets SET data = ? WHERE ticket_id = ?",
                         (json.dumps(ticket), ticket_id))
        return True

    def delete(self, ticket_id: str) -> bool:
        with self._connection(DataSaveError) as conn:
            cursor = conn.execute("DELETE FROM tickets WHERE ticket_id = ?", (ticket_id,))
        return cursor.rowcount > 0

    @contextmanager
    def _connection(self, error_class: type[StorageError]) -> Iterator[sqlite3.Connection]:
        """One short-lived connection per operation (Flask serves requests on
        separate threads). Commits on success and rolls back on any error.
        sqlite3 errors become error_class, except a corrupt database, which is
        always DataReadError so it is never written to."""
        is_new = not os.path.exists(self._path)
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with closing(sqlite3.connect(self._path, timeout=self._lock_timeout)) as conn:
                with conn:  # one transaction
                    conn.execute(self._SCHEMA)
                    if is_new:
                        for ticket in self._starting_tickets():
                            self._insert(conn, ticket)
                    yield conn
        except StorageError:
            raise
        except sqlite3.IntegrityError as error:  # e.g. a ticket_id that already exists
            raise DataSaveError(f"Ticket rejected by the database: {error}") from error
        except (sqlite3.OperationalError, OSError) as error:  # locked, cannot open, disk full
            logging.error(f"Database error on {self._path}: {error}")
            raise error_class(f"Database error on {self._path}: {error}") from error
        except sqlite3.DatabaseError as error:  # "file is not a database"
            logging.error(f"Database {self._path} is corrupt: {error}")
            raise DataReadError(f"Database {self._path} is corrupt: {error}") from error

    def _starting_tickets(self) -> list[dict]:
        # Existing tickets.json first, else the samples; a missing or broken
        # file is skipped, so the database may start empty.
        for path in (self._json_path, self._sample_path):
            try:
                tickets = JsonFile(path).read()
            except DataReadError:
                continue
            if isinstance(tickets, list):
                return [ticket for ticket in tickets if isinstance(ticket, dict)]
        return []

    @staticmethod
    def _insert(conn: sqlite3.Connection, ticket: dict) -> None:
        conn.execute("INSERT INTO tickets (ticket_id, department, data) VALUES (?, ?, ?)",
                     (ticket.get("ticket_id"), ticket.get("department"), json.dumps(ticket)))

    @staticmethod
    def _decode(data: str) -> dict:
        try:
            ticket = json.loads(data)
        except json.JSONDecodeError as error:
            raise DataReadError(f"A stored ticket is not valid JSON: {error}") from error
        if not isinstance(ticket, dict):
            raise DataReadError("A stored ticket is not a JSON object.")
        return ticket


# --- choosing the storage (Factory pattern) ---------------------------------

class RepositoryFactory:
    """The one place that decides which TicketRepository to build, from the
    HTPRS_STORAGE setting. Callers only know TicketRepository; a new storage
    type is one new class plus one entry in _REPOSITORIES."""

    _REPOSITORIES: dict[str, type[TicketRepository]] = {
        "json": JsonTicketRepository,
        "sqlite": SqliteTicketRepository,
    }

    @classmethod
    def create(cls, storage: str | None = None, **options: Any) -> TicketRepository:
        name = (storage or os.getenv(STORAGE_SETTING) or "json").strip().lower()
        if name not in cls._REPOSITORIES:
            raise ValueError(f"Unknown storage {name!r}; "
                             f"choose one of {', '.join(cls._REPOSITORIES)}.")
        return cls._REPOSITORIES[name](**options)


# --- TEMPORARY compatibility layer ------------------------------------------
# io_manager.py calls these Phase 1 names and expects their old return values
# (None / False / status strings) instead of exceptions. They go through
# RepositoryFactory, so the current io_manager already uses whichever storage
# HTPRS_STORAGE names. Delete once io_manager uses the repository directly.

def load_records() -> list[dict] | None:
    try:
        return RepositoryFactory.create().load_all()
    except DataReadError:
        return None


def add_record(records: list[dict], record: dict) -> bool:
    try:
        RepositoryFactory.create().add(record)
    except StorageError:
        return False
    records.append(record)  # keep the caller's list in step with storage
    return True


def find_records(records: list[dict], field: str, value: Any) -> list[dict]:
    return [record for record in records if record.get(field) == value]


def set_resolution(records: list[dict], ticket_id: str, is_resolved: bool,
                   resolved_at: str | None = None) -> str:
    """Returns "updated", "not_found" or "save_failed"."""
    try:
        found = RepositoryFactory.create().set_resolution(ticket_id, is_resolved, resolved_at)
    except StorageError:
        return "save_failed"
    if not found:
        return "not_found"
    for ticket in records:  # keep the caller's list in step with storage
        if ticket.get("ticket_id") == ticket_id:
            ticket["is_resolved"] = is_resolved
            ticket["resolved_at"] = resolved_at if is_resolved else None
    return "updated"
