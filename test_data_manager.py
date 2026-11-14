"""Offline tests for data_manager. Every test works in its own temporary
folder, so the real data/tickets.json is never touched.

Run:  python -m unittest test_data_manager   or   python main.py --test
"""
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

import data_manager
from data_manager import (DataReadError, DataSaveError, JsonTicketRepository,
                          RepositoryFactory, SqliteTicketRepository, StorageError,
                          TicketRepository)


def make_ticket(number: int, department: str = "Infra") -> dict:
    return {"ticket_id": f"T-{number:04d}", "department": department,
            "is_resolved": False, "resolved_at": None}


class TempFolderTest(unittest.TestCase):
    """Gives each test an empty temporary folder."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.folder = self._temp.name
        self.sample_path = os.path.join(self.folder, "sampleTickets.json")
        self.json_path = os.path.join(self.folder, "data", "tickets.json")
        self.db_path = os.path.join(self.folder, "data", "tickets.db")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def write(self, path: str, content) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(content if isinstance(content, str) else json.dumps(content))


class RepositoryContractTests:
    """Mixed into a TestCase whose setUp builds self.repository."""

    repository: TicketRepository

    def make_repository(self) -> TicketRepository:
        raise NotImplementedError

    def test_contract_new_store_starts_from_sample(self):
        self.write(self.sample_path, [make_ticket(1), make_ticket(2)])
        ids = [ticket["ticket_id"] for ticket in self.repository.load_all()]
        self.assertEqual(ids, ["T-0001", "T-0002"])

    def test_contract_no_sample_gives_empty_store(self):
        self.assertEqual(self.repository.load_all(), [])

    def test_contract_added_ticket_survives_restart(self):
        self.repository.add(make_ticket(1))
        self.assertEqual(self.make_repository().load_all(), [make_ticket(1)])

    def test_contract_add_keeps_existing_tickets(self):
        self.write(self.sample_path, [make_ticket(1)])
        self.repository.add(make_ticket(2))
        self.assertEqual(len(self.repository.load_all()), 2)

    def test_contract_get(self):
        self.write(self.sample_path, [make_ticket(1, "HR"), make_ticket(2)])
        self.assertEqual(self.repository.get("T-0001")["department"], "HR")
        self.assertIsNone(self.repository.get("T-9999"))

    def test_contract_find_by_department(self):
        self.write(self.sample_path, [make_ticket(1, "Cyber"), make_ticket(2, "HR"),
                                      make_ticket(3, "Cyber")])
        found = self.repository.find("department", "Cyber")
        self.assertEqual([ticket["ticket_id"] for ticket in found], ["T-0001", "T-0003"])

    def test_contract_find_by_other_field(self):
        self.write(self.sample_path, [make_ticket(1), {**make_ticket(2), "is_resolved": True}])
        found = self.repository.find("is_resolved", True)
        self.assertEqual([ticket["ticket_id"] for ticket in found], ["T-0002"])

    def test_contract_resolve_then_reopen(self):
        self.write(self.sample_path, [make_ticket(1)])
        self.assertTrue(self.repository.set_resolution("T-0001", True, "2026-02-01T10:00:00+00:00"))
        ticket = self.make_repository().get("T-0001")
        self.assertEqual((ticket["is_resolved"], ticket["resolved_at"]),
                         (True, "2026-02-01T10:00:00+00:00"))
        self.assertTrue(self.repository.set_resolution("T-0001", False, "ignored"))
        ticket = self.make_repository().get("T-0001")
        self.assertEqual((ticket["is_resolved"], ticket["resolved_at"]), (False, None))

    def test_contract_resolve_unknown_ticket(self):
        self.assertFalse(self.repository.set_resolution("T-9999", True, "now"))

    def test_contract_delete(self):
        self.write(self.sample_path, [make_ticket(1), make_ticket(2)])
        self.assertTrue(self.repository.delete("T-0001"))
        self.assertFalse(self.repository.delete("T-0001"))
        ids = [ticket["ticket_id"] for ticket in self.make_repository().load_all()]
        self.assertEqual(ids, ["T-0002"])

    def test_contract_is_a_ticket_repository(self):
        self.assertIsInstance(self.repository, TicketRepository)


class JsonTicketRepositoryTests(RepositoryContractTests, TempFolderTest):
    def setUp(self) -> None:
        super().setUp()
        self.path = self.json_path
        self.repository = self.make_repository()

    def make_repository(self) -> JsonTicketRepository:
        return JsonTicketRepository(self.path, self.sample_path)

    def read_store(self) -> str:
        with open(self.path, encoding="utf-8") as file:
            return file.read()

    # --- loading ---
    def test_missing_file_is_created_from_sample(self):
        self.write(self.sample_path, [make_ticket(1), make_ticket(2)])
        self.assertEqual(len(self.repository.load_all()), 2)
        self.assertTrue(os.path.exists(self.path))

    def test_missing_file_and_sample_gives_empty_list(self):
        self.assertEqual(self.repository.load_all(), [])

    def test_existing_file_is_loaded_not_reset(self):
        self.write(self.sample_path, [make_ticket(1)])
        self.write(self.path, [make_ticket(5), make_ticket(6)])
        ids = [ticket["ticket_id"] for ticket in self.repository.load_all()]
        self.assertEqual(ids, ["T-0005", "T-0006"])

    def test_corrupt_json_raises(self):
        self.write(self.path, "[{ broken")
        with self.assertRaises(DataReadError):
            self.repository.load_all()

    def test_top_level_not_a_list_raises(self):
        self.write(self.path, {"ticket_id": "T-0001"})
        with self.assertRaises(DataReadError):
            self.repository.load_all()

    def test_list_of_non_tickets_raises(self):
        self.write(self.path, [1, 2, 3])
        with self.assertRaises(DataReadError):
            self.repository.load_all()

    # --- adding ---
    def test_added_ticket_survives_restart(self):
        self.repository.add(make_ticket(1))
        reloaded = JsonTicketRepository(self.path, self.sample_path)  # a separate run
        self.assertEqual(reloaded.load_all(), [make_ticket(1)])

    def test_add_keeps_existing_tickets(self):
        self.write(self.path, [make_ticket(1)])
        self.repository.add(make_ticket(2))
        self.assertEqual(len(self.repository.load_all()), 2)

    def test_add_to_corrupt_file_does_not_overwrite_it(self):
        self.write(self.path, "[{ broken")
        with self.assertRaises(DataReadError):
            self.repository.add(make_ticket(1))
        self.assertEqual(self.read_store(), "[{ broken")

    def test_failed_add_leaves_file_unchanged(self):
        self.write(self.path, [make_ticket(1)])
        before = self.read_store()
        with patch("data_manager.os.replace", side_effect=OSError("disk full")), \
                self.assertLogs(level="ERROR"):
            with self.assertRaises(DataSaveError):
                self.repository.add(make_ticket(2))
        self.assertEqual(self.read_store(), before)
        self.assertFalse(os.path.exists(self.path + ".tmp"))  # temp file cleaned up

    def test_unserialisable_ticket_raises_save_error(self):
        self.write(self.path, [make_ticket(1)])
        before = self.read_store()
        with self.assertRaises(DataSaveError), self.assertLogs(level="ERROR"):
            self.repository.add({**make_ticket(2), "submitted_at": object()})
        self.assertEqual(self.read_store(), before)
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    # --- saving: locked file retries ---
    @patch("data_manager.time.sleep")
    def test_locked_file_is_retried(self, sleep):
        self.write(self.path, [])
        real_replace = os.replace
        attempts = []

        def locked_once(src, dst):
            attempts.append(dst)
            if len(attempts) == 1:
                raise PermissionError("locked by OneDrive")
            real_replace(src, dst)

        with patch("data_manager.os.replace", side_effect=locked_once):
            self.repository.add(make_ticket(1))
        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(data_manager.SAVE_RETRY_DELAY_SECONDS)
        self.assertEqual(len(self.repository.load_all()), 1)

    @patch("data_manager.time.sleep")
    def test_save_gives_up_after_all_attempts(self, _sleep):
        self.write(self.path, [])
        with patch("data_manager.os.replace", side_effect=PermissionError("locked")) as replace:
            with self.assertRaises(DataSaveError), self.assertLogs(level="ERROR"):
                self.repository.add(make_ticket(1))
        self.assertEqual(replace.call_count, data_manager.SAVE_ATTEMPTS)

    # --- finding ---
    def test_find_by_department(self):
        self.write(self.path, [make_ticket(1, "Cyber"), make_ticket(2, "HR"),
                               make_ticket(3, "Cyber")])
        found = self.repository.find("department", "Cyber")
        self.assertEqual([ticket["ticket_id"] for ticket in found], ["T-0001", "T-0003"])

    def test_find_with_no_match(self):
        self.write(self.path, [make_ticket(1, "Cyber")])
        self.assertEqual(self.repository.find("department", "Finance"), [])

    # --- resolving ---
    def test_resolve_sets_timestamp(self):
        self.write(self.path, [make_ticket(1)])
        self.assertTrue(self.repository.set_resolution("T-0001", True, "2026-02-01T10:00:00+00:00"))
        ticket = self.repository.load_all()[0]
        self.assertEqual((ticket["is_resolved"], ticket["resolved_at"]),
                         (True, "2026-02-01T10:00:00+00:00"))

    def test_reopen_clears_timestamp(self):
        self.write(self.path, [{**make_ticket(1), "is_resolved": True, "resolved_at": "earlier"}])
        self.assertTrue(self.repository.set_resolution("T-0001", False, "ignored"))
        ticket = self.repository.load_all()[0]
        self.assertEqual((ticket["is_resolved"], ticket["resolved_at"]), (False, None))

    def test_resolve_unknown_ticket(self):
        self.write(self.path, [make_ticket(1)])
        self.assertFalse(self.repository.set_resolution("T-9999", True, "now"))

    def test_failed_resolve_leaves_file_unchanged(self):
        self.write(self.path, [make_ticket(1)])
        before = self.read_store()
        with patch("data_manager.os.replace", side_effect=OSError("disk full")), \
                self.assertLogs(level="ERROR"):
            with self.assertRaises(DataSaveError):
                self.repository.set_resolution("T-0001", True, "now")
        self.assertEqual(self.read_store(), before)

    # --- design ---
    def test_abstract_base_cannot_be_created(self):
        with self.assertRaises(TypeError):
            TicketRepository()

    def test_path_is_read_only(self):
        with self.assertRaises(AttributeError):
            self.repository.path = "somewhere/else.json"

    def test_errors_share_one_base_class(self):
        self.assertTrue(issubclass(DataReadError, StorageError))
        self.assertTrue(issubclass(DataSaveError, StorageError))


class SqliteTicketRepositoryTests(RepositoryContractTests, TempFolderTest):
    def setUp(self) -> None:
        super().setUp()
        self.repository = self.make_repository()

    def make_repository(self) -> SqliteTicketRepository:
        return SqliteTicketRepository(self.db_path, self.sample_path, self.json_path)

    def test_new_database_takes_existing_json_tickets(self):
        self.write(self.sample_path, [make_ticket(1)])
        self.write(self.json_path, [make_ticket(7), make_ticket(8)])
        ids = [ticket["ticket_id"] for ticket in self.repository.load_all()]
        self.assertEqual(ids, ["T-0007", "T-0008"])  # not the samples

    def test_starting_tickets_are_loaded_only_once(self):
        self.write(self.sample_path, [make_ticket(1)])
        self.repository.delete("T-0001")
        self.assertEqual(self.make_repository().load_all(), [])

    def test_broken_starting_files_give_empty_store(self):
        self.write(self.json_path, "{ not json")
        self.write(self.sample_path, "{ not json")
        self.assertEqual(self.repository.load_all(), [])

    def test_duplicate_ticket_id_raises_save_error(self):
        self.repository.add(make_ticket(1))
        with self.assertRaises(DataSaveError):
            self.repository.add(make_ticket(1))
        self.assertEqual(len(self.repository.load_all()), 1)

    def test_corrupt_database_raises_and_is_never_written(self):
        self.write(self.db_path, "this is not a database" * 100)
        with open(self.db_path, "rb") as file:
            before = file.read()
        with self.assertRaises(DataReadError), self.assertLogs(level="ERROR"):
            self.repository.load_all()
        with self.assertRaises(DataReadError), self.assertLogs(level="ERROR"):
            self.repository.add(make_ticket(1))
        with open(self.db_path, "rb") as file:
            self.assertEqual(file.read(), before)

    def test_corrupt_row_raises(self):
        self.repository.add(make_ticket(1))
        with closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute("UPDATE tickets SET data = '{ broken'")
        with self.assertRaises(DataReadError):
            self.repository.load_all()

    def test_locked_database_raises_save_error(self):
        self.repository.add(make_ticket(1))
        self.repository._lock_timeout = 0
        blocker = sqlite3.connect(self.db_path)
        try:
            blocker.execute("BEGIN EXCLUSIVE")
            with self.assertRaises(DataSaveError), self.assertLogs(level="ERROR"):
                self.repository.add(make_ticket(2))
        finally:
            blocker.close()
        self.assertEqual(len(self.repository.load_all()), 1)  # rolled back


class RepositoryFactoryTests(TempFolderTest):
    def test_json_is_the_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(RepositoryFactory.create(), JsonTicketRepository)

    def test_setting_selects_sqlite(self):
        with patch.dict(os.environ, {"HTPRS_STORAGE": " SQLite "}):
            self.assertIsInstance(RepositoryFactory.create(), SqliteTicketRepository)

    def test_unknown_storage_raises(self):
        with self.assertRaises(ValueError):
            RepositoryFactory.create("excel")

    def test_options_are_passed_on(self):
        repository = RepositoryFactory.create("sqlite", path=self.db_path)
        self.assertEqual(repository.path, self.db_path)


class CompatibilityLayerTests:
    """Mixed into a TempFolderTest: the temporary Phase 1 functions keep their
    old return values, for whichever storage HTPRS_STORAGE names."""

    storage = "json"

    def setUp(self) -> None:
        super().setUp()
        for active in (patch("data_manager.DATA_FILE", self.json_path),
                       patch("data_manager.DATABASE_FILE", self.db_path),
                       patch("data_manager.SAMPLE_FILE", self.sample_path),
                       patch.dict(os.environ, {"HTPRS_STORAGE": self.storage})):
            active.start()
            self.addCleanup(active.stop)

    def test_load_missing_gives_empty_list(self):
        self.assertEqual(data_manager.load_records(), [])

    def test_add_find_and_resolve(self):
        records = []
        self.assertTrue(data_manager.add_record(records, make_ticket(1, "Cyber")))
        self.assertEqual(records, [make_ticket(1, "Cyber")])
        stored = data_manager.load_records()
        self.assertEqual(data_manager.find_records(stored, "department", "Cyber"),
                         [make_ticket(1, "Cyber")])
        self.assertEqual(data_manager.set_resolution(stored, "T-0001", True, "now"), "updated")
        self.assertEqual(stored[0]["resolved_at"], "now")
        self.assertEqual(data_manager.set_resolution(stored, "T-9999", True, "now"), "not_found")


class JsonCompatibilityTests(CompatibilityLayerTests, TempFolderTest):
    def test_load_corrupt_gives_none(self):
        self.write(self.json_path, "[{ broken")
        self.assertIsNone(data_manager.load_records())

    def test_failed_saves_return_false_and_save_failed(self):
        records = [make_ticket(1)]
        self.write(self.json_path, records)
        with patch("data_manager.os.replace", side_effect=OSError("disk full")), \
                self.assertLogs(level="ERROR"):
            self.assertFalse(data_manager.add_record(records, make_ticket(2)))
            self.assertEqual(data_manager.set_resolution(records, "T-0001", True), "save_failed")
        self.assertEqual(records, [make_ticket(1)])  # caller's list unchanged


class SqliteCompatibilityTests(CompatibilityLayerTests, TempFolderTest):
    storage = "sqlite"

    def test_tickets_go_to_the_database(self):
        data_manager.add_record([], make_ticket(1))
        self.assertTrue(os.path.exists(self.db_path))
        self.assertFalse(os.path.exists(self.json_path))


def run_all_tests() -> int:
    """Runs every test here and returns the number of failures (for main.py)."""
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case)
        for case in (JsonTicketRepositoryTests, SqliteTicketRepositoryTests,
                     RepositoryFactoryTests, JsonCompatibilityTests, SqliteCompatibilityTests))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return len(result.failures) + len(result.errors)


if __name__ == "__main__":
    unittest.main()
