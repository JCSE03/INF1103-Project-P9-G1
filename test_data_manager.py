"""Offline tests for data_manager. Every test works in its own temporary
folder, so the real data/tickets.json is never touched.

Run:  python -m unittest test_data_manager   or   python main.py --test
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import data_manager
from data_manager import (DataReadError, DataSaveError, JsonTicketRepository,
                          StorageError, TicketRepository)


def make_ticket(number: int, department: str = "Infra") -> dict:
    return {"ticket_id": f"T-{number:04d}", "department": department,
            "is_resolved": False, "resolved_at": None}


class JsonTicketRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._temp.name, "data", "tickets.json")
        self.sample_path = os.path.join(self._temp.name, "sampleTickets.json")
        self.repository = JsonTicketRepository(self.path, self.sample_path)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def write(self, path: str, content) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(content if isinstance(content, str) else json.dumps(content))

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

    def test_json_repository_is_a_ticket_repository(self):
        self.assertIsInstance(self.repository, TicketRepository)

    def test_path_is_read_only(self):
        with self.assertRaises(AttributeError):
            self.repository.path = "somewhere/else.json"

    def test_errors_share_one_base_class(self):
        self.assertTrue(issubclass(DataReadError, StorageError))
        self.assertTrue(issubclass(DataSaveError, StorageError))


class CompatibilityLayerTests(unittest.TestCase):
    """The temporary Phase 1 functions keep their old return values."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        folder = self._temp.name
        self.path = os.path.join(folder, "data", "tickets.json")
        patches = [patch("data_manager.DATA_FILE", self.path),
                   patch("data_manager.SAMPLE_FILE", os.path.join(folder, "none.json"))]
        for active in patches:
            active.start()
            self.addCleanup(active.stop)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_load_missing_gives_empty_list(self):
        self.assertEqual(data_manager.load_records(), [])

    def test_load_corrupt_gives_none(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w", encoding="utf-8") as file:
            file.write("[{ broken")
        self.assertIsNone(data_manager.load_records())

    def test_add_and_find(self):
        records = []
        self.assertTrue(data_manager.add_record(records, make_ticket(1, "Cyber")))
        self.assertEqual(data_manager.find_records(data_manager.load_records(),
                                                   "department", "Cyber"), [make_ticket(1, "Cyber")])

    def test_failed_add_returns_false_and_undoes(self):
        records = [make_ticket(1)]
        with patch("data_manager.os.replace", side_effect=OSError("disk full")), \
                self.assertLogs(level="ERROR"):
            self.assertFalse(data_manager.add_record(records, make_ticket(2)))
        self.assertEqual(records, [make_ticket(1)])

    def test_set_resolution_statuses(self):
        records = [make_ticket(1)]
        self.assertEqual(data_manager.set_resolution(records, "T-0001", True, "now"), "updated")
        self.assertEqual(data_manager.set_resolution(records, "T-9999", True, "now"), "not_found")
        with patch("data_manager.os.replace", side_effect=OSError("disk full")), \
                self.assertLogs(level="ERROR"):
            self.assertEqual(data_manager.set_resolution(records, "T-0001", False), "save_failed")
        self.assertEqual((records[0]["is_resolved"], records[0]["resolved_at"]), (True, "now"))


def run_all_tests() -> int:
    """Runs every test here and returns the number of failures (for main.py)."""
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case)
        for case in (JsonTicketRepositoryTests, CompatibilityLayerTests))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return len(result.failures) + len(result.errors)


if __name__ == "__main__":
    unittest.main()
