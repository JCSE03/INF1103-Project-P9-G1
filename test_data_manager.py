# test_data_manager.py - offline tests for data_manager.py

# Run with:  python test_data_manager.py
# Uses a temporary folder, so your real data/tickets.json is never touched.

import os
import sys
import tempfile
from unittest.mock import patch

import data_manager as dm


def temp_data_file(folder, name="tickets.json"):
    # Point both files at the temp folder. There is no sample there, so a
    # missing tickets.json still starts empty unless a test creates one.
    return patch.multiple(dm, DATA_FILE=os.path.join(folder, name),
                          SAMPLE_FILE=os.path.join(folder, "sampleTickets.json"))


# ---------- load ----------

def test_missing_file_gives_empty_list():
    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder):
        assert dm.load_records() == []


def test_missing_file_is_seeded_from_sample():
    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder):
        with open(dm.SAMPLE_FILE, "w") as file:
            file.write('[{"ticket_id": "T-0001"}]')
        assert dm.load_records() == [{"ticket_id": "T-0001"}]   # copied from the sample
        assert os.path.exists(dm.DATA_FILE)                     # tickets.json now exists


def test_existing_file_is_not_replaced_by_sample():
    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder):
        with open(dm.SAMPLE_FILE, "w") as file:
            file.write('[{"ticket_id": "SAMPLE"}]')
        dm.save_records([{"ticket_id": "REAL"}])
        assert dm.load_records() == [{"ticket_id": "REAL"}]


def test_corrupt_file_gives_none():
    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder):
        with open(dm.DATA_FILE, "w") as file:
            file.write("{ this is not json")
        assert dm.load_records() is None


def test_wrong_shape_gives_none():
    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder):
        with open(dm.DATA_FILE, "w") as file:
            file.write('{"not": "a list"}')
        assert dm.load_records() is None


# ---------- save / add ----------

def test_add_then_load_round_trip():
    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder):
        records = dm.load_records()
        assert dm.add_record(records, {"ticket_id": "T-0001"}) is True
        assert dm.load_records() == [{"ticket_id": "T-0001"}]


def test_save_creates_missing_folder():
    with tempfile.TemporaryDirectory() as folder:
        with temp_data_file(os.path.join(folder, "new_folder")):
            assert dm.save_records([{"ticket_id": "T-0001"}]) is True
            assert dm.load_records() == [{"ticket_id": "T-0001"}]


def test_failed_save_returns_false_and_undoes_the_add():
    with tempfile.TemporaryDirectory() as folder:
        blocker = os.path.join(folder, "blocker")
        with open(blocker, "w") as file:      # a FILE where a folder is needed
            file.write("x")
        with temp_data_file(blocker):
            records = [{"ticket_id": "T-0001"}]
            assert dm.add_record(records, {"ticket_id": "T-0002"}) is False
            assert records == [{"ticket_id": "T-0001"}]


def test_no_temp_file_left_behind():
    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder):
        dm.save_records([{"ticket_id": "T-0001"}])
        assert os.listdir(folder) == ["tickets.json"]


def test_locked_file_is_retried():
    # Windows/OneDrive can lock tickets.json for a moment: the save retries.
    real_replace = os.replace
    calls = []

    def locked_once(src, dst):
        calls.append(dst)
        if len(calls) == 1:
            raise PermissionError("file is being used by another process")
        real_replace(src, dst)

    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder), \
            patch.object(dm, "SAVE_RETRY_DELAY_SECONDS", 0), \
            patch.object(dm.os, "replace", locked_once):
        assert dm.save_records([{"ticket_id": "T-0001"}]) is True
        assert len(calls) == 2
        assert dm.load_records() == [{"ticket_id": "T-0001"}]


def test_file_locked_too_long_gives_up():
    def always_locked(src, dst):
        raise PermissionError("still locked")

    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder), \
            patch.object(dm, "SAVE_RETRY_DELAY_SECONDS", 0), \
            patch.object(dm.os, "replace", always_locked):
        assert dm.save_records([{"ticket_id": "T-0001"}]) is False


# ---------- query ----------

def test_find_records_filters_by_field():
    records = [
        {"ticket_id": "A", "department": "Cyber", "needs_review": False},
        {"ticket_id": "B", "department": "Infra", "needs_review": True},
        {"ticket_id": "C", "department": "Cyber", "needs_review": True},
    ]
    assert [r["ticket_id"] for r in dm.find_records(records, "department", "Cyber")] == ["A", "C"]
    assert [r["ticket_id"] for r in dm.find_records(records, "needs_review", True)] == ["B", "C"]
    assert dm.find_records(records, "department", "HR") == []


def test_find_records_ignores_records_missing_the_field():
    assert dm.find_records([{"ticket_id": "A"}], "department", "Cyber") == []


# ---------- resolve / reopen ----------

def test_resolve_then_reopen_is_saved():
    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder):
        records = [{"ticket_id": "T-0001"}, {"ticket_id": "T-0002"}]
        dm.save_records(records)

        assert dm.set_resolution(records, "T-0002", True, "2026-10-13T05:00:00+00:00") == "updated"
        saved = dm.load_records()
        assert saved[1]["is_resolved"] is True
        assert saved[1]["resolved_at"] == "2026-10-13T05:00:00+00:00"
        assert "is_resolved" not in saved[0]          # other tickets untouched

        assert dm.set_resolution(records, "T-0002", False, "ignored") == "updated"
        saved = dm.load_records()
        assert saved[1]["is_resolved"] is False
        assert saved[1]["resolved_at"] is None       # reopening clears the time


def test_resolve_unknown_ticket_is_not_found():
    with tempfile.TemporaryDirectory() as folder, temp_data_file(folder):
        records = [{"ticket_id": "T-0001"}]
        assert dm.set_resolution(records, "T-9999", True, "now") == "not_found"
        assert records == [{"ticket_id": "T-0001"}]
        assert not os.path.exists(dm.DATA_FILE)      # nothing written


def test_failed_resolve_save_is_undone():
    with tempfile.TemporaryDirectory() as folder:
        blocker = os.path.join(folder, "blocker")
        with open(blocker, "w") as file:      # a FILE where a folder is needed
            file.write("x")
        with temp_data_file(os.path.join(blocker, "tickets.json")):
            records = [{"ticket_id": "T-0001", "is_resolved": False, "resolved_at": None}]
            assert dm.set_resolution(records, "T-0001", True, "now") == "save_failed"
            assert records == [{"ticket_id": "T-0001", "is_resolved": False, "resolved_at": None}]


# ---------- test runner ----------

def run_all_tests():
    """Run every test_ function, print PASS/FAIL, return the failure count."""
    tests = [(name, f) for name, f in sorted(globals().items())
             if name.startswith("test_") and callable(f)]
    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception as error:
            failures += 1
            print(f"FAIL  {name}  {error!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run_all_tests() else 0)