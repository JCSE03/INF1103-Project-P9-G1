# test_analytics.py - offline tests for analytics.py and the /analytics routes

# Run with:  python test_analytics.py
# No API key or internet needed, and a temporary folder is used for the
# routes, so your real data/tickets.json is never touched.

import json
import logging
import os
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# io_manager imports ai_manager, which refuses to load without a key.
os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")

import analytics  # noqa: E402
import data_manager  # noqa: E402
import io_manager  # noqa: E402
import logic_manager  # noqa: E402
from analytics import (  # noqa: E402
    AnalyticsReport, ConfidenceBands, DepartmentWorkload, Metric, Overview,
    OverdueTickets, PriorityBreakdown, ResolutionTimes, SlaPolicy, SubmissionTrend,
)


NOW = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)   # the fixed "now" of the tests


def run(metric, tickets):
    """A metric's result at the fixed time NOW."""
    return metric.compute(tickets, NOW)


def open_for(hours, **fields):
    """An open ticket that was submitted `hours` before NOW."""
    return ticket(submitted_at=(NOW - timedelta(hours=hours)).isoformat(), **fields)


def ticket(**fields):
    base = {"ticket_id": "T-1", "submitted_at": "2026-10-12T01:00:00+00:00",
            "priority": "P3", "department": "Infra", "category": "software",
            "needs_review": False, "confidence_score": 0.9,
            "is_resolved": False, "resolved_at": None}
    return {**base, **fields}


def resolved(hours, priority="P3", start="2026-10-12T01:00:00+00:00"):
    end = datetime.fromisoformat(start).timestamp() + hours * 3600
    return ticket(priority=priority, submitted_at=start, is_resolved=True,
                  resolved_at=datetime.fromtimestamp(end, timezone.utc).isoformat())


@contextmanager
def data_file(content):
    """A Flask test client whose tickets.json is a temporary file."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "tickets.json")
        with open(path, "w", encoding="utf-8") as file:
            file.write(content if isinstance(content, str) else json.dumps(content))
        with patch.object(data_manager, "DATA_FILE", path), \
                patch.object(data_manager, "SAMPLE_FILE", os.path.join(folder, "none.json")):
            yield io_manager.app.test_client()


# ---------- the metrics ----------

def test_overview_counts():
    tickets = [ticket(priority="P1"),                                   # urgent, open
               ticket(priority="P2"),                                   # urgent, open
               ticket(needs_review=True, confidence_score=0.5),         # open, needs review
               ticket(priority="P1", is_resolved=True, needs_review=True),  # resolved: counts for neither
               ticket(is_resolved=True)]
    result = run(Overview(), tickets)
    assert (result["total"], result["open"], result["resolved"]) == (5, 3, 2)
    assert result["resolution_rate"] == 0.4
    assert result["urgent_open"] == 2 and result["needs_review_open"] == 1
    assert result["average_confidence"] == 0.82


def test_priority_breakdown_keeps_all_priorities_in_order():
    rows = run(PriorityBreakdown(), [ticket(priority="P1"), ticket(priority="P1", is_resolved=True),
                                        ticket(priority="weird"), ticket(priority=None)])
    assert [r["label"] for r in rows] == ["P1", "P2", "P3", "P4", "P5", "unknown", "weird"]
    assert rows[0] == {"label": "P1", "open": 1, "resolved": 1}
    assert rows[1] == {"label": "P2", "open": 0, "resolved": 0}      # empty ones still shown


def test_department_breakdown_is_busiest_first():
    rows = run(DepartmentWorkload(), [ticket(department="HR"), ticket(department="Cyber"),
                                         ticket(department="Cyber"), ticket(department="Cyber", is_resolved=True)])
    assert [(r["label"], r["open"], r["resolved"]) for r in rows] == [("Cyber", 2, 1), ("HR", 1, 0)]


def test_submission_trend_fills_gaps_and_caps_the_range():
    days = run(SubmissionTrend(), [ticket(submitted_at="2026-10-01T10:00:00+00:00"),
                                      ticket(submitted_at="2026-10-03T10:00:00+00:00"),
                                      ticket(submitted_at="2026-10-03T11:00:00+00:00")])
    assert days == [{"label": "2026-10-01", "count": 1}, {"label": "2026-10-02", "count": 0},
                    {"label": "2026-10-03", "count": 2}]

    long = run(SubmissionTrend(), [ticket(submitted_at="2026-08-01T00:00:00+00:00"),
                                      ticket(submitted_at="2026-10-12T00:00:00+00:00")])
    assert len(long) == analytics.MAX_TREND_DAYS and long[-1]["label"] == "2026-10-12"

    # other time zones are converted to UTC; a time without a zone counts as UTC
    shifted = run(SubmissionTrend(), [ticket(submitted_at="2026-10-12T23:30:00-05:00"),
                                         ticket(submitted_at="2026-10-13T08:00:00")])
    assert shifted == [{"label": "2026-10-13", "count": 2}]
    assert run(SubmissionTrend(), []) == []
    assert Metric.when({"t": "2026-10-13T08:00:00"}, "t").utcoffset().total_seconds() == 0
    assert Metric.when({"t": "nonsense"}, "t") is None and Metric.when({}, "t") is None


def test_resolution_times():
    tickets = [resolved(2, "P1"), resolved(4, "P1"), resolved(10, "P3"),
               ticket(resolved_at="2026-10-12T05:00:00+00:00"),                     # not resolved
               ticket(is_resolved=True, resolved_at="2026-10-11T01:00:00+00:00"),   # ends before it starts
               ticket(is_resolved=True, resolved_at=None)]                          # no end time
    result = run(ResolutionTimes(), tickets)
    assert result["counted"] == 3 and result["average"] == 5.3 and result["median"] == 4.0
    assert result["by_priority"] == [{"label": "P1", "average": 3.0}, {"label": "P3", "average": 10.0}]
    assert run(ResolutionTimes(), [ticket()]) == {"counted": 0, "average": None,
                                                     "median": None, "by_priority": []}


def test_confidence_bands():
    scores = [0.5, 0.6, 0.79, 0.8, 0.95, 1.0, 1.5, -0.1, "high", True, None, float("nan")]
    bands = run(ConfidenceBands(), [ticket(confidence_score=s) for s in scores])
    assert [b["count"] for b in bands] == [1, 2, 1, 2]       # impossible scores are left out
    assert [b["label"] for b in bands][0] == "under 60%"


# ---------- target times and overdue tickets ----------

DEFAULT_TARGETS = {"P1": 4, "P2": 8, "P3": 24, "P4": 72, "P5": 120}


def test_sla_policy_defaults_and_overrides():
    policy = SlaPolicy()
    assert policy.as_dict() == DEFAULT_TARGETS and policy.problems == []
    custom = SlaPolicy({"P1": 2, "P3": 36.5})
    assert (custom.hours_for("P1"), custom.hours_for("P3"), custom.hours_for("P2")) == (2, 36.5, 8)

    # a bad value falls back to the default and is reported, one by one
    bad = SlaPolicy({"P1": -1, "P2": 0, "P3": "soon", "P4": True, "P5": float("nan"), "P9": 5})
    assert bad.as_dict() == DEFAULT_TARGETS and len(bad.problems) == 6
    assert policy.hours_for("P9") is None and policy.hours_for(["P1"]) is None
    assert policy.hours_for(None) is None

    for odd in (float("nan"), float("inf")):       # an infinite target would hide every overdue ticket
        assert len(SlaPolicy({"P5": odd}).problems) == 1 and SlaPolicy({"P5": odd}).hours_for("P5") == 120

    policy.as_dict()["P1"] = 99          # copies only: it cannot be changed from outside
    bad.problems.append("x")
    assert policy.hours_for("P1") == 4 and len(bad.problems) == 6


def test_sla_policy_loads_from_a_file_and_survives_bad_files():
    with tempfile.TemporaryDirectory() as folder:
        def write(text):
            path = os.path.join(folder, "sla.json")
            with open(path, "w", encoding="utf-8") as file:
                file.write(text)
            return path

        missing = SlaPolicy.load(os.path.join(folder, "missing.json"))
        assert missing.as_dict() == DEFAULT_TARGETS and missing.problems == []   # no file is fine
        assert SlaPolicy.load(write('{"P1": 1, "P2": 2}')).hours_for("P2") == 2
        for broken in ("{not json", "[1, 2]", ""):
            policy = SlaPolicy.load(write(broken))
            assert policy.as_dict() == DEFAULT_TARGETS and policy.problems, broken
        mixed = SlaPolicy.load(write('{"P1": 1, "P2": "x"}'))
        assert (mixed.hours_for("P1"), mixed.hours_for("P2"), len(mixed.problems)) == (1, 8, 1)


def test_overdue_tickets_are_found_and_ordered():
    tickets = [open_for(5, ticket_id="T-A", priority="P1"),                 # 5 h of 4: overdue
               open_for(3.5, ticket_id="T-B", priority="P1"),               # 88% used: due soon
               open_for(2, ticket_id="T-C", priority="P1"),                 # 50% used: fine
               open_for(30, ticket_id="T-D", priority="P3"),                # 30 h of 24: overdue
               open_for(100, ticket_id="T-E", priority="P1", is_resolved=True),   # resolved: ignored
               open_for(100, ticket_id="T-F", priority="weird"),            # no target for it: ignored
               ticket(ticket_id="T-G", priority="P1", submitted_at=None),   # no time: ignored
               open_for(-3, ticket_id="T-H", priority="P1")]                # in the future: ignored
    result = run(OverdueTickets(SlaPolicy()), tickets)
    assert (result["overdue"], result["due_soon"]) == (2, 1)
    assert [(t["ticket_id"], t["status"]) for t in result["tickets"]] == [
        ("T-A", "overdue"), ("T-D", "overdue"), ("T-B", "due_soon")]   # overdue first, equal 125% by id
    assert result["tickets"][0] == {"ticket_id": "T-A", "title": "(no title)", "priority": "P1",
                                    "department": "Infra", "hours_open": 5.0, "limit_hours": 4,
                                    "status": "overdue"}
    assert result["targets"] == DEFAULT_TARGETS and result["policy_problems"] == []


def test_overdue_boundaries_and_limits():
    status = lambda hours: run(OverdueTickets(SlaPolicy()), [open_for(hours, priority="P1")])["tickets"]
    assert status(2.9) == []                                  # under 75% of the target
    assert status(3)[0]["status"] == "due_soon"               # exactly 75%
    assert status(4)[0]["status"] == "due_soon"               # exactly at the target is not late yet
    assert status(4.1)[0]["status"] == "overdue"

    many = [open_for(5 + i, ticket_id=f"T-{i:02d}", priority="P1") for i in range(15)]
    result = run(OverdueTickets(SlaPolicy()), many)
    assert result["overdue"] == 15 and len(result["tickets"]) == analytics.MAX_AT_RISK_LISTED
    assert result["tickets"][0]["ticket_id"] == "T-14"        # the worst one first

    private = open_for(9, priority="P1", name="Alice Tan", contact_number="91234567",
                       ticket_title="x" * 200)
    row = run(OverdueTickets(SlaPolicy()), [private])
    assert "Alice" not in json.dumps(row) and "91234567" not in json.dumps(row)
    assert len(row["tickets"][0]["title"]) == 80


def test_target_times_can_be_changed():
    strict = OverdueTickets(SlaPolicy({"P1": 1}))
    result = run(strict, [open_for(2, priority="P1")])
    assert result["overdue"] == 1 and result["targets"]["P1"] == 1
    report = AnalyticsReport(sla_policy=SlaPolicy({"P3": 1})).build([open_for(2)], now=NOW)
    assert report["at_risk"]["overdue"] == 1
    naive = datetime(2026, 10, 20, 12)                         # no time zone: counted as UTC
    report = AnalyticsReport(sla_policy=SlaPolicy()).build([open_for(5, priority="P1")], now=naive)
    assert report["at_risk"]["overdue"] == 1 and report["generated_at"].endswith("+00:00")


def test_default_report_reads_sla_policy_json():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "sla_policy.json")
        with open(path, "w", encoding="utf-8") as file:
            file.write('{"P3": 1}')
        with patch.object(analytics, "SLA_FILE", path):
            report = AnalyticsReport().build([open_for(2)], now=NOW)       # a P3 ticket, 2 h old
        assert report["at_risk"]["overdue"] == 1 and report["at_risk"]["targets"]["P3"] == 1
        with patch.object(analytics, "SLA_FILE", os.path.join(folder, "missing.json")):
            report = AnalyticsReport().build([open_for(2)], now=NOW)       # no file: defaults
        assert report["at_risk"]["overdue"] == 0 and report["at_risk"]["targets"]["P3"] == 24


# ---------- the report ----------

def test_report_has_every_section():
    report = AnalyticsReport().build([ticket()], now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert report["generated_at"] == "2026-11-01T00:00:00+00:00"
    for key in ("overview", "by_priority", "by_department", "by_category", "per_day",
                "resolution_hours", "ai_confidence", "at_risk"):
        assert report[key] is not None, key
    assert report["failed_metrics"] == [] and report["ignored_records"] == 0


def test_empty_data():
    report = AnalyticsReport().build([])
    assert report["overview"]["total"] == 0 and report["overview"]["resolution_rate"] == 0
    assert report["overview"]["average_confidence"] is None
    assert report["per_day"] == [] and report["resolution_hours"]["counted"] == 0
    assert [r["label"] for r in report["by_priority"]] == ["P1", "P2", "P3", "P4", "P5"]
    assert report["at_risk"]["overdue"] == 0 and report["at_risk"]["tickets"] == []


def test_messy_records_never_crash_and_stay_json_safe():
    records = [None, "text", 42, [], {},
               {"priority": ["P1"], "department": 5, "confidence_score": "high",
                "submitted_at": 12345, "is_resolved": "yes", "needs_review": "no"},
               {"submitted_at": "not a date", "confidence_score": float("inf"), "priority": {"a": 1}}]
    report = AnalyticsReport().build(records)
    assert report["ignored_records"] == 4 and report["overview"]["total"] == 3
    assert report["failed_metrics"] == []
    json.dumps(report, allow_nan=False)        # no NaN/Infinity: the browser could not read it


def test_report_matches_the_sample_file():
    with open(data_manager.SAMPLE_FILE, encoding="utf-8") as file:
        records = json.load(file)
    report = AnalyticsReport().build(records)
    assert report["overview"]["total"] == len(records)
    for key, field in (("by_priority", "priority"), ("by_department", "department"),
                       ("by_category", "category")):
        expected = Counter(r[field] for r in records)
        assert {r["label"]: r["open"] + r["resolved"] for r in report[key] if r["open"] + r["resolved"]} == dict(expected), key
    assert sum(day["count"] for day in report["per_day"]) == len(records)


def test_report_is_open_for_new_metrics_and_isolates_failures():
    try:
        Metric()
        raise AssertionError("Metric should be abstract")
    except TypeError:
        pass

    class Urgent(Metric):
        key = "urgent"

        def compute(self, tickets, now):
            return sum(t.get("priority") == "P1" for t in tickets)

    class Broken(Metric):
        key = "broken"

        def compute(self, tickets, now):
            raise ValueError("secret-ticket-text")

    report = AnalyticsReport([Urgent()]).build([ticket(priority="P1"), ticket()])
    assert report["urgent"] == 1 and "overview" not in report      # metrics are injected

    with patch.object(logging, "error") as logged:
        report = AnalyticsReport([Broken(), Urgent()]).build([ticket(priority="P1")])
    assert report["broken"] is None and report["failed_metrics"] == ["broken"]
    assert report["urgent"] == 1                                    # the others still ran
    assert "secret-ticket-text" not in logged.call_args.args[0]     # no ticket data in the log


# ---------- the routes ----------

def test_api_returns_a_report_and_filters_by_department():
    records = [ticket(department="Infra"), ticket(department="Infra", is_resolved=True),
               ticket(department="HR")]
    with data_file(records) as client:
        data = client.get("/api/analytics").get_json()
        assert data["status"] == "success" and data["report"]["overview"]["total"] == 3
        assert "at_risk" in data["report"]
        hr = client.get("/api/analytics?department=HR").get_json()["report"]
        assert hr["overview"]["total"] == 1 and hr["by_department"][0]["label"] == "HR"


def test_api_rejects_unknown_department_and_unreadable_data():
    with data_file([ticket()]) as client:
        response = client.get("/api/analytics?department=Nope")
        assert response.status_code == 400 and response.get_json()["status"] == "error"
    with data_file("{this is not json") as client:
        response = client.get("/api/analytics")
        assert response.status_code == 503 and response.get_json()["status"] == "error"


def test_pages_render_and_link_to_each_other():
    with data_file([ticket()]) as client:
        page = client.get("/analytics")
        assert page.status_code == 200 and b"IT Helpdesk Analytics" in page.data
        for department in logic_manager.DEPARTMENTS:
            assert department.encode() in page.data
        assert b'href="/analytics"' in client.get("/helpdesk").data


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