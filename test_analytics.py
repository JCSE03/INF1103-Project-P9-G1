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
from datetime import datetime, timezone
from unittest.mock import patch

# io_manager imports ai_manager, which refuses to load without a key.
os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")

import analytics  # noqa: E402
import data_manager  # noqa: E402
import io_manager  # noqa: E402
import logic_manager  # noqa: E402
from analytics import (  # noqa: E402
    AnalyticsReport, ConfidenceBands, DepartmentWorkload, Metric, Overview,
    PriorityBreakdown, ResolutionTimes, SubmissionTrend,
)


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
    result = Overview().compute(tickets)
    assert (result["total"], result["open"], result["resolved"]) == (5, 3, 2)
    assert result["resolution_rate"] == 0.4
    assert result["urgent_open"] == 2 and result["needs_review_open"] == 1
    assert result["average_confidence"] == 0.82


def test_priority_breakdown_keeps_all_priorities_in_order():
    rows = PriorityBreakdown().compute([ticket(priority="P1"), ticket(priority="P1", is_resolved=True),
                                        ticket(priority="weird"), ticket(priority=None)])
    assert [r["label"] for r in rows] == ["P1", "P2", "P3", "P4", "P5", "unknown", "weird"]
    assert rows[0] == {"label": "P1", "open": 1, "resolved": 1}
    assert rows[1] == {"label": "P2", "open": 0, "resolved": 0}      # empty ones still shown


def test_department_breakdown_is_busiest_first():
    rows = DepartmentWorkload().compute([ticket(department="HR"), ticket(department="Cyber"),
                                         ticket(department="Cyber"), ticket(department="Cyber", is_resolved=True)])
    assert [(r["label"], r["open"], r["resolved"]) for r in rows] == [("Cyber", 2, 1), ("HR", 1, 0)]


def test_submission_trend_fills_gaps_and_caps_the_range():
    days = SubmissionTrend().compute([ticket(submitted_at="2026-10-01T10:00:00+00:00"),
                                      ticket(submitted_at="2026-10-03T10:00:00+00:00"),
                                      ticket(submitted_at="2026-10-03T11:00:00+00:00")])
    assert days == [{"label": "2026-10-01", "count": 1}, {"label": "2026-10-02", "count": 0},
                    {"label": "2026-10-03", "count": 2}]

    long = SubmissionTrend().compute([ticket(submitted_at="2026-08-01T00:00:00+00:00"),
                                      ticket(submitted_at="2026-10-12T00:00:00+00:00")])
    assert len(long) == analytics.MAX_TREND_DAYS and long[-1]["label"] == "2026-10-12"

    # other time zones are converted to UTC; a time without a zone counts as UTC
    shifted = SubmissionTrend().compute([ticket(submitted_at="2026-10-12T23:30:00-05:00"),
                                         ticket(submitted_at="2026-10-13T08:00:00")])
    assert shifted == [{"label": "2026-10-13", "count": 2}]
    assert SubmissionTrend().compute([]) == []
    assert Metric.when({"t": "2026-10-13T08:00:00"}, "t").utcoffset().total_seconds() == 0
    assert Metric.when({"t": "nonsense"}, "t") is None and Metric.when({}, "t") is None


def test_resolution_times():
    tickets = [resolved(2, "P1"), resolved(4, "P1"), resolved(10, "P3"),
               ticket(resolved_at="2026-10-12T05:00:00+00:00"),                     # not resolved
               ticket(is_resolved=True, resolved_at="2026-10-11T01:00:00+00:00"),   # ends before it starts
               ticket(is_resolved=True, resolved_at=None)]                          # no end time
    result = ResolutionTimes().compute(tickets)
    assert result["counted"] == 3 and result["average"] == 5.3 and result["median"] == 4.0
    assert result["by_priority"] == [{"label": "P1", "average": 3.0}, {"label": "P3", "average": 10.0}]
    assert ResolutionTimes().compute([ticket()]) == {"counted": 0, "average": None,
                                                     "median": None, "by_priority": []}


def test_confidence_bands():
    scores = [0.5, 0.6, 0.79, 0.8, 0.95, 1.0, 1.5, -0.1, "high", True, None, float("nan")]
    bands = ConfidenceBands().compute([ticket(confidence_score=s) for s in scores])
    assert [b["count"] for b in bands] == [1, 2, 1, 2]       # impossible scores are left out
    assert [b["label"] for b in bands][0] == "under 60%"


# ---------- the report ----------

def test_report_has_every_section():
    report = AnalyticsReport().build([ticket()], now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert report["generated_at"] == "2026-11-01T00:00:00+00:00"
    for key in ("overview", "by_priority", "by_department", "by_category", "per_day",
                "resolution_hours", "ai_confidence"):
        assert report[key] is not None, key
    assert report["failed_metrics"] == [] and report["ignored_records"] == 0


def test_empty_data():
    report = AnalyticsReport().build([])
    assert report["overview"]["total"] == 0 and report["overview"]["resolution_rate"] == 0
    assert report["overview"]["average_confidence"] is None
    assert report["per_day"] == [] and report["resolution_hours"]["counted"] == 0
    assert [r["label"] for r in report["by_priority"]] == ["P1", "P2", "P3", "P4", "P5"]


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

        def compute(self, tickets):
            return sum(t.get("priority") == "P1" for t in tickets)

    class Broken(Metric):
        key = "broken"

        def compute(self, tickets):
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