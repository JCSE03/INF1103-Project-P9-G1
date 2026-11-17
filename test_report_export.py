# test_report_export.py - offline tests for report_export.py and the download route

# Run with:  python test_report_export.py
# No API key, internet or AI request is needed; the route tests use a
# temporary data file, so your real tickets are never touched.

import csv
import io
import os
import sys

# io_manager imports ai_manager, which refuses to load without a key.
os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")

from ai_summary import SummaryResult  # noqa: E402
from analytics import AnalyticsReport, SlaPolicy  # noqa: E402
from report_export import BOM, FORMULA_STARTS, CsvReportExporter  # noqa: E402
from test_ai_summary import GOOD, GOOD_REPLY, summarizer_in_route  # noqa: E402
from test_analytics import NOW, data_file, open_for, ticket  # noqa: E402

EXPORTER = CsvReportExporter()
TITLES = ["IT Helpdesk analytics report", "Overview", "Tickets by priority",
          "Workload by department", "Tickets by category", "Tickets submitted per day (UTC)",
          "Hours to resolve", "AI confidence", "Tickets at risk of missing their target time",
          "AI summary"]
TRICKY_TITLE = 'Printer, "jammed"\nagain'


def report_of(*tickets):
    return AnalyticsReport(sla_policy=SlaPolicy()).build(list(tickets), now=NOW)


def read(text):
    """The CSV text as rows, the way a spreadsheet reads it."""
    return list(csv.reader(io.StringIO(text.lstrip(BOM))))


def value(rows, label):
    """The cell next to the first row that starts with `label`."""
    return next(row[1] for row in rows if row and row[0] == label)


def has(rows, label):
    """True if some row starts with `label` (blank rows are skipped)."""
    return any(row and row[0] == label for row in rows)


def sample_report():
    return report_of(open_for(10, priority="P1", ticket_id="T-0042", ticket_title=TRICKY_TITLE),
                     ticket(is_resolved=True))


# ---------- the exporter ----------

def test_every_dashboard_section_is_in_the_file():
    rows = read(EXPORTER.render(sample_report()))
    assert [r[0] for r in rows if len(r) == 1 and r[0] in TITLES] == TITLES


def test_numbers_stay_numbers_and_match_the_report():
    report = sample_report()
    rows = read(EXPORTER.render(report, department="Infra"))
    assert report["overview"]["total"] == 2
    assert (value(rows, "Total tickets"), value(rows, "Open"), value(rows, "Resolved")) == ("2", "1", "1")
    assert value(rows, "Resolution rate (%)") == "50.0"
    assert value(rows, "Department") == "Infra" and value(rows, "Generated at (UTC)") == report["generated_at"]
    assert ["P1", "1", "0", "1"] in rows and ["P2", "0", "0", "0"] in rows   # empty priorities stay
    assert ["software", "1", "1", "2"] in rows                               # total = open + resolved
    late = next(r for r in rows if r and r[0] == "T-0042")
    assert late[2:] == ["P1", "Infra", "10.0", "4", "Overdue"]


def test_long_overdue_lists_say_how_many_are_shown():
    many = [open_for(10 + n, priority="P1", ticket_id=f"T-{n:02d}") for n in range(12)]
    rows = read(EXPORTER.render(report_of(*many)))
    assert ["Showing the 10 worst of 12"] in rows
    assert len([r for r in rows if len(r) == 7 and r[0].startswith("T-")]) == 10
    assert ["Showing the 1 worst of 1"] not in read(EXPORTER.render(sample_report()))


def test_commas_quotes_and_line_breaks_survive():
    late = next(r for r in read(EXPORTER.render(sample_report())) if r and r[0] == "T-0042")
    assert late[1] == TRICKY_TITLE


def test_spreadsheet_formulas_are_blocked():
    dangerous = ['=HYPERLINK("http://evil.example","click")', "+1+1", "-2+3", "@SUM(A1)", "\t=1", "\r=1"]
    tickets = [open_for(10, priority="P1", ticket_id=f"T-{n}", ticket_title=title)
               for n, title in enumerate(dangerous)]
    report = report_of(*tickets, open_for(1, department="=evil", ticket_id="T-X"))
    summary = SummaryResult("success", "=cmd|' /C calc'!A0", ("+bad",), ("@bad",))
    rows = read(EXPORTER.render(report, summary))
    for row in rows:
        for cell in row:
            assert not cell.startswith(FORMULA_STARTS) or cell.startswith("'"), cell
    titles = {r[1] for r in rows if len(r) == 7 and r[0].startswith("T-")}
    assert titles == {"'" + title for title in dangerous}      # shown as typed, with an apostrophe
    assert ["'=evil", "1", "0", "1"] in rows                   # labels from the data too
    assert value(rows, "Headline") == "'=cmd|' /C calc'!A0"
    assert value(rows, "Total tickets") == str(len(dangerous) + 1)           # numbers untouched


def test_ai_summary_is_included_only_when_it_succeeded():
    report = sample_report()
    summary = SummaryResult("success", GOOD["headline"], tuple(GOOD["highlights"]), tuple(GOOD["watch_items"]))
    rows = read(EXPORTER.render(report, summary))
    assert value(rows, "Headline") == GOOD["headline"]
    assert [r[1] for r in rows if r and r[0] == "Highlight"] == GOOD["highlights"]
    assert [r[1] for r in rows if r and r[0] == "Watch"] == GOOD["watch_items"]
    assert any(r and r[0] == "Note" and "AI-generated" in r[1] for r in rows)
    for none_yet in (None, SummaryResult("error", reason="down"), SummaryResult("rate_limited", reason="x")):
        rows = read(EXPORTER.render(report, none_yet))
        assert has(rows, "Not generated") and not has(rows, "Headline")


def test_a_damaged_report_still_gives_a_file():
    damaged = [{}, None, "text",
               {"overview": None, "by_priority": "oops", "per_day": [None, 5], "ai_confidence": {"a": 1},
                "at_risk": {"tickets": "x", "targets": [1], "overdue": "many", "policy_problems": 3},
                "resolution_hours": {"by_priority": 7}, "by_department": [None, {"label": "HR"}]}]
    for report in damaged:
        text = EXPORTER.render(report)
        assert text.startswith(BOM) and "IT Helpdesk analytics report" in text
        assert "Not available" in text, report


def test_file_format_details():
    text = EXPORTER.render(sample_report())
    assert text.startswith(BOM)                       # so Excel reads it as UTF-8
    assert text.endswith("\r\n") and text.count("\r\n") > 20     # the line ending spreadsheets expect
    assert EXPORTER.content_type.startswith("text/csv") and EXPORTER.extension == "csv"


# ---------- the download route ----------

RECORDS = [ticket(ticket_id="T-1", department="Infra"), ticket(ticket_id="T-2", department="Infra", is_resolved=True),
           ticket(ticket_id="T-3", department="HR")]


def test_route_sends_a_csv_file():
    with data_file(RECORDS) as client:
        response = client.get("/api/analytics/export")
        assert response.status_code == 200
        assert response.headers["Content-Type"].startswith("text/csv")
        assert response.headers["Cache-Control"] == "no-store"
        assert response.data.startswith(b"\xef\xbb\xbf")
        rows = read(response.data.decode("utf-8"))
        day = value(rows, "Generated at (UTC)")[:10]
        assert response.headers["Content-Disposition"] == f'attachment; filename="helpdesk-report-all-{day}.csv"'
        assert value(rows, "Total tickets") == "3" and value(rows, "Department") == "All departments"

        infra = client.get("/api/analytics/export?department=Infra")
        rows = read(infra.data.decode("utf-8"))
        assert "helpdesk-report-Infra-" in infra.headers["Content-Disposition"]
        assert value(rows, "Total tickets") == "2" and value(rows, "Department") == "Infra"


def test_route_refuses_bad_requests_cleanly():
    with data_file(RECORDS) as client:
        response = client.get("/api/analytics/export?department=Nope")
        assert response.status_code == 400 and response.get_json()["status"] == "error"
        assert client.post("/api/analytics/export").status_code == 405
    with data_file("{broken") as client:
        response = client.get("/api/analytics/export")
        assert response.status_code == 503 and response.get_json()["status"] == "error"


def test_download_never_asks_the_ai_and_shows_an_existing_summary():
    with data_file(RECORDS) as client, summarizer_in_route(client, GOOD_REPLY) as provider:
        before = read(client.get("/api/analytics/export").data.decode("utf-8"))
        assert has(before, "Not generated") and provider.prompts == []

        assert client.post("/api/analytics/summary").status_code == 200       # the one AI request
        assert len(provider.prompts) == 1

        after = read(client.get("/api/analytics/export").data.decode("utf-8"))
        assert value(after, "Headline") == GOOD["headline"]
        for _ in range(3):
            client.get("/api/analytics/export")
        assert len(provider.prompts) == 1                                     # downloads cost nothing

        other = read(client.get("/api/analytics/export?department=HR").data.decode("utf-8"))
        assert has(other, "Not generated")                                    # other numbers: no summary


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