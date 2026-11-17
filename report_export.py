"""Report export: the analytics report as a CSV file that opens in Excel.

CsvReportExporter only formats: it gets a finished report (and, if one exists,
its AI summary) and returns text. It never reads tickets, never calls the AI
and never touches Flask, so it can be tested with plain dicts.
"""

import csv
import io
from typing import Any

from ai_summary import SummaryResult

BOM = "\ufeff"  # makes Excel read the file as UTF-8 (ticket titles can have any characters)
# A cell starting with one of these is run as a formula by Excel and others.
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")
NOT_AVAILABLE = "Not available"

Rows = list[list[Any]]


class CsvReportExporter:
    """Turns an analytics report into CSV text, one titled block per section
    of the dashboard, so it reads like the page and still opens as a sheet.

    Ticket titles are typed by users, so every text cell is made safe against
    spreadsheet formula injection (a title such as =HYPERLINK(...) would
    otherwise run when the file is opened)."""

    content_type = "text/csv; charset=utf-8"
    extension = "csv"

    def render(self, report: dict, summary: SummaryResult | None = None,
               department: str | None = None) -> str:
        """The whole file as text. `summary` is only used if it succeeded."""
        report = report if isinstance(report, dict) else {}
        blocks: list[Rows] = [
            [["IT Helpdesk analytics report"],
             ["Generated at (UTC)", report.get("generated_at", "")],
             ["Department", department or "All departments"]],
            self._overview(report.get("overview")),
            self._grouped("Tickets by priority", "Priority", report.get("by_priority")),
            self._grouped("Workload by department", "Department", report.get("by_department")),
            self._grouped("Tickets by category", "Category", report.get("by_category")),
            self._counts("Tickets submitted per day (UTC)", "Date", "Tickets", report.get("per_day")),
            self._resolution(report.get("resolution_hours")),
            self._counts("AI confidence", "Band", "Tickets", report.get("ai_confidence")),
            self._at_risk(report.get("at_risk")),
            self._summary(summary),
        ]
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\r\n")
        for number, block in enumerate(blocks):
            if number:
                writer.writerow([])
            for row in block:
                writer.writerow([self._safe(cell) for cell in row])
        return BOM + out.getvalue()

    # --- one method per block of the file --------------------------------

    def _overview(self, data: Any) -> Rows:
        if not isinstance(data, dict):
            return [["Overview"], [NOT_AVAILABLE]]
        return [["Overview"], ["Measure", "Value"],
                ["Total tickets", data.get("total")],
                ["Open", data.get("open")],
                ["Resolved", data.get("resolved")],
                ["Resolution rate (%)", self._percent(data.get("resolution_rate"))],
                ["Urgent open (P1-P2)", data.get("urgent_open")],
                ["Open, needs review", data.get("needs_review_open")],
                ["Average AI confidence (%)", self._percent(data.get("average_confidence"))]]

    def _grouped(self, title: str, label: str, rows: Any) -> Rows:
        if not isinstance(rows, list):
            return [[title], [NOT_AVAILABLE]]
        block: Rows = [[title], [label, "Open", "Resolved", "Total"]]
        for row in rows:
            if isinstance(row, dict):
                opened, resolved = row.get("open"), row.get("resolved")
                total = opened + resolved if self._is_number(opened) and self._is_number(resolved) else None
                block.append([row.get("label"), opened, resolved, total])
        return block

    def _counts(self, title: str, label: str, value: str, rows: Any) -> Rows:
        if not isinstance(rows, list):
            return [[title], [NOT_AVAILABLE]]
        return [[title], [label, value]] + [[row.get("label"), row.get("count")]
                                           for row in rows if isinstance(row, dict)]

    def _resolution(self, data: Any) -> Rows:
        if not isinstance(data, dict):
            return [["Hours to resolve"], [NOT_AVAILABLE]]
        block: Rows = [["Hours to resolve"], ["Measure", "Value"],
                       ["Resolved tickets counted", data.get("counted")],
                       ["Average (hours)", data.get("average")],
                       ["Median (hours)", data.get("median")]]
        for row in self._list(data.get("by_priority")):
            if isinstance(row, dict):
                block.append([f"Average for {row.get('label')} (hours)", row.get("average")])
        return block

    def _at_risk(self, data: Any) -> Rows:
        title = "Tickets at risk of missing their target time"
        if not isinstance(data, dict):
            return [[title], [NOT_AVAILABLE]]
        tickets = self._list(data.get("tickets"))
        targets = data.get("targets") if isinstance(data.get("targets"), dict) else {}
        block: Rows = [[title],
                       ["Overdue", data.get("overdue")], ["Due soon", data.get("due_soon")]]
        in_total = sum(n for n in (data.get("overdue"), data.get("due_soon")) if self._is_number(n))
        if in_total > len(tickets):
            block.append([f"Showing the {len(tickets)} worst of {in_total:g}"])
        for problem in self._list(data.get("policy_problems")):
            block.append(["Note", problem])
        block.append(["Target hours per priority", "; ".join(f"{p} {h}" for p, h in targets.items())])
        block.append(["Ticket", "Title", "Priority", "Department",
                      "Open for (hours)", "Target (hours)", "Status"])
        for ticket in tickets:
            if isinstance(ticket, dict):
                status = {"overdue": "Overdue", "due_soon": "Due soon"}.get(ticket.get("status"),
                                                                            ticket.get("status"))
                block.append([ticket.get("ticket_id"), ticket.get("title"), ticket.get("priority"),
                              ticket.get("department"), ticket.get("hours_open"),
                              ticket.get("limit_hours"), status])
        return block

    def _summary(self, summary: SummaryResult | None) -> Rows:
        if summary is None or summary.status != "success":
            return [["AI summary"],
                    ["Not generated", "Use 'Generate AI summary' on the dashboard before downloading to include one."]]
        block: Rows = [["AI summary"], ["Headline", summary.headline]]
        block += [["Highlight", text] for text in summary.highlights]
        block += [["Watch", text] for text in summary.watch_items]
        block.append(["Note", "AI-generated from the numbers in this report. Check it against them."])
        return block

    # --- small helpers ----------------------------------------------------

    @staticmethod
    def _list(value: Any) -> list:
        return value if isinstance(value, list) else []

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @classmethod
    def _percent(cls, value: Any) -> Any:
        return round(value * 100, 1) if cls._is_number(value) else None

    @staticmethod
    def _safe(cell: Any) -> Any:
        """Numbers stay numbers; text that a spreadsheet could run as a formula
        gets a leading apostrophe, which shows nothing but stops it running."""
        if isinstance(cell, str) and cell.startswith(FORMULA_STARTS):
            return "'" + cell
        return "" if cell is None else cell