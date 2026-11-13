"""Analytics: turns the stored tickets into the numbers behind the dashboard.

AnalyticsReport runs a list of Metric objects over the ticket records and
returns one JSON-ready dict. It only reads: no Flask, no files, no AI call, so
it can be tested with plain lists of dicts.
"""

import json
import logging
import math
import os
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Any

PRIORITIES = ("P1", "P2", "P3", "P4", "P5")
URGENT_PRIORITIES = ("P1", "P2")
# (label, from, up to but not including): the bands of the AI confidence chart
CONFIDENCE_BANDS = (("under 60%", 0.0, 0.6), ("60-79%", 0.6, 0.8),
                    ("80-89%", 0.8, 0.9), ("90-100%", 0.9, 1.01))
MAX_TREND_DAYS = 30

# Hours a ticket may stay open before it is overdue. Can be changed in
# sla_policy.json (next to this file) without touching the code.
DEFAULT_SLA_HOURS = {"P1": 4, "P2": 8, "P3": 24, "P4": 72, "P5": 120}
SLA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sla_policy.json")
DUE_SOON_SHARE = 0.75      # "due soon" = 75% of the target time used up
MAX_AT_RISK_LISTED = 10


class Metric(ABC):
    """One analysis of the stored tickets. A subclass sets `key` (its name in
    the report) and says how to compute it. The helpers for reading messy
    records (tickets.json can be edited by hand) are shared here."""

    key: str

    @abstractmethod
    def compute(self, tickets: Sequence[dict], now: datetime) -> Any:
        """JSON-ready result. Must cope with missing or odd values. `now` is
        the time of the report (UTC), so a result depends only on the data
        and on `now`, never on the clock."""

    @staticmethod
    def is_open(ticket: dict) -> bool:
        return ticket.get("is_resolved") is not True

    @staticmethod
    def text(ticket: dict, field: str, default: str = "unknown") -> str:
        value = ticket.get(field)
        return value if isinstance(value, str) and value else default

    @staticmethod
    def number(ticket: dict, field: str) -> float | None:
        value = ticket.get(field)
        usable = (isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value))
        return float(value) if usable else None

    @staticmethod
    def when(ticket: dict, field: str) -> datetime | None:
        try:
            moment = datetime.fromisoformat(ticket[field])
        except (KeyError, TypeError, ValueError):
            return None
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


class Overview(Metric):
    key = "overview"

    def compute(self, tickets: Sequence[dict], now: datetime) -> dict:
        open_tickets = [t for t in tickets if self.is_open(t)]
        scores = [s for s in (self.number(t, "confidence_score") for t in tickets)
                  if s is not None]
        total = len(tickets)
        return {
            "total": total,
            "open": len(open_tickets),
            "resolved": total - len(open_tickets),
            "resolution_rate": round((total - len(open_tickets)) / total, 3) if total else 0,
            "urgent_open": sum(t.get("priority") in URGENT_PRIORITIES for t in open_tickets),
            "needs_review_open": sum(t.get("needs_review") is True for t in open_tickets),
            "average_confidence": round(mean(scores), 3) if scores else None,
        }


class GroupedMetric(Metric):
    """Open and resolved tickets per value of one field. A subclass names the
    field and, if it wants, a fixed order for the values (always shown, even
    with no tickets); other values follow, busiest first."""

    field: str
    order: Sequence[str] = ()

    def compute(self, tickets: Sequence[dict], now: datetime) -> list[dict]:
        counts: dict[str, dict[str, int]] = defaultdict(lambda: {"open": 0, "resolved": 0})
        for ticket in tickets:
            counts[self.text(ticket, self.field)]["open" if self.is_open(ticket) else "resolved"] += 1
        others = sorted((v for v in counts if v not in self.order),
                        key=lambda v: (-sum(counts[v].values()), v))
        return [{"label": v, **counts.get(v, {"open": 0, "resolved": 0})}
                for v in [*self.order, *others]]


class PriorityBreakdown(GroupedMetric):
    key, field, order = "by_priority", "priority", PRIORITIES


class DepartmentWorkload(GroupedMetric):
    key, field = "by_department", "department"


class CategoryBreakdown(GroupedMetric):
    key, field = "by_category", "category"


class SubmissionTrend(Metric):
    """Tickets submitted per day (UTC), the last MAX_TREND_DAYS days that
    have data, with empty days filled in as zero."""
    key = "per_day"

    def compute(self, tickets: Sequence[dict], now: datetime) -> list[dict]:
        moments = (self.when(t, "submitted_at") for t in tickets)
        days = Counter(m.astimezone(timezone.utc).date() for m in moments if m)
        if not days:
            return []
        last = max(days)
        length = min((last - min(days)).days + 1, MAX_TREND_DAYS)
        series = []
        for offset in range(length - 1, -1, -1):
            day = last - timedelta(days=offset)
            series.append({"label": day.isoformat(), "count": days.get(day, 0)})
        return series


class ResolutionTimes(Metric):
    """Hours from submission to resolution, for resolved tickets that have
    both timestamps in the right order."""
    key = "resolution_hours"

    def compute(self, tickets: Sequence[dict], now: datetime) -> dict:
        by_priority: dict[str, list[float]] = defaultdict(list)
        for ticket in tickets:
            start, end = self.when(ticket, "submitted_at"), self.when(ticket, "resolved_at")
            if ticket.get("is_resolved") is True and start and end and end >= start:
                by_priority[self.text(ticket, "priority")].append(
                    (end - start).total_seconds() / 3600)
        hours = [h for group in by_priority.values() for h in group]
        return {
            "counted": len(hours),
            "average": round(mean(hours), 1) if hours else None,
            "median": round(median(hours), 1) if hours else None,
            "by_priority": [{"label": p, "average": round(mean(by_priority[p]), 1)}
                            for p in PRIORITIES if by_priority.get(p)],
        }


class ConfidenceBands(Metric):
    """How sure the AI was: tickets per confidence band."""
    key = "ai_confidence"

    def compute(self, tickets: Sequence[dict], now: datetime) -> list[dict]:
        scores = [s for s in (self.number(t, "confidence_score") for t in tickets)
                  if s is not None and 0 <= s <= 1]
        return [{"label": label, "count": sum(low <= s < high for s in scores)}
                for label, low, high in CONFIDENCE_BANDS]


class SlaPolicy:
    """How many hours each priority may stay open before it is overdue. The
    defaults can be overridden in sla_policy.json, e.g. {"P1": 2}. A missing
    or broken file, or one bad value in it, falls back to the default and is
    listed in `problems`, so a typo can never break the dashboard."""

    def __init__(self, hours: dict | None = None) -> None:
        self._hours: dict[str, float] = dict(DEFAULT_SLA_HOURS)
        self._problems: list[str] = []
        for priority, value in (hours or {}).items():
            usable = (isinstance(value, (int, float)) and not isinstance(value, bool)
                      and math.isfinite(value) and value > 0)
            if priority in DEFAULT_SLA_HOURS and usable:
                self._hours[priority] = int(value) if float(value).is_integer() else float(value)
            else:
                self._problems.append(f"Ignored invalid target for {str(priority)[:20]!r}.")

    @classmethod
    def load(cls, path: str | None = None) -> "SlaPolicy":
        path = SLA_FILE if path is None else path
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError):
            policy = cls()
            policy._problems.append("sla_policy.json could not be read; default targets used.")
            return policy
        if not isinstance(data, dict):
            policy = cls()
            policy._problems.append("sla_policy.json must be an object; default targets used.")
            return policy
        return cls(data)

    @property
    def problems(self) -> list[str]:
        return list(self._problems)

    def hours_for(self, priority: Any) -> float | None:
        return self._hours.get(priority) if isinstance(priority, str) else None

    def as_dict(self) -> dict[str, float]:
        return dict(self._hours)


class OverdueTickets(Metric):
    """Open tickets that have passed (overdue) or nearly used up (due soon)
    the target time for their priority. The worst few are listed."""
    key = "at_risk"

    def __init__(self, policy: SlaPolicy | None = None) -> None:
        self._policy = policy if policy is not None else SlaPolicy.load()

    def compute(self, tickets: Sequence[dict], now: datetime) -> dict:
        rows = []
        for ticket in tickets:
            limit = self._policy.hours_for(ticket.get("priority"))
            start = self.when(ticket, "submitted_at")
            if not self.is_open(ticket) or limit is None or start is None:
                continue
            hours = (now - start).total_seconds() / 3600
            if hours / limit < DUE_SOON_SHARE:      # also skips tickets dated in the future
                continue
            rows.append({
                "ticket_id": self.text(ticket, "ticket_id", "?"),
                "title": self.text(ticket, "ticket_title", "(no title)")[:80],
                "priority": ticket["priority"],
                "department": self.text(ticket, "department"),
                "hours_open": round(hours, 1),
                "limit_hours": limit,
                "status": "overdue" if hours > limit else "due_soon",
                "_share": hours / limit,
            })
        rows.sort(key=lambda r: (-r["_share"], r["ticket_id"]))     # the most overdue first
        listed = [{k: v for k, v in row.items() if k != "_share"}
                  for row in rows[:MAX_AT_RISK_LISTED]]
        return {
            "overdue": sum(r["status"] == "overdue" for r in rows),
            "due_soon": sum(r["status"] == "due_soon" for r in rows),
            "tickets": listed,
            "targets": self._policy.as_dict(),
            "policy_problems": self._policy.problems,
        }


class AnalyticsReport:
    """Runs its metrics over the ticket records. The metrics are passed in, so
    a new analysis is a new Metric subclass and nothing here changes; the
    default set is the one the dashboard shows. One metric failing does not
    stop the others."""

    def __init__(self, metrics: Sequence[Metric] | None = None,
                 sla_policy: SlaPolicy | None = None) -> None:
        self._metrics = (list(metrics) if metrics is not None
                         else self.default_metrics(sla_policy))

    @staticmethod
    def default_metrics(sla_policy: SlaPolicy | None = None) -> list[Metric]:
        return [Overview(), PriorityBreakdown(), DepartmentWorkload(),
                CategoryBreakdown(), SubmissionTrend(), ResolutionTimes(),
                ConfidenceBands(), OverdueTickets(sla_policy)]

    def build(self, records: Sequence[Any], now: datetime | None = None) -> dict:
        """The report for these records. Anything that is not a ticket dict is
        ignored (and counted). `now` is only for tests."""
        tickets = [r for r in records if isinstance(r, dict)]
        moment = now or datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        report: dict[str, Any] = {
            "generated_at": moment.isoformat(timespec="seconds"),
            "ignored_records": len(records) - len(tickets),
            "failed_metrics": [],
        }
        for metric in self._metrics:
            try:
                report[metric.key] = metric.compute(tickets, moment)
            except Exception as error:  # a broken metric must not break the page
                logging.error(f"Analytics metric {metric.key!r} failed: {type(error).__name__}")
                report[metric.key] = None
                report["failed_metrics"].append(metric.key)
        return report