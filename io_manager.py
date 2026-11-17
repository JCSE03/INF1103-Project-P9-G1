import os
import re
import webbrowser
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from threading import Lock, Timer
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

from ai_manager import (
    AIManager,
    GeminiProvider,
    InputSanitizer,
    MODEL_NAME,
    SYSTEM_INSTRUCTION,
    SensitiveDataRedactor,
    api_key,
    client,
)
from ai_summary import ReportSummarizer
from ai_troubleshooting import TicketAdvisor, TroubleshootingAdvisor
from analytics import AnalyticsReport
from data_manager import (
    DataReadError,
    DataSaveError,
    RepositoryFactory,
    TicketRepository,
)
from logic_manager import LogicManager
from pending_tickets import InMemoryPendingTickets, PendingTicketStore
from report_export import CsvReportExporter

# The different returns of the Flask routes
RouteResult = str | Response | tuple[Response, int]

GONE_REASON = (
    "This request has expired or was already answered. "
    "Please submit your request again if you still need help."
)


class BaseIOManager(ABC):
    """Abstract base class for application I/O managers."""

    @staticmethod
    def notify(message: str) -> None:
        print(f"[Helpdesk] {message}")

    @staticmethod
    def format_error(reason: str, code: int) -> tuple[Response, int]:
        return jsonify({"status": "error", "reason": reason, "manual_review": True}), code

    @abstractmethod
    def start(self) -> None:
        """Start the I/O interface."""


class InputValidator(ABC):
    """Abstract base class for payload validators."""

    @abstractmethod
    def validate(self, payload: Any) -> tuple[dict[str, Any] | None, str | None]:
        """Validate raw input and return (cleaned_data, error_message)."""


class TicketFormValidator(InputValidator):
    """Validates and sanitizes incoming ticket submission payloads."""

    MAX_NAME_LENGTH = 100
    MAX_TITLE_LENGTH = 120
    CONTACT_PATTERN = re.compile(r"^(?:\+65[\s-]?)?[3689]\d{3}[\s-]?\d{4}$")

    def __init__(
        self,
        sanitizer: InputSanitizer | None = None,
        redactor: SensitiveDataRedactor | None = None,
    ) -> None:
        self._sanitizer = sanitizer or InputSanitizer()
        self._redactor = redactor or SensitiveDataRedactor()

    def validate(self, payload: Any) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(payload, dict):
            return None, "Request must be a JSON object."

        fields = {key: payload.get(key) for key in ("name", "contact_number", "ticket_title")}
        if not all(isinstance(value, str) and value.strip() for value in fields.values()):
            return None, "Please fill in your name, contact number and a ticket title."

        name = self._sanitizer.sanitize(fields["name"], self.MAX_NAME_LENGTH)
        contact = self._sanitizer.sanitize(fields["contact_number"])
        title = self._sanitizer.sanitize(fields["ticket_title"], self.MAX_TITLE_LENGTH)

        if not self.CONTACT_PATTERN.match(contact):
            return None, "Contact number should be a valid Singapore number, e.g. +65 9123 4567."

        redacted_title, _ = self._redactor.redact(title)

        description = payload.get("problem_description")
        ticket_text = (
            f"{fields['ticket_title']}\n{description}"
            if isinstance(description, str) and description.strip()
            else ""
        )

        return {
            "requester": {
                "name": name,
                "contact_number": contact,
                "ticket_title": redacted_title,
            },
            "ticket_text": ticket_text,
        }, None


class ResolutionPayloadValidator(InputValidator):
    """Validates ticket resolution toggle requests."""

    def validate(self, payload: Any) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(payload, dict) or not isinstance(payload.get("is_resolved"), bool):
            return None, 'Send {"is_resolved": true} or {"is_resolved": false}.'
        return {"is_resolved": payload["is_resolved"]}, None


class AnswerPayloadValidator(InputValidator):
    """Validates the requester's answer to "did the steps solve it?"."""

    def validate(self, payload: Any) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(payload, dict) or not isinstance(payload.get("solved"), bool):
            return None, 'Send {"solved": true} or {"solved": false}.'
        return {"solved": payload["solved"]}, None


class FlaskIOManager(BaseIOManager):
    """Web-based I/O manager built on Flask."""

    STATUS_CODES = {
        "success": 200,
        "invalid": 400,
        "blocked": 403,
        "rate_limited": 429,
        "error": 503,
    }

    def __init__(
        self,
        ai_mgr: AIManager | None = None,
        logic_mgr: LogicManager | None = None,
        repository: TicketRepository | None = None,
        host: str = "127.0.0.1",
        port: int = 5000,
        auto_open: bool = True,
        ticket_validator: InputValidator | None = None,
        resolution_validator: InputValidator | None = None,
        summarizer: ReportSummarizer | None = None,
        advisor: TicketAdvisor | None = None,
        pending: PendingTicketStore | None = None,
        answer_validator: InputValidator | None = None,
        report_exporter: CsvReportExporter | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.auto_open = auto_open

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.app = Flask(__name__, template_folder=base_dir)
        self._data_lock = Lock()

        self.ticket_validator: InputValidator = ticket_validator or TicketFormValidator()
        self.resolution_validator: InputValidator = resolution_validator or ResolutionPayloadValidator()
        self.answer_validator: InputValidator = answer_validator or AnswerPayloadValidator()

        if ai_mgr is not None:
            self.ai_manager = ai_mgr
        else:
            provider = GeminiProvider(
                client, MODEL_NAME, SYSTEM_INSTRUCTION, key_hint=api_key[-4:]
            )
            self.ai_manager = AIManager(provider)

        self.logic_manager: LogicManager = logic_mgr or LogicManager()
        self.repository: TicketRepository = repository or RepositoryFactory.create()
        # One shared summarizer, so its cache and request limit apply to every visitor.
        self.summarizer: ReportSummarizer = summarizer or ReportSummarizer.with_gemini()
        # Pre-ticket AI troubleshooting: low-urgency tickets are held here while
        # the requester tries the steps, and only saved if they did not help.
        self.advisor: TicketAdvisor = advisor or TroubleshootingAdvisor.with_gemini()
        self.pending: PendingTicketStore = pending or InMemoryPendingTickets()
        self.report_exporter: CsvReportExporter = report_exporter or CsvReportExporter()

        self._register_routes()

    def _register_routes(self) -> None:
        self.app.add_url_rule("/", view_func=self.home, methods=["GET"])
        self.app.add_url_rule("/submit-ticket", view_func=self.submit_ticket, methods=["POST"])
        self.app.add_url_rule(
            "/api/pending-tickets/<token>/answer",
            view_func=self.answer_troubleshooting,
            methods=["POST"],
        )
        self.app.add_url_rule("/queue", view_func=self.queue, methods=["GET"])
        self.app.add_url_rule("/helpdesk", view_func=self.helpdesk, methods=["GET"])
        self.app.add_url_rule("/departments/<department>", view_func=self.department_page, methods=["GET"])
        self.app.add_url_rule(
            "/api/tickets/<ticket_id>/toggle-resolution",
            view_func=self.toggle_resolution,
            methods=["POST"],
        )
        self.app.add_url_rule("/analytics", view_func=self.analytics_page, methods=["GET"])
        self.app.add_url_rule("/api/analytics", view_func=self.analytics_report, methods=["GET"])
        self.app.add_url_rule(
            "/api/analytics/summary",
            view_func=self.analytics_summary,
            methods=["POST"],
        )
        self.app.add_url_rule("/api/analytics/export", view_func=self.analytics_export, methods=["GET"])

    def home(self) -> str:
        return render_template("index.html")

    def submit_ticket(self) -> RouteResult:
        raw_data = request.get_json(silent=True)

        validated, error_reason = self.ticket_validator.validate(raw_data)
        if error_reason or validated is None:
            return jsonify({
                "status": "invalid",
                "reason": error_reason,
                "manual_review": True,
            }), 400

        user_id = request.remote_addr or "anonymous"
        classification = self.ai_manager.classify(validated["ticket_text"], user_id=user_id)
        result = classification.to_dict()
        self.notify(f"Ticket from {user_id}: {result['status']}")

        if not classification.is_success:
            return jsonify(result), self.STATUS_CODES.get(result["status"], 500)

        decision = self.logic_manager.process(result).to_dict()
        requester = validated["requester"]

        # Low-urgency Infra tickets get AI troubleshooting steps first. The
        # ticket is held (not saved) until the requester says the steps did not
        # help. If the AI cannot write steps, fall through and save as normal.
        if self.logic_manager.offers_self_service(decision):
            advice = self.advisor.advise(
                decision["category"], decision["summary"],
                validated["ticket_text"], user_id=user_id,
            )
            if advice.is_success:
                held = {"requester": requester, "decision": decision}
                token = self.pending.hold(held)
                self.notify(f"Ticket from {user_id} held for troubleshooting.")
                return jsonify({
                    "status": "troubleshooting",
                    "pending_id": token,
                    "troubleshooting": advice.to_dict(),
                    "manual_review": False,
                }), 200
            self.notify(f"Troubleshooting unavailable ({advice.status}); saving ticket.")

        saved, error_response = self._save_ticket(requester, decision)
        if error_response:
            return error_response

        result["ticket"] = saved
        result["manual_review"] = bool(result.get("manual_review") or saved["needs_review"])
        return jsonify(result), 200

    def _save_ticket(
        self, requester: dict, decision: dict
    ) -> tuple[dict | None, tuple[Response, int] | None]:
        """Save a processed ticket. Returns (ticket, None) or (None, error_response)."""
        with self._data_lock:
            try:
                # Data Manager hands out the next free id, so ids stay unique
                # even after a ticket has been deleted.
                ticket_id = self.repository.next_ticket_id()
            except DataReadError:
                self.notify("Ticket data is unreadable; ticket was NOT saved.")
                return None, self.format_error(
                    "Ticket data could not be read, so the ticket was not saved.", 503
                )

            ticket = {
                "ticket_id": ticket_id,
                "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **requester,
                **decision,
                "is_resolved": False,
                "resolved_at": None,
            }
            try:
                self.repository.add(ticket)
            except DataSaveError:
                self.notify("Could not save ticket data; ticket was NOT saved.")
                return None, self.format_error(
                    "The ticket could not be saved. Please try again.", 503
                )

        self.notify(f"Saved {ticket['ticket_id']}: {ticket['priority']} -> {ticket['department']}")
        return ticket, None

    def answer_troubleshooting(self, token: str) -> RouteResult:
        """The requester's answer to the troubleshooting steps: solved drops the
        held ticket, not solved saves it for the IT team."""
        validated, error_reason = self.answer_validator.validate(
            request.get_json(silent=True)
        )
        if error_reason or validated is None:
            return self.format_error(error_reason or "Invalid payload.", 400)

        held = self.pending.take(token)
        if held is None:
            return self.format_error(GONE_REASON, 410)

        if validated["solved"]:
            self.notify("Troubleshooting solved the issue; ticket dropped.")
            return jsonify({"status": "closed"})

        saved, error_response = self._save_ticket(held["requester"], held["decision"])
        if error_response:
            self.pending.put_back(token, held)  # let the requester try again
            return error_response

        return jsonify({
            "status": "success",
            "ticket": saved,
            "manual_review": bool(saved["needs_review"]),
        })

    def queue(self) -> RouteResult:
        department = request.args.get("department")
        if department and department not in self.logic_manager.departments:
            return self.format_error("Unknown department.", 400)

        with self._data_lock:
            try:
                records = (
                    self.repository.find("department", department)
                    if department
                    else self.repository.load_all()
                )
            except DataReadError:
                return self.format_error("Ticket data could not be read.", 503)

        return jsonify({"status": "success", "tickets": self.logic_manager.sort_queue(records)})

    def helpdesk(self) -> str:
        """Dashboard for helpdesk staff showing every department's queue."""
        return render_template(
            "helpdesk.html",
            initial_department="all",
            departments=self.logic_manager.departments,
        )

    def department_page(self, department: str) -> RouteResult:
        if department not in self.logic_manager.departments:
            return self.format_error("Unknown department.", 404)
        return render_template(
            "helpdesk.html",
            initial_department=department,
            departments=self.logic_manager.departments,
        )

    def analytics_page(self) -> str:
        """Charts and statistics for IT managers."""
        return render_template("analytics.html", departments=self.logic_manager.departments)

    def _report_for_request(self) -> tuple[dict | None, tuple[Response, int] | None]:
        """The analytics report for ?department= (all departments by default), as
        (report, None), or (None, error_response) if the request or the data is bad."""
        department = request.args.get("department")
        if department and department not in self.logic_manager.departments:
            return None, self.format_error("Unknown department.", 400)

        with self._data_lock:
            try:
                records = (
                    self.repository.find("department", department)
                    if department
                    else self.repository.load_all()
                )
            except DataReadError:
                return None, self.format_error("Ticket data could not be read.", 503)
        return AnalyticsReport().build(records), None

    def analytics_report(self) -> RouteResult:
        report, error = self._report_for_request()
        return error or jsonify({"status": "success", "report": report})

    def analytics_summary(self) -> RouteResult:
        """An AI-written summary of the same report. POST, because it costs an AI request."""
        report, error = self._report_for_request()
        if error:
            return error

        result = self.summarizer.summarize(report, user_id=request.remote_addr or "anonymous")
        code = 200 if result.status == "no_data" else self.STATUS_CODES.get(result.status, 500)
        return jsonify(result.to_dict()), code

    def analytics_export(self) -> RouteResult:
        """The report as a downloadable CSV file. It never asks the AI: the file
        has an AI summary only if one was already made for these numbers."""
        report, error = self._report_for_request()
        if error:
            return error

        department = request.args.get("department")
        body = self.report_exporter.render(report, self.summarizer.cached_summary(report), department)
        name = re.sub(r"[^A-Za-z0-9_-]", "", department or "all")
        filename = f"helpdesk-report-{name}-{report['generated_at'][:10]}.{self.report_exporter.extension}"
        return Response(
            body.encode("utf-8"),
            headers={
                "Content-Type": self.report_exporter.content_type,
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    def toggle_resolution(self, ticket_id: str) -> RouteResult:
        raw_data = request.get_json(silent=True)

        validated, error_reason = self.resolution_validator.validate(raw_data)
        if error_reason or validated is None:
            return self.format_error(error_reason or "Invalid payload.", 400)

        is_resolved = validated["is_resolved"]
        resolved_at = (
            datetime.now(timezone.utc).isoformat(timespec="seconds")
            if is_resolved
            else None
        )

        with self._data_lock:
            try:
                updated = self.repository.set_resolution(ticket_id, is_resolved, resolved_at)
            except DataReadError:
                return self.format_error(
                    "Ticket data could not be read, so nothing was changed.", 503
                )
            except DataSaveError:
                return self.format_error("The change could not be saved. Please try again.", 503)

        if not updated:
            return self.format_error("Ticket not found.", 404)

        self.notify(f"{ticket_id} {'resolved' if is_resolved else 'reopened'}")
        return jsonify({
            "status": "success",
            "ticket_id": ticket_id,
            "is_resolved": is_resolved,
            "resolved_at": resolved_at,
        })

    def open_browser(self) -> None:
        """Open the ticket form and the helpdesk dashboard in two browser tabs."""
        webbrowser.open(f"http://{self.host}:{self.port}/")
        webbrowser.open(f"http://{self.host}:{self.port}/helpdesk")

    def start(self) -> None:
        self.notify("Starting web server... Press CTRL+C in the terminal to stop it.")
        if self.auto_open:
            Timer(1, self.open_browser).start()
        self.app.run(host=self.host, port=self.port, debug=False)


if __name__ == "__main__":
    FlaskIOManager().start()