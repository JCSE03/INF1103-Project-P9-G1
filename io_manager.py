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
from data_manager import (
    DataReadError,
    DataSaveError,
    JsonTicketRepository,
    TicketRepository,
)
from logic_manager import LogicManager

RouteResult = str | Response | tuple[Response, int]


class InputValidator(ABC):
    """Interface for validating and sanitizing incoming I/O payloads."""

    @abstractmethod
    def validate(self, payload: Any) -> tuple[dict[str, Any] | None, str | None]:
        """Validate raw input payload and return (sanitized_data, error_reason)."""


class BaseIOManager(ABC):
    """Abstract base class defining shared I/O, logging, and error-formatting behavior."""

    STATUS_CODES: dict[str, int] = {
        "success": 200,
        "invalid": 400,
        "blocked": 403,
        "rate_limited": 429,
        "error": 503,
    }

    def notify(self, message: str) -> None:
        print(message)

    def format_error(self, reason: str, code: int) -> tuple[Response, int]:
        return jsonify({
            "status": "error",
            "reason": reason,
            "manual_review": True,
        }), code

    @abstractmethod
    def start(self) -> None:
        """Start the I/O interface."""


class TicketFormValidator(InputValidator):
    """Validates and sanitizes incoming ticket submission payloads."""

    MAX_NAME_LENGTH = 100
    MAX_TITLE_LENGTH = 120
    CONTACT_PATTERN = re.compile(r"^\+?[0-9][0-9 -]{6,18}[0-9]$")

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
            return None, "Contact number should be 8 to 15 digits, e.g. +65 9123 4567."

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


class FlaskIOManager(BaseIOManager):
    """Web-based I/O manager built on Flask."""

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
    ) -> None:
        self.host = host
        self.port = port
        self.auto_open = auto_open

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.app = Flask(__name__, template_folder=base_dir)
        self._data_lock = Lock()

        self.ticket_validator: InputValidator = ticket_validator or TicketFormValidator()
        self.resolution_validator: InputValidator = resolution_validator or ResolutionPayloadValidator()

        provider = GeminiProvider(
            client, MODEL_NAME, SYSTEM_INSTRUCTION, key_hint=api_key[-4:]
        )
        self.ai_manager: AIManager = ai_mgr or AIManager(provider)
        self.logic_manager: LogicManager = logic_mgr or LogicManager()
        self.repository: TicketRepository = repository or JsonTicketRepository()

        self._register_routes()

    def _register_routes(self) -> None:
        self.app.add_url_rule("/", view_func=self.home, methods=["GET"])
        self.app.add_url_rule("/submit-ticket", view_func=self.submit_ticket, methods=["POST"])
        self.app.add_url_rule("/queue", view_func=self.queue, methods=["GET"])
        self.app.add_url_rule("/helpdesk", view_func=self.helpdesk, methods=["GET"])
        self.app.add_url_rule("/departments/<department>", view_func=self.department_page, methods=["GET"])
        self.app.add_url_rule(
            "/api/tickets/<ticket_id>/toggle-resolution",
            view_func=self.toggle_resolution,
            methods=["POST"],
        )

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

        with self._data_lock:
            try:
                records = self.repository.load_all()
                ticket = {
                    "ticket_id": f"T-{len(records) + 1:04d}",
                    "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **validated["requester"],
                    **decision,
                    "is_resolved": False,
                    "resolved_at": None,
                }
                self.repository.add(ticket)
            except DataReadError:
                self.notify("tickets.json is unreadable; ticket was NOT saved.")
                return self.format_error(
                    "Ticket data file could not be read, so the ticket was not saved.", 503
                )
            except DataSaveError:
                self.notify("Could not write tickets.json; ticket was NOT saved.")
                return self.format_error("The ticket could not be saved. Please try again.", 503)

        self.notify(f"Saved {ticket['ticket_id']}: {ticket['priority']} -> {ticket['department']}")

        result["ticket"] = ticket
        result["manual_review"] = bool(result.get("manual_review") or ticket["needs_review"])
        return jsonify(result), 200

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
                return self.format_error("Ticket data file could not be read.", 503)

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
                    "Ticket data file could not be read, so nothing was changed.", 503
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