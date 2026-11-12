from abc import ABC, abstractmethod
from dataclasses import dataclass

CYBER_CATEGORY = "cybersecurity"
SINGLE_PERSON_SCOPE = "individual"


class InvalidAIResultError(ValueError):
    pass


@dataclass(frozen=True)
class AIFacts:
    category: str
    severity: str
    scope: str
    confidence: float
    summary: str
    flagged: bool = False

    @classmethod
    def from_ai_result(cls, ai_result: dict) -> "AIFacts":
        if not isinstance(ai_result, dict) or ai_result.get("status") != "success":
            raise InvalidAIResultError("Only a successful AI result can be processed.")

        data = ai_result.get("data")
        if not isinstance(data, dict):
            raise InvalidAIResultError("The AI result has no 'data' section.")

        try:
            return cls(
                category=data["category"],
                severity=data["severity"],
                scope=data["affected_scope"],
                confidence=float(data["confidence_score"]),
                summary=data["summary"],
                flagged=bool(ai_result.get("flagged", False)),
            )
        except KeyError as missing:
            raise InvalidAIResultError(f"The AI result is missing {missing}.") from None
        except (TypeError, ValueError):
            raise InvalidAIResultError("confidence_score is not a number.") from None


class ScoringRubric:

    DEFAULT_SEVERITY_POINTS = {"low": 10, "medium": 25, "high": 40, "critical": 55}

    DEFAULT_SCOPE_POINTS = {"individual": 0, "multiple_users": 10,
                            "department": 20, "organisation": 30}

    DEFAULT_CYBER_BONUS = 10

    DEFAULT_THRESHOLDS = (("P1", 70), ("P2", 50), ("P3", 35), ("P4", 25))
    LOWEST_PRIORITY = "P5"

    def __init__(self, severity_points: dict[str, int] | None = None,
                 scope_points: dict[str, int] | None = None,
                 cyber_bonus: int | None = None,
                 thresholds: list[tuple[str, int]] | None = None) -> None:
        self._severity_points = dict(severity_points or self.DEFAULT_SEVERITY_POINTS)
        self._scope_points = dict(scope_points or self.DEFAULT_SCOPE_POINTS)
        self._cyber_bonus = self.DEFAULT_CYBER_BONUS if cyber_bonus is None else cyber_bonus
        self._thresholds = sorted(thresholds or self.DEFAULT_THRESHOLDS,
                                  key=lambda pair: pair[1], reverse=True)

    @property
    def severities(self) -> tuple[str, ...]:
        return tuple(self._severity_points)

    @property
    def scopes(self) -> tuple[str, ...]:
        return tuple(self._scope_points)

    def score(self, facts: AIFacts) -> int:
        score = (self._points(self._severity_points, facts.severity, "severity")
                 + self._points(self._scope_points, facts.scope, "affected_scope"))
        if facts.category == CYBER_CATEGORY:
            score += self._cyber_bonus
        return score

    def priority_for(self, score: int) -> str:
        for priority, minimum_score in self._thresholds:
            if score >= minimum_score:
                return priority
        return self.LOWEST_PRIORITY

    @staticmethod
    def _points(table: dict[str, int], value: str, field_name: str) -> int:
        if value not in table:
            raise InvalidAIResultError(f"Unknown {field_name}: {value!r}")
        return table[value]


class Router:

    DEFAULT_ROUTES = {
        "cybersecurity": "Cyber",
        "hr": "HR",
        "finance": "Finance",
        "hardware": "Infra",
        "software": "Infra",
        "network": "Infra",
        "account_access": "Infra",
        "other": "Infra",
    }

    def __init__(self, routes: dict[str, str] | None = None) -> None:
        self._routes = dict(routes or self.DEFAULT_ROUTES)

    @property
    def departments(self) -> list[str]:
        return list(dict.fromkeys(self._routes.values()))

    def department_for(self, category: str) -> str:
        if category not in self._routes:
            raise InvalidAIResultError(f"Unknown category: {category!r}")
        return self._routes[category]


class PriorityRule(ABC):

    @abstractmethod
    def adjust(self, priority: str, facts: AIFacts) -> str:
        ...


class CyberMultiUserRule(PriorityRule):

    def adjust(self, priority: str, facts: AIFacts) -> str:
        if facts.category == CYBER_CATEGORY and facts.scope != SINGLE_PERSON_SCOPE:
            return "P1"
        return priority


class ReviewRule(ABC):

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def reason(self) -> str:
        return self._reason

    def check(self, facts: AIFacts) -> str | None:
        return self._reason if self._is_triggered(facts) else None

    @abstractmethod
    def _is_triggered(self, facts: AIFacts) -> bool:
        ...


class LowConfidenceRule(ReviewRule):

    DEFAULT_THRESHOLD = 0.6

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        super().__init__(f"AI confidence is below {threshold:.0%}")
        self._threshold = threshold

    def _is_triggered(self, facts: AIFacts) -> bool:
        return facts.confidence < self._threshold


class AIFlaggedRule(ReviewRule):

    def __init__(self) -> None:
        super().__init__("Flagged by the security check: verify the requester's identity")

    def _is_triggered(self, facts: AIFacts) -> bool:
        return facts.flagged


@dataclass(frozen=True)
class TicketDecision:
    priority: str
    priority_score: int
    department: str
    facts: AIFacts
    review_reasons: tuple[str, ...] = ()

    @property
    def priority_number(self) -> int:
        return int(self.priority[1:])

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)

    def to_dict(self) -> dict:
        return {
            "priority": self.priority,
            "priority_number": self.priority_number,
            "priority_score": self.priority_score,
            "department": self.department,
            "needs_review": self.needs_review,
            "review_reasons": list(self.review_reasons),
            "category": self.facts.category,
            "severity": self.facts.severity,
            "affected_scope": self.facts.scope,
            "summary": self.facts.summary,
            "confidence_score": self.facts.confidence,
        }


class TicketQueue:

    UNKNOWN_PRIORITY_NUMBER = 99

    @classmethod
    def sort_key(cls, ticket: dict) -> tuple[int, int, str]:
        return (ticket.get("priority_number", cls.UNKNOWN_PRIORITY_NUMBER),
                -ticket.get("priority_score", 0),
                ticket.get("submitted_at", ""))

    def sort(self, tickets: list[dict]) -> list[dict]:
        return sorted(tickets, key=self.sort_key)


class LogicManager:

    def __init__(self, rubric: ScoringRubric | None = None,
                 router: Router | None = None,
                 priority_rules: list[PriorityRule] | None = None,
                 review_rules: list[ReviewRule] | None = None) -> None:
        self._rubric = rubric if rubric is not None else ScoringRubric()
        self._router = router if router is not None else Router()
        self._priority_rules = (list(priority_rules) if priority_rules is not None
                                else [CyberMultiUserRule()])
        self._review_rules = (list(review_rules) if review_rules is not None
                              else [LowConfidenceRule(), AIFlaggedRule()])
        self._queue = TicketQueue()

    @property
    def departments(self) -> list[str]:
        return self._router.departments

    def process(self, ai_result: dict) -> TicketDecision:
        facts = AIFacts.from_ai_result(ai_result)

        score = self._rubric.score(facts)
        priority = self._rubric.priority_for(score)

        for rule in self._priority_rules:
            priority = rule.adjust(priority, facts)

        reasons = tuple(reason for rule in self._review_rules
                        if (reason := rule.check(facts)) is not None)

        return TicketDecision(
            priority=priority,
            priority_score=score,
            department=self._router.department_for(facts.category),
            facts=facts,
            review_reasons=reasons,
        )

    def sort_queue(self, tickets: list[dict]) -> list[dict]:
        return self._queue.sort(tickets)


_default_manager = LogicManager()
DEPARTMENTS = _default_manager.departments


def process_ticket(ai_result: dict) -> dict:
    return _default_manager.process(ai_result).to_dict()


def sort_queue(tickets: list[dict]) -> list[dict]:
    return _default_manager.sort_queue(tickets)