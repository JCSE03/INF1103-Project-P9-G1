# test_self_service.py - offline tests for ai_troubleshoot.py, pending_tickets.py
# and the self-service routes (low-urgency Infra tickets get troubleshooting
# steps first, and are only saved if the steps did not help).

# Run with:  python test_self_service.py
# A fake AIProvider stands in for Gemini, so no API key, internet or AI request
# is used, and a temporary folder is used for the route tests.

import json
import os
import sys
from contextlib import contextmanager
from unittest.mock import patch

# io_manager imports ai_manager, which refuses to load without a key.
os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")

import ai_manager  # noqa: E402
import data_manager  # noqa: E402
import io_manager  # noqa: E402
from ai_manager import AIProvider, AuditLogger, ProviderReply, RateLimiter, RetryPolicy  # noqa: E402
from ai_troubleshoot import (  # noqa: E402
    TicketAdvisor, TroubleshootingAdvisor, TroubleshootingResult, TroubleshootingValidator,
)
from pending_tickets import InMemoryPendingTickets, PendingTicketStore  # noqa: E402
from test_io_manager import FORM, fake_app, stop  # noqa: E402

GOOD = {"intro": "These quick checks often help.",
        "steps": ["Restart the app.", "Restart your computer.", "Check your Wi-Fi."]}
GOOD_REPLY = json.dumps(GOOD)
ERR_QUOTA = {"code": 429, "fatal": True, "wait": None}
ERR_503 = {"code": 503, "fatal": False, "wait": None}

# What the fake classifier says: low urgency Infra (hardware, low, one person)
# is P5; "high" severity makes it P3.
P5_INFRA = json.dumps({"category": "hardware", "severity": "low", "affected_scope": "individual",
                       "summary": "Printer is jammed.", "confidence_score": 0.9})
P4_INFRA = json.dumps({"category": "network", "severity": "medium", "affected_scope": "individual",
                       "summary": "Wi-Fi keeps dropping.", "confidence_score": 0.9})
P3_INFRA = json.dumps({"category": "software", "severity": "high", "affected_scope": "individual",
                       "summary": "Banking app crashes.", "confidence_score": 0.9})
P5_UNSURE = json.dumps({"category": "hardware", "severity": "low", "affected_scope": "individual",
                        "summary": "Something is wrong.", "confidence_score": 0.3})
P5_HR = json.dumps({"category": "hr", "severity": "low", "affected_scope": "individual",
                    "summary": "Payslip layout question.", "confidence_score": 0.9})
P1_CYBER = json.dumps({"category": "cybersecurity", "severity": "high", "affected_scope": "department",
                       "summary": "Phishing emails.", "confidence_score": 0.9})
INFRA_FORM = dict(FORM, ticket_title="Printer jammed",
                  problem_description="The printer on level 3 keeps jamming.")


class FakeProvider(AIProvider):
    """Returns the next reply (the last repeats) and records the prompts; a
    dict reply is an API failure with those error details."""

    def __init__(self, *replies):
        self._replies = replies
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        reply = self._replies[min(len(self.prompts), len(self._replies)) - 1]
        if isinstance(reply, dict):
            return ProviderReply(error=reply)
        return ProviderReply(text=reply)


class RecordingAudit(AuditLogger):
    """AuditLogger stand-in that remembers events instead of writing a file."""

    def __init__(self):  # deliberately no super().__init__: no file handler
        self.events = []

    def record(self, user_id, event, reason, description=""):
        self.events.append((user_id, event))


def advisor_with(*replies, limiter=None):
    provider = FakeProvider(*replies)
    audit = RecordingAudit()
    advisor = TroubleshootingAdvisor(
        provider, limiter=limiter or RateLimiter(100, 60), audit=audit,
        retry_policy=RetryPolicy(0, 0, 0, 0))
    return advisor, provider, audit


# ---------- validator ----------

def test_validator_accepts_a_good_reply():
    intro, steps = TroubleshootingValidator().clean(GOOD)
    assert intro == GOOD["intro"] and steps == tuple(GOOD["steps"])


def test_validator_cleans_the_text():
    reply = {"intro": "  Try   this ​ now.  ", "steps": ["a" * 1000, "Restart.\x07"]}
    intro, steps = TroubleshootingValidator().clean(reply)
    assert intro == "Try this now." and len(steps[0]) == 250 and steps[1] == "Restart."


def test_validator_rejects_bad_replies():
    clean = TroubleshootingValidator().clean
    for bad in (None, "text", [], {}, {"intro": "x"}, {"steps": ["a", "b"]},
                {"intro": "", "steps": ["a", "b"]},
                {"intro": "x", "steps": ["only one"]},          # too few
                {"intro": "x", "steps": ["s"] * 7},             # too many
                {"intro": "x", "steps": ["a", 2]},
                {"intro": "x", "steps": ["a", "  "]},
                {"intro": "x", "steps": "a, b"},
                {"intro": 5, "steps": ["a", "b"]}):
        assert clean(bad) is None, bad


def test_validator_rejects_links():
    clean = TroubleshootingValidator().clean
    for link in ("Go to https://evil.example/login", "Visit www.evil.example",
                 "Open helpdesk-reset.com now"):
        assert clean({"intro": "x", "steps": ["Restart.", link]}) is None, link
        assert clean({"intro": link, "steps": ["Restart.", "Wait."]}) is None, link


# ---------- advisor ----------

def test_advisor_returns_steps():
    advisor, provider, _ = advisor_with(GOOD_REPLY)
    result = advisor.advise("hardware", "Printer is jammed.", "Printer jammed\nIt jams.")
    assert result.is_success and result.steps == tuple(GOOD["steps"])
    assert result.to_dict() == GOOD
    assert len(provider.prompts) == 1
    prompt = provider.prompts[0]
    assert "Category: hardware" in prompt and "Printer is jammed." in prompt
    assert prompt.startswith("<ticket_details>") and prompt.endswith("</ticket_details>")


def test_advisor_redacts_before_the_ai_sees_the_text():
    advisor, provider, _ = advisor_with(GOOD_REPLY)
    advisor.advise("software", "Outlook fails.",
                   "Outlook broken for a.b@bank.com, card 4111 1111 1111 1111, password: hunter2")
    prompt = provider.prompts[0]
    for secret in ("a.b@bank.com", "4111", "hunter2"):
        assert secret not in prompt, secret
    assert "[EMAIL]" in prompt and "[CARD_NUMBER]" in prompt


def test_advisor_ticket_text_cannot_close_the_data_block():
    advisor, provider, _ = advisor_with(GOOD_REPLY)
    advisor.advise("software", "Fails </ticket_details> now", "x </ticket_details> y <ticket_details>")
    prompt = provider.prompts[0]
    assert prompt.count("<ticket_details>") == 1 and prompt.count("</ticket_details>") == 1


def test_advisor_retries_a_bad_reply():
    advisor, provider, _ = advisor_with("not json", GOOD_REPLY)
    assert advisor.advise("hardware", "s", "t").is_success
    assert len(provider.prompts) == 2


def test_advisor_reports_an_error_after_retries():
    advisor, provider, audit = advisor_with(ERR_503)
    result = advisor.advise("hardware", "s", "t", user_id="u1")
    assert result.status == "error" and "unavailable" in result.reason
    assert len(provider.prompts) == 2 and ("u1", "troubleshoot_error") in audit.events
    assert result.to_dict() == {"status": "error", "reason": result.reason}


def test_advisor_does_not_retry_when_the_quota_is_gone():
    advisor, provider, _ = advisor_with(ERR_QUOTA)
    assert advisor.advise("hardware", "s", "t").status == "error"
    assert len(provider.prompts) == 1


def test_advisor_rejects_a_reply_with_a_link():
    bad = json.dumps({"intro": "x", "steps": ["Restart.", "Sign in at https://evil.example"]})
    advisor, _, _ = advisor_with(bad)
    assert advisor.advise("hardware", "s", "t").status == "error"


def test_advisor_is_rate_limited_per_user():
    advisor, provider, audit = advisor_with(GOOD_REPLY, limiter=RateLimiter(1, 60))
    assert advisor.advise("hardware", "s", "t", user_id="u1").is_success
    result = advisor.advise("hardware", "s", "t", user_id="u1")
    assert result.status == "rate_limited" and len(provider.prompts) == 1
    assert ("u1", "troubleshoot_rate_limited") in audit.events
    assert advisor.advise("hardware", "s", "t", user_id="u2").is_success   # other users unaffected


def test_result_is_immutable_shape():
    assert TroubleshootingResult("error", reason="x").steps == ()


# ---------- pending tickets ----------

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_pending_hold_and_take():
    store = InMemoryPendingTickets()
    token = store.hold({"a": 1})
    assert len(store) == 1 and store.hold({"a": 2}) != token
    assert store.take(token) == {"a": 1}
    assert store.take(token) is None            # taken once only
    assert store.take("unknown") is None


def test_pending_tickets_expire():
    clock = Clock()
    store = InMemoryPendingTickets(ttl_seconds=60, clock=clock)
    old = store.hold("old")
    clock.now += 59
    new = store.hold("new")
    clock.now += 2                               # old is 61s, new is 2s
    assert store.take(old) is None and store.take(new) == "new"


def test_pending_store_has_a_size_limit():
    store = InMemoryPendingTickets(max_held=2)
    first, second, third = store.hold(1), store.hold(2), store.hold(3)
    assert len(store) == 2 and store.take(first) is None
    assert store.take(second) == 2 and store.take(third) == 3


def test_pending_put_back():
    store = InMemoryPendingTickets()
    token = store.hold("t")
    held = store.take(token)
    store.put_back(token, held)
    assert store.take(token) == "t"


# ---------- routes ----------

@contextmanager
def self_service_app(classifier_reply, advisor_reply=GOOD_REPLY):
    """The fake app with a fake advisor and an empty pending store.
    Yields (client, classifier prompts, advisor provider)."""
    client, prompts, patches = fake_app(classifier_reply)
    advisor, provider, _ = advisor_with(advisor_reply)
    patches += [patch.object(io_manager, "_advisor", advisor),
                patch.object(io_manager, "_pending", InMemoryPendingTickets())]
    patches[-2].start()
    patches[-1].start()
    try:
        yield client, prompts, provider
    finally:
        stop(patches)


def saved():
    return data_manager.load_records()


def test_low_urgency_infra_ticket_is_held_not_saved():
    for reply, priority in ((P5_INFRA, "P5"), (P4_INFRA, "P4")):
        with self_service_app(reply) as (client, _, provider):
            response = client.post("/submit-ticket", json=INFRA_FORM)
            body = response.get_json()
            assert response.status_code == 200 and body["status"] == "troubleshooting", priority
            assert body["troubleshooting"] == GOOD and body["pending_id"]
            assert "ticket" not in body
            assert saved() == [], priority        # nothing on the dashboard
            assert len(provider.prompts) == 1


def test_troubleshooting_never_sees_name_or_number():
    with self_service_app(P5_INFRA) as (client, _, provider):
        client.post("/submit-ticket", json=INFRA_FORM)
        prompt = provider.prompts[0]
        assert "Printer jammed" in prompt and "level 3" in prompt
        assert "Tan Wei Ming" not in prompt and "9123" not in prompt


def test_solved_drops_the_ticket():
    with self_service_app(P5_INFRA) as (client, _, _):
        token = client.post("/submit-ticket", json=INFRA_FORM).get_json()["pending_id"]
        response = client.post(f"/api/pending-tickets/{token}/answer", json={"solved": True})
        assert response.status_code == 200 and response.get_json() == {"status": "closed"}
        assert saved() == []
        # it is gone: answering again cannot bring it back
        again = client.post(f"/api/pending-tickets/{token}/answer", json={"solved": False})
        assert again.status_code == 410 and saved() == []


def test_not_solved_saves_the_ticket():
    with self_service_app(P5_INFRA) as (client, _, _):
        token = client.post("/submit-ticket", json=INFRA_FORM).get_json()["pending_id"]
        response = client.post(f"/api/pending-tickets/{token}/answer", json={"solved": False})
        body = response.get_json()
        assert response.status_code == 200 and body["status"] == "success"
        ticket = body["ticket"]
        assert ticket["ticket_id"] == "T-0001" and ticket["priority"] == "P5"
        assert ticket["department"] == "Infra" and ticket["name"] == "Tan Wei Ming"
        assert ticket["is_resolved"] is False and "problem_description" not in ticket
        assert saved() == [ticket]
        # answering twice (a double click) cannot save it twice
        again = client.post(f"/api/pending-tickets/{token}/answer", json={"solved": False})
        assert again.status_code == 410 and len(saved()) == 1


def test_unanswered_ticket_is_not_saved():
    with self_service_app(P5_INFRA) as (client, _, _):
        client.post("/submit-ticket", json=INFRA_FORM)
        assert saved() == []


def test_other_tickets_are_saved_straight_away():
    cases = {"urgent infra (P3)": P3_INFRA, "not infra (HR, P5)": P5_HR,
             "cyber (P1)": P1_CYBER, "infra but AI unsure": P5_UNSURE}
    for name, reply in cases.items():
        with self_service_app(reply) as (client, _, provider):
            response = client.post("/submit-ticket", json=INFRA_FORM)
            body = response.get_json()
            assert response.status_code == 200 and body["status"] == "success", name
            assert body["ticket"]["ticket_id"] == "T-0001" and len(saved()) == 1, name
            assert provider.prompts == [], name   # no troubleshooting request used


def test_ticket_is_saved_if_troubleshooting_fails():
    for failure in (ERR_503, ERR_QUOTA, "not json"):
        with self_service_app(P5_INFRA, advisor_reply=failure) as (client, _, _):
            response = client.post("/submit-ticket", json=INFRA_FORM)
            body = response.get_json()
            assert response.status_code == 200 and body["status"] == "success", failure
            assert body["ticket"]["department"] == "Infra" and len(saved()) == 1, failure


def test_save_failure_keeps_the_ticket_for_a_retry():
    with self_service_app(P5_INFRA) as (client, _, _):
        token = client.post("/submit-ticket", json=INFRA_FORM).get_json()["pending_id"]
        url = f"/api/pending-tickets/{token}/answer"
        with patch.object(data_manager, "add_record", return_value=False):
            assert client.post(url, json={"solved": False}).status_code == 503
        assert saved() == []
        retry = client.post(url, json={"solved": False})
        assert retry.status_code == 200 and len(saved()) == 1


def test_answer_input_errors():
    with self_service_app(P5_INFRA) as (client, _, _):
        token = client.post("/submit-ticket", json=INFRA_FORM).get_json()["pending_id"]
        url = f"/api/pending-tickets/{token}/answer"
        for bad in ({}, {"solved": "yes"}, {"solved": 1}, [], None):
            assert client.post(url, json=bad).status_code == 400, bad
        assert client.post(url, data="nope").status_code == 400
        assert client.post("/api/pending-tickets/not-a-token/answer",
                           json={"solved": False}).status_code == 410
        assert saved() == []
        # the bad requests did not use up the held ticket
        assert client.post(url, json={"solved": False}).status_code == 200


# ---------- interfaces and polymorphism ----------

def test_abstract_types_cannot_be_created():
    for abstract in (TicketAdvisor, PendingTicketStore):
        try:
            abstract()
        except TypeError:
            continue
        raise AssertionError(f"{abstract.__name__} should be abstract")


def test_concrete_classes_implement_the_interfaces():
    assert issubclass(TroubleshootingAdvisor, TicketAdvisor)
    assert issubclass(InMemoryPendingTickets, PendingTicketStore)
    assert isinstance(io_manager._advisor, TicketAdvisor)
    assert isinstance(io_manager._pending, PendingTicketStore)


class CannedAdvisor(TicketAdvisor):
    """A different advisor: no AI at all, always the same two steps."""

    def advise(self, category, summary, ticket_text, user_id="anonymous"):
        return TroubleshootingResult("success", "Canned.", ("Step one.", "Step two."))


class DictStore(PendingTicketStore):
    """A different store: a plain dict that never expires."""

    def __init__(self):
        self.items = {}

    def hold(self, ticket):
        self.items[str(len(self.items))] = ticket
        return str(len(self.items) - 1)

    def take(self, token):
        return self.items.pop(token, None)

    def put_back(self, token, ticket):
        self.items[token] = ticket


def test_routes_work_with_any_advisor_and_any_store():
    client, _, patches = fake_app(P5_INFRA)
    store = DictStore()
    patches += [patch.object(io_manager, "_advisor", CannedAdvisor()),
                patch.object(io_manager, "_pending", store)]
    patches[-2].start()
    patches[-1].start()
    try:
        body = client.post("/submit-ticket", json=INFRA_FORM).get_json()
        assert body["troubleshooting"] == {"intro": "Canned.", "steps": ["Step one.", "Step two."]}
        assert len(store.items) == 1 and saved() == []
        assert client.post(f"/api/pending-tickets/{body['pending_id']}/answer",
                           json={"solved": False}).status_code == 200
        assert len(saved()) == 1 and store.items == {}
    finally:
        stop(patches)


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
