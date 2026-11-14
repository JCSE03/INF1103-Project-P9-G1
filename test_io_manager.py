# test_io_manager.py - offline tests for io_manager.py (form checks and routes)

# Run with:  python test_io_manager.py
# No API key or internet needed: Gemini is replaced by a fake reply, and a
# temporary folder is used, so your real data/tickets.json is never touched.

import json
import os
import sys
import tempfile

# ai_manager refuses to import without a key; the fake provider never uses it.
os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")

from ai_manager import AIManager, AIProvider, ProviderReply, RetryPolicy  # noqa: E402
from data_manager import JsonTicketRepository  # noqa: E402
from io_manager import FlaskIOManager, TicketFormValidator  # noqa: E402

FORM = {
    "name": "Tan Wei Ming",
    "contact_number": "+65 9123 4567",
    "ticket_title": "Phishing emails",
    "problem_description": "Several of us received emails asking for our bank login.",
}
AI_REPLY = json.dumps({
    "category": "cybersecurity", "severity": "medium",
    "affected_scope": "multiple_users", "summary": "Staff received phishing emails.",
    "confidence_score": 0.9,
})


class FakeProvider(AIProvider):
    """Stands in for Gemini: always gives the same reply and records every prompt."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return ProviderReply(text=self.reply)


def fake_app(reply=AI_REPLY):
    """A test client with a fake Gemini, no retry delays and a temporary
    tickets.json. Returns (client, prompts sent to the fake, repository);
    client.data_path is the temporary tickets.json."""
    folder = tempfile.mkdtemp()
    provider = FakeProvider(reply)
    repository = JsonTicketRepository(os.path.join(folder, "tickets.json"),
                                      os.path.join(folder, "sampleTickets.json"))
    manager = FlaskIOManager(
        ai_mgr=AIManager(provider, retry_policy=RetryPolicy(0, 0, 0, 0)),
        repository=repository,
        auto_open=False,
    )
    client = manager.app.test_client()
    client.data_path = os.path.join(folder, "tickets.json")
    return client, provider.prompts, repository


def requester_details(form):
    """(requester details, problem) for a form, like the old helper returned."""
    data, problem = TicketFormValidator().validate(form)
    return (data["requester"] if data else None), problem


# ---------- requester details (name, contact, title) ----------

def test_valid_details_are_cleaned():
    details, problem = requester_details(dict(FORM, name="  Tan   Wei Ming "))
    assert problem is None
    assert details == {"name": "Tan Wei Ming", "contact_number": "+65 9123 4567",
                       "ticket_title": "Phishing emails"}


def test_missing_details_are_rejected():
    for field in ("name", "contact_number", "ticket_title"):
        for bad in ("", "   ", None, 42):
            details, problem = requester_details(dict(FORM, **{field: bad}))
            assert details is None and problem, (field, bad)


def test_contact_number_format():
    for good in ("91234567", "+65 9123 4567", "6123-4567", "+6581234567", "3123 4567"):
        assert requester_details(dict(FORM, contact_number=good))[1] is None, good
    for bad in ("1234", "call me", "+65 9123 456x", "1" * 25,
                "9000123412",      # too long
                "9123456",         # too short
                "51234567",        # Singapore numbers don't start with 5
                "+60 9123 4567"):  # not a Singapore code
        assert requester_details(dict(FORM, contact_number=bad))[1], bad


def test_card_number_in_title_is_redacted():
    details, _ = requester_details(dict(FORM, ticket_title="Card 4111 1111 1111 1111 declined"))
    assert "4111" not in details["ticket_title"] and "[CARD_NUMBER]" in details["ticket_title"]


# ---------- submitting a ticket ----------

def test_ticket_is_saved_with_requester_and_unresolved():
    client, prompts, repository = fake_app()
    response = client.post("/submit-ticket", json=FORM)
    assert response.status_code == 200
    ticket = response.get_json()["ticket"]
    assert ticket["name"] == "Tan Wei Ming"
    assert ticket["ticket_title"] == "Phishing emails"
    assert ticket["priority"] == "P1" and ticket["department"] == "Cyber"
    assert ticket["is_resolved"] is False and ticket["resolved_at"] is None
    assert "problem_description" not in ticket      # raw text never stored
    # Gemini sees the title and description, never the name or number
    assert "Phishing emails" in prompts[0]
    assert "Tan Wei Ming" not in prompts[0] and "9123" not in prompts[0]


def test_bad_form_never_reaches_the_ai():
    client, prompts, repository = fake_app()
    response = client.post("/submit-ticket", json=dict(FORM, contact_number="nope"))
    assert response.status_code == 400
    assert prompts == []
    assert repository.load_all() == []


# ---------- resolving and reopening ----------

def test_resolve_and_reopen():
    client, _, repository = fake_app()
    client.post("/submit-ticket", json=FORM)
    url = "/api/tickets/T-0001/toggle-resolution"

    data = client.post(url, json={"is_resolved": True}).get_json()
    assert data["status"] == "success" and data["resolved_at"]
    assert repository.load_all()[0]["is_resolved"] is True

    data = client.post(url, json={"is_resolved": False}).get_json()
    assert data["is_resolved"] is False
    saved = repository.load_all()[0]
    assert saved["is_resolved"] is False and saved["resolved_at"] is None


def test_resolve_errors():
    client, _, repository = fake_app()
    client.post("/submit-ticket", json=FORM)
    assert client.post("/api/tickets/T-9999/toggle-resolution",
                       json={"is_resolved": True}).status_code == 404
    assert client.post("/api/tickets/T-0001/toggle-resolution",
                       json={"is_resolved": "yes"}).status_code == 400


def test_corrupt_file_is_never_overwritten():
    client, _, repository = fake_app()
    with open(client.data_path, "w") as file:
        file.write("{ corrupt")
    assert client.post("/api/tickets/T-0001/toggle-resolution",
                       json={"is_resolved": True}).status_code == 503
    assert client.post("/submit-ticket", json=FORM).status_code == 503
    with open(client.data_path) as file:
        assert file.read() == "{ corrupt"


def test_dashboard_pages():
    client, _, repository = fake_app()
    assert client.get("/helpdesk").status_code == 200
    assert client.get("/departments/Cyber").status_code == 200
    assert client.get("/departments/Nowhere").status_code == 404


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