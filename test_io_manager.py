# test_io_manager.py - offline tests for io_manager.py (form checks and routes)

# Run with:  python test_io_manager.py
# No API key or internet needed: Gemini is replaced by a fake reply, and a
# temporary folder is used, so your real data/tickets.json is never touched.

import json
import os
import sys
import tempfile
from unittest.mock import patch

# ai_manager refuses to import without a key; the fake reply never uses it.
os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")

import ai_manager  # noqa: E402
import data_manager  # noqa: E402
import io_manager  # noqa: E402

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


def fake_app(reply=AI_REPLY):
    """A test client with a fake Gemini, a temporary tickets.json and no
    retry delays. Returns (client, prompts sent to the fake, patches)."""
    prompts = []

    def fake_call_api(prompt):
        prompts.append(prompt)
        return reply

    folder = tempfile.mkdtemp()
    patches = [
        patch.object(ai_manager, "call_api", fake_call_api),
        patch.object(ai_manager, "RETRY_DELAY_SECONDS", 0),
        patch.object(data_manager, "DATA_FILE", os.path.join(folder, "tickets.json")),
    ]
    for p in patches:
        p.start()
    ai_manager._request_times.clear()
    return io_manager.app.test_client(), prompts, patches


def stop(patches):
    for p in patches:
        p.stop()


# ---------- requester details (name, contact, title) ----------

def test_valid_details_are_cleaned():
    details, problem = io_manager.requester_details(dict(FORM, name="  Tan   Wei Ming "))
    assert problem is None
    assert details == {"name": "Tan Wei Ming", "contact_number": "+65 9123 4567",
                       "ticket_title": "Phishing emails"}


def test_missing_details_are_rejected():
    for field in ("name", "contact_number", "ticket_title"):
        for bad in ("", "   ", None, 42):
            details, problem = io_manager.requester_details(dict(FORM, **{field: bad}))
            assert details is None and problem, (field, bad)


def test_contact_number_format():
    for good in ("91234567", "+65 9123 4567", "6123-4567"):
        assert io_manager.requester_details(dict(FORM, contact_number=good))[1] is None, good
    for bad in ("1234", "call me", "+65 9123 456x", "1" * 25):
        assert io_manager.requester_details(dict(FORM, contact_number=bad))[1], bad


def test_card_number_in_title_is_redacted():
    details, _ = io_manager.requester_details(dict(FORM, ticket_title="Card 4111 1111 1111 1111 declined"))
    assert "4111" not in details["ticket_title"] and "[CARD_NUMBER]" in details["ticket_title"]


# ---------- submitting a ticket ----------

def test_ticket_is_saved_with_requester_and_unresolved():
    client, prompts, patches = fake_app()
    try:
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
    finally:
        stop(patches)


def test_bad_form_never_reaches_the_ai():
    client, prompts, patches = fake_app()
    try:
        response = client.post("/submit-ticket", json=dict(FORM, contact_number="nope"))
        assert response.status_code == 400
        assert prompts == []
        assert data_manager.load_records() == []
    finally:
        stop(patches)


# ---------- resolving and reopening ----------

def test_resolve_and_reopen():
    client, _, patches = fake_app()
    try:
        client.post("/submit-ticket", json=FORM)
        url = "/api/tickets/T-0001/toggle-resolution"

        data = client.post(url, json={"is_resolved": True}).get_json()
        assert data["status"] == "success" and data["resolved_at"]
        assert data_manager.load_records()[0]["is_resolved"] is True

        data = client.post(url, json={"is_resolved": False}).get_json()
        assert data["is_resolved"] is False
        saved = data_manager.load_records()[0]
        assert saved["is_resolved"] is False and saved["resolved_at"] is None
    finally:
        stop(patches)


def test_resolve_errors():
    client, _, patches = fake_app()
    try:
        client.post("/submit-ticket", json=FORM)
        assert client.post("/api/tickets/T-9999/toggle-resolution",
                           json={"is_resolved": True}).status_code == 404
        assert client.post("/api/tickets/T-0001/toggle-resolution",
                           json={"is_resolved": "yes"}).status_code == 400
    finally:
        stop(patches)


def test_corrupt_file_is_never_overwritten():
    client, _, patches = fake_app()
    try:
        os.makedirs(os.path.dirname(data_manager.DATA_FILE), exist_ok=True)
        with open(data_manager.DATA_FILE, "w") as file:
            file.write("{ corrupt")
        assert client.post("/api/tickets/T-0001/toggle-resolution",
                           json={"is_resolved": True}).status_code == 503
        assert client.post("/submit-ticket", json=FORM).status_code == 503
        with open(data_manager.DATA_FILE) as file:
            assert file.read() == "{ corrupt"
    finally:
        stop(patches)


def test_dashboard_pages():
    client, _, patches = fake_app()
    try:
        assert client.get("/helpdesk").status_code == 200
        assert client.get("/departments/Cyber").status_code == 200
        assert client.get("/departments/Nowhere").status_code == 404
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