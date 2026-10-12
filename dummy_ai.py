"""Dummy output to check that ai_manager works, with NO real Gemini call.

Only Gemini is faked (ai_manager.call_api is swapped for fake_call_api).
Everything else is the real code path: sanitising, security check, rate
limit, redaction, prompt building, JSON parsing, schema validation and the
final result dict. Tickets that security_check rejects never reach the fake.
Successful tickets are then passed through logic_manager (priority, routing).
Nothing is saved: the demo never touches data/tickets.json.

    python main.py --demo
"""

import json
import os
from unittest.mock import patch

# ai_manager refuses to import without a key; the fake never uses it.
os.environ.setdefault("GEMINI_API_KEY", "offline-demo-key")

import ai_manager  # noqa: E402
import io_manager  # noqa: E402
import logic_manager  # noqa: E402
import test_ai  # noqa: E402  (reuses its sample tickets and comparison)

# keyword in the (redacted) prompt -> canned model reply
CANNED = [
    ("core banking", "software", "critical", "organisation",
     "Core banking system unavailable across multiple branches.", 0.95),
    ("wifi", "network", "high", "department",
     "Staff in a branch cannot connect to the office WiFi.", 0.93),
    ("password", "account_access", "medium", "individual",
     "User forgot a privileged account password and needs a reset.", 0.9),
    ("outlook", "software", "medium", "individual",
     "Outlook closes when an attachment is opened.", 0.88),
    ("printer", "hardware", "low", "individual",
     "A printer is jammed and shows a paper error.", 0.9),
]


def fake_call_api(prompt):
    """Stands in for ai_manager.call_api: returns the JSON text Gemini would."""
    text = prompt.lower()
    for keyword, category, severity, scope, summary, score in CANNED:
        if keyword in text:
            return json.dumps({
                "category": category, "severity": severity,
                "affected_scope": scope, "summary": summary,
                "confidence_score": score,
            })
    return json.dumps({
        "category": "other", "severity": "low", "affected_scope": "individual",
        "summary": "Unclear issue.", "confidence_score": 0.3,
    })


def run_demo():
    say = io_manager.print_message
    say("Running DUMMY AI Manager demo (fake Gemini, real pipeline)...")

    ai_manager._request_times.clear()
    passed = 0
    with patch.object(ai_manager, "call_api", fake_call_api):
        for record in test_ai.sample_records:
            result = ai_manager.classify_ticket(record, user_id="demo")

            say("\n" + "=" * 60)
            say("TICKET: " + record["problem_description"])
            say(json.dumps(result, indent=4))

            if result["status"] == "success":
                assert ai_manager.validate_response(result["data"]), result
                decision = logic_manager.process_ticket(result)
                say(f"  LOGIC: {decision['priority']} (score "
                    f"{decision['priority_score']}) -> {decision['department']}, "
                    f"human review: {decision['needs_review']}")

            problems = test_ai.matches_expected(result, record["expected"])
            say("  RESULT: " + ("as expected" if not problems
                                else "MISMATCH -> " + "; ".join(problems)))
            passed += not problems

    total = len(test_ai.sample_records)
    say("\n" + "=" * 60)
    say(f"{passed}/{total} tickets behaved as expected.")
    return passed == total
