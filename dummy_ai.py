"""Dummy output to check that ai_manager works, with NO real Gemini call.

Only Gemini is faked (AIManager is given a CannedProvider in place of the
GeminiProvider). Everything else is the real code path: sanitising, security
check, rate limit, redaction, prompt building, JSON parsing, schema validation
and the final result. Tickets the security check rejects never reach the fake.
Successful tickets are then passed through logic_manager (priority, routing).
Nothing is saved: the demo never touches data/tickets.json.

    python main.py --demo
"""

import json
import os

# ai_manager refuses to import without a key; the fake never uses it.
os.environ.setdefault("GEMINI_API_KEY", "offline-demo-key")

from ai_manager import (  # noqa: E402
    AIManager, AIProvider, ProviderReply, ResponseValidator,
)
from logic_manager import LogicManager  # noqa: E402
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


class CannedProvider(AIProvider):
    """Stands in for GeminiProvider: returns the JSON text Gemini would."""

    def generate(self, prompt: str) -> ProviderReply:
        text = prompt.lower()
        for keyword, category, severity, scope, summary, score in CANNED:
            if keyword in text:
                return ProviderReply(text=json.dumps({
                    "category": category, "severity": severity,
                    "affected_scope": scope, "summary": summary,
                    "confidence_score": score,
                }))
        return ProviderReply(text=json.dumps({
            "category": "other", "severity": "low", "affected_scope": "individual",
            "summary": "Unclear issue.", "confidence_score": 0.3,
        }))


def run_demo() -> bool:
    say = print
    say("Running DUMMY AI Manager demo (fake Gemini, real pipeline)...")

    ai, logic, validator = AIManager(CannedProvider()), LogicManager(), ResponseValidator()
    passed = 0
    for record in test_ai.sample_records:
        result = ai.classify(record, user_id="demo").to_dict()

        say("\n" + "=" * 60)
        say("TICKET: " + record["problem_description"])
        say(json.dumps(result, indent=4))

        if result["status"] == "success":
            assert validator.is_valid(result["data"]), result
            decision = logic.process(result).to_dict()
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