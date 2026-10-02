"""
test_logic_manager.py - offline tests for logic_manager.py

HOW TO RUN:   python test_logic_manager.py
    - Needs NO API key and NO internet: every "AI response" below is
      hardcoded, shaped exactly like ai_manager.classify_ticket() output.
    - Procedural on purpose: plain functions + assert, no classes
      (unittest.TestCase would break the "no class definitions" rule).
    - Any function whose name starts with test_ is run automatically.
    - Exits with code 1 if any test fails (so Docker / CI notice).
"""

import sys

import logic_manager as lm


# ============================================================
# HELPERS - build sample inputs (override only what a test cares about)
# ============================================================

def make_record(**changes) -> dict:
    """A typical ticket as io_manager would pass it (low-impact by default)."""
    record = {
        "ticket_title": "Test ticket",
        "problem_description": "Something is wrong.",
        "affected_service": "Office printer",
        "affected_users": "individual",
        "work_blocked": "no",
        "workaround_available": "yes",
    }
    record.update(changes)
    return record


def make_ai_result(**changes) -> dict:
    """A successful ai_manager result with hardcoded classification data."""
    data = {
        "category": "hardware",
        "severity": "low",
        "affected_scope": "individual",
        "summary": "Printer is jammed.",
        "confidence_score": 0.9,
    }
    data.update(changes)
    return {"status": "success", "data": data}


# ============================================================
# CONFIGURATION CONSISTENCY
# ============================================================

def test_routing_covers_every_ai_category():
    # Must mirror ai_manager.VALID_CATEGORIES (copied here so this test
    # never imports ai_manager, which needs an API key to load).
    ai_categories = {
        "hardware", "software", "network", "account_access",
        "cybersecurity", "hr", "finance", "other",
    }
    assert set(lm.ROUTING.keys()) == ai_categories


def test_every_priority_has_an_sla():
    for priority in lm.PRIORITY_ORDER:
        sla = lm.get_sla(priority)
        assert sla["response_minutes"] > 0
        assert sla["resolution_hours"] > 0


def test_get_sla_returns_a_copy():
    sla = lm.get_sla("P1")
    sla["response_minutes"] = 9999
    assert lm.get_sla("P1")["response_minutes"] == 15


# ============================================================
# SMALL HELPERS
# ============================================================

def test_parse_yes_no():
    assert lm.parse_yes_no("yes") is True
    assert lm.parse_yes_no("Yes ") is True
    assert lm.parse_yes_no(True) is True
    assert lm.parse_yes_no("no") is False
    assert lm.parse_yes_no(False) is False
    assert lm.parse_yes_no(None) is False


def test_parse_reported_scope():
    assert lm.parse_reported_scope("department") == "department"
    assert lm.parse_reported_scope("Organisation") == "organisation"
    assert lm.parse_reported_scope(1) == "individual"
    assert lm.parse_reported_scope("35") == "multiple_users"
    assert lm.parse_reported_scope("50+") == "multiple_users"
    assert lm.parse_reported_scope("lots") is None
    assert lm.parse_reported_scope(0) is None
    assert lm.parse_reported_scope(None) is None


def test_score_to_priority_boundaries():
    assert lm.score_to_priority(100) == "P1"
    assert lm.score_to_priority(70) == "P1"
    assert lm.score_to_priority(69) == "P2"
    assert lm.score_to_priority(50) == "P2"
    assert lm.score_to_priority(49) == "P3"
    assert lm.score_to_priority(30) == "P3"
    assert lm.score_to_priority(29) == "P4"
    assert lm.score_to_priority(0) == "P4"


def test_score_is_capped_at_100():
    score = lm.calculate_priority_score(
        "critical", "organisation", True, False, "cybersecurity", True
    )
    assert score == 100


def test_more_urgent_picks_the_lower_p_number():
    assert lm.more_urgent("P1", "P3") == "P1"
    assert lm.more_urgent("P4", "P2") == "P2"
    assert lm.more_urgent("P2", "P2") == "P2"


# ============================================================
# BUSINESS RULES (one rule at a time)
# ============================================================

def test_scope_is_raised_when_user_reports_bigger_scope():
    assert lm.reconcile_scope("individual", "department") == ("department", True)


def test_scope_is_never_lowered():
    assert lm.reconcile_scope("organisation", "individual") == ("organisation", False)


def test_scope_unchanged_when_user_answer_unknown():
    assert lm.reconcile_scope("department", None) == ("department", False)


def test_critical_service_needs_keyword_and_high_severity():
    assert lm.is_critical_service_hit("Core banking application", "x", "critical") is True
    assert lm.is_critical_service_hit("Core banking application", "x", "low") is False
    assert lm.is_critical_service_hit("Shared drive", "x", "critical") is False


def test_cybersecurity_override_floor_is_p2_for_one_user():
    assert lm.apply_priority_overrides("P4", "cybersecurity", "individual")[0] == "P2"


def test_cybersecurity_override_is_p1_beyond_one_user():
    assert lm.apply_priority_overrides("P3", "cybersecurity", "department")[0] == "P1"


def test_override_never_lowers_priority():
    priority, reason = lm.apply_priority_overrides("P1", "cybersecurity", "individual")
    assert priority == "P1" and reason is None


def test_non_security_ticket_has_no_override():
    assert lm.apply_priority_overrides("P4", "hardware", "individual") == ("P4", None)


def test_escalation_rules():
    assert lm.check_escalation("P1", "low", "individual")[0] is True
    assert lm.check_escalation("P2", "high", "department")[0] is True
    assert lm.check_escalation("P2", "critical", "organisation")[0] is True
    assert lm.check_escalation("P2", "high", "individual")[0] is False
    assert lm.check_escalation("P3", "medium", "department")[0] is False


def test_review_rules():
    assert lm.check_needs_review(0.59, False)[0] is True    # low confidence
    assert lm.check_needs_review(0.60, False)[0] is False   # exactly at threshold
    assert lm.check_needs_review(0.95, True)[0] is True     # scope disagreement
    assert lm.check_needs_review(0.95, False)[0] is False


def test_outcome_priority_order():
    assert lm.decide_outcome(True, True) == "escalated"   # urgency wins
    assert lm.decide_outcome(False, True) == "flagged"
    assert lm.decide_outcome(False, False) == "routed"


# ============================================================
# END-TO-END: process_ticket with hardcoded AI responses
# ============================================================

def test_org_wide_banking_outage_is_p1_and_escalated():
    record = make_record(
        affected_service="Mobile banking app",
        affected_users="organisation",
        work_blocked="yes",
        workaround_available="no",
    )
    ai = make_ai_result(
        category="software", severity="critical",
        affected_scope="organisation", summary="Banking app login fails for all users.",
    )
    result = lm.process_ticket(record, ai)
    assert result["priority"] == "P1"
    assert result["priority_score"] == 90
    assert result["outcome"] == "escalated"
    assert result["escalate"] is True
    assert result["assigned_department"] == "Application Support"
    assert result["sla_response_minutes"] == 15


def test_department_server_outage_is_p2_and_escalated():
    record = make_record(
        affected_service="Server X", affected_users="department",
        work_blocked="yes", workaround_available="no",
    )
    ai = make_ai_result(category="network", severity="high", affected_scope="department")
    result = lm.process_ticket(record, ai)
    assert result["priority"] == "P2"
    assert result["escalate"] is True
    assert result["outcome"] == "escalated"
    assert result["assigned_department"] == "Network Operations"


def test_minor_individual_issue_is_p4_and_auto_routed():
    result = lm.process_ticket(make_record(), make_ai_result())
    assert result["priority"] == "P4"
    assert result["outcome"] == "routed"
    assert result["escalate"] is False
    assert result["requires_human_review"] is False
    assert result["assigned_department"] == "Desktop & Hardware Support"


def test_single_user_phishing_gets_p2_floor_and_goes_to_soc():
    ai = make_ai_result(category="cybersecurity", severity="medium")
    result = lm.process_ticket(make_record(affected_service="Email"), ai)
    assert result["priority"] == "P2"
    assert result["assigned_department"] == "Security Operations Centre (SOC)"
    assert result["escalate"] is False


def test_multi_user_cybersecurity_is_forced_to_p1():
    record = make_record(affected_users="multiple_users")
    ai = make_ai_result(
        category="cybersecurity", severity="medium", affected_scope="multiple_users"
    )
    result = lm.process_ticket(record, ai)
    assert result["priority"] == "P1"
    assert result["outcome"] == "escalated"


def test_low_confidence_ticket_is_flagged_for_review():
    result = lm.process_ticket(make_record(), make_ai_result(confidence_score=0.4))
    assert result["outcome"] == "flagged"
    assert result["requires_human_review"] is True


def test_ai_underestimating_scope_is_corrected_and_flagged():
    record = make_record(affected_users="department")
    ai = make_ai_result(affected_scope="individual")     # AI under-estimated
    result = lm.process_ticket(record, ai)
    assert result["affected_scope"] == "department"
    assert result["requires_human_review"] is True


def test_core_banking_keyword_adds_priority():
    base = dict(work_blocked="yes", workaround_available="no")
    ai = make_ai_result(category="software", severity="high")
    banking = lm.process_ticket(make_record(affected_service="Core banking system", **base), ai)
    other = lm.process_ticket(make_record(affected_service="Shared drive", **base), ai)
    assert banking["priority_score"] == other["priority_score"] + lm.CRITICAL_SERVICE_POINTS


def test_numeric_user_count_still_works():
    record = make_record(affected_users=200)
    result = lm.process_ticket(record, make_ai_result(affected_scope="organisation"))
    assert result["affected_scope"] == "organisation"     # AI value kept


# ============================================================
# ERROR HANDLING (feeds the report's exception handling matrix)
# ============================================================

def test_invalid_ticket_is_rejected():
    ai = {"status": "invalid", "reason": "Prompt injection detected."}
    result = lm.process_ticket(make_record(), ai)
    assert result["outcome"] == "rejected"
    assert result["priority"] is None
    assert result["security_alert"] is False


def test_blocked_ticket_is_rejected_with_security_alert():
    ai = {"status": "blocked", "reason": "Credential request."}
    result = lm.process_ticket(make_record(), ai)
    assert result["outcome"] == "rejected"
    assert result["security_alert"] is True


def test_ai_failure_goes_to_manual_triage_not_lost():
    ai = {"status": "error", "reason": "AI returned no valid classification."}
    result = lm.process_ticket(make_record(), ai)
    assert result["outcome"] == "flagged"
    assert result["priority"] == lm.AI_FAILURE_PRIORITY
    assert result["assigned_department"] == lm.GENERAL_DEPARTMENT
    assert result["requires_human_review"] is True


def test_success_with_missing_fields_does_not_crash():
    ai = {"status": "success", "data": {"category": "hardware"}}
    result = lm.process_ticket(make_record(), ai)
    assert result["outcome"] == "flagged"
    assert "missing" in result["decision_reasons"]


def test_garbage_ai_result_does_not_crash():
    for garbage in (None, "oops", 42, [], {}):
        result = lm.process_ticket(make_record(), garbage)
        assert result["outcome"] == "flagged"


def test_missing_record_fields_do_not_crash():
    result = lm.process_ticket({}, make_ai_result())
    assert result["outcome"] in ("routed", "flagged", "escalated")


def test_every_decision_has_the_same_keys():
    # data_manager relies on one consistent layout for every saved record.
    results = [
        lm.process_ticket(make_record(), make_ai_result()),
        lm.process_ticket(make_record(), {"status": "invalid", "reason": "x"}),
        lm.process_ticket(make_record(), {"status": "error", "reason": "x"}),
    ]
    assert results[0].keys() == results[1].keys() == results[2].keys()


def test_same_input_gives_same_output():
    record, ai = make_record(), make_ai_result(severity="high")
    assert lm.process_ticket(record, ai) == lm.process_ticket(record, ai)


# ============================================================
# TEST RUNNER
# ============================================================

def run_all_tests() -> int:
    """Run every test_ function, print PASS/FAIL, return the failure count."""
    tests = [
        (name, func)
        for name, func in sorted(globals().items())
        if name.startswith("test_") and callable(func)
    ]

    failures = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except AssertionError:
            failures += 1
            print(f"FAIL  {name}")
        except Exception as error:   # a crash is also a failure
            failures += 1
            print(f"ERROR {name}: {error!r}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run_all_tests() else 0)