# test_logic_manager.py - offline tests for logic_manager.py

# Run with:  python test_logic_manager.py
# No API key or internet needed: every AI result below is hardcoded.

import sys

import logic_manager as lm


def make_ai_result(category="hardware", severity="low", scope="individual",
                   confidence=0.9, flagged=False):
    """A fake ai_manager result. Change only what a test cares about."""
    return {
        "status": "success",
        "data": {
            "category": category,
            "severity": severity,
            "affected_scope": scope,
            "summary": "Test summary.",
            "confidence_score": confidence,
        },
        "flagged": flagged,
    }


def make_ticket(ticket_id, priority_number, score, submitted_at):
    """A minimal saved ticket, just enough for sorting."""
    return {
        "ticket_id": ticket_id,
        "priority_number": priority_number,
        "priority_score": score,
        "submitted_at": submitted_at,
    }


# ---------- scoring ----------

def test_score_adds_severity_and_scope():
    assert lm.calculate_priority_score("high", "department", "network") == 60


def test_cyber_gets_bonus_points():
    assert lm.calculate_priority_score("medium", "individual", "cybersecurity") == 35


def test_score_to_priority_boundaries():
    assert lm.score_to_priority(70) == "P1"
    assert lm.score_to_priority(69) == "P2"
    assert lm.score_to_priority(50) == "P2"
    assert lm.score_to_priority(49) == "P3"
    assert lm.score_to_priority(35) == "P3"
    assert lm.score_to_priority(34) == "P4"
    assert lm.score_to_priority(25) == "P4"
    assert lm.score_to_priority(24) == "P5"


def test_every_priority_is_reachable():
    # Try every severity/scope combination, with and without cyber
    reached = set()
    for category in ("hardware", "cybersecurity"):
        for severity in lm.SEVERITY_POINTS:
            for scope in lm.SCOPE_POINTS:
                score = lm.calculate_priority_score(severity, scope, category)
                reached.add(lm.score_to_priority(score))
    assert reached == {"P1", "P2", "P3", "P4", "P5"}


# ---------- cyber override ----------

def test_cyber_beyond_one_person_is_p1():
    assert lm.apply_cyber_override("P3", "cybersecurity", "multiple_users") == "P1"


def test_cyber_for_one_person_keeps_its_priority():
    assert lm.apply_cyber_override("P3", "cybersecurity", "individual") == "P3"


def test_non_cyber_is_never_overridden():
    assert lm.apply_cyber_override("P5", "hardware", "organisation") == "P5"


# ---------- routing ----------

def test_routing():
    assert lm.get_department("cybersecurity") == "Cyber"
    assert lm.get_department("hr") == "HR"
    assert lm.get_department("finance") == "Finance"
    for category in ("hardware", "software", "network", "account_access", "other"):
        assert lm.get_department(category) == "Infra"


def test_routing_only_uses_the_four_departments():
    assert set(lm.ROUTING.values()) == set(lm.DEPARTMENTS)


# ---------- review flag ----------

def test_review_flag():
    assert lm.needs_human_review(0.59, False) is True    # unsure
    assert lm.needs_human_review(0.60, False) is False   # exactly at threshold
    assert lm.needs_human_review(0.95, True) is True     # AI flagged it
    assert lm.needs_human_review(0.95, False) is False


# ---------- process_ticket (the worked examples) ----------

def test_core_banking_outage_is_p1_infra():
    result = lm.process_ticket(make_ai_result("software", "critical", "organisation"))
    assert result["priority_score"] == 85
    assert result["priority"] == "P1"
    assert result["priority_number"] == 1
    assert result["department"] == "Infra"


def test_department_server_down_is_p2():
    result = lm.process_ticket(make_ai_result("network", "high", "department"))
    assert result["priority_score"] == 60
    assert result["priority"] == "P2"


def test_single_user_phishing_is_p3_cyber():
    result = lm.process_ticket(make_ai_result("cybersecurity", "medium", "individual"))
    assert result["priority"] == "P3"
    assert result["department"] == "Cyber"


def test_multi_user_phishing_is_forced_to_p1():
    result = lm.process_ticket(make_ai_result("cybersecurity", "medium", "multiple_users"))
    assert result["priority_score"] == 45     # score alone would be P3
    assert result["priority"] == "P1"


def test_outlook_crash_is_p4():
    result = lm.process_ticket(make_ai_result("software", "medium", "individual"))
    assert result["priority"] == "P4"


def test_jammed_printer_is_p5():
    result = lm.process_ticket(make_ai_result("hardware", "low", "individual"))
    assert result["priority"] == "P5"
    assert result["needs_review"] is False


def test_low_confidence_is_flagged_for_review():
    result = lm.process_ticket(make_ai_result(confidence=0.4))
    assert result["needs_review"] is True


def test_ai_fields_are_copied_into_the_decision():
    result = lm.process_ticket(make_ai_result("hr", "low", "individual", 0.8))
    assert result["category"] == "hr"
    assert result["severity"] == "low"
    assert result["affected_scope"] == "individual"
    assert result["summary"] == "Test summary."
    assert result["confidence_score"] == 0.8


# ---------- queue order ----------

def test_queue_highest_priority_first():
    tickets = [
        make_ticket("C", 3, 40, "2026-10-01T09:00"),
        make_ticket("A", 1, 85, "2026-10-01T10:00"),
        make_ticket("E", 5, 10, "2026-10-01T08:00"),
        make_ticket("B", 2, 60, "2026-10-01T11:00"),
        make_ticket("D", 4, 25, "2026-10-01T07:00"),
    ]
    assert [t["ticket_id"] for t in lm.sort_queue(tickets)] == ["A", "B", "C", "D", "E"]


def test_queue_higher_score_wins_a_tie():
    tickets = [make_ticket("low", 2, 50, "a"), make_ticket("high", 2, 65, "b")]
    assert [t["ticket_id"] for t in lm.sort_queue(tickets)] == ["high", "low"]


def test_queue_older_ticket_wins_a_full_tie():
    tickets = [make_ticket("new", 3, 40, "2026-10-02"), make_ticket("old", 3, 40, "2026-10-01")]
    assert [t["ticket_id"] for t in lm.sort_queue(tickets)] == ["old", "new"]


def test_new_ticket_slots_into_place():
    queue = [make_ticket("P1", 1, 85, "a"), make_ticket("P3", 3, 40, "b")]
    queue.append(make_ticket("new-P2", 2, 60, "c"))
    assert [t["ticket_id"] for t in lm.sort_queue(queue)] == ["P1", "new-P2", "P3"]


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