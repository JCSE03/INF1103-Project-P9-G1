import sys

import logic_manager as lm


def make_ai_result(category="hardware", severity="low", scope="individual",
                   confidence=0.9, flagged=False):
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


def make_facts(category="hardware", severity="low", scope="individual",
               confidence=0.9, flagged=False):
    return lm.AIFacts.from_ai_result(
        make_ai_result(category, severity, scope, confidence, flagged))


def make_ticket(ticket_id, priority_number, score, submitted_at):
    return {
        "ticket_id": ticket_id,
        "priority_number": priority_number,
        "priority_score": score,
        "submitted_at": submitted_at,
    }


def raises(error_type, function, *args):
    try:
        function(*args)
    except error_type:
        return True
    return False


manager = lm.LogicManager()
rubric = lm.ScoringRubric()


def test_facts_read_from_ai_result():
    facts = make_facts("network", "high", "department", 0.75, True)
    assert facts.category == "network"
    assert facts.severity == "high"
    assert facts.scope == "department"
    assert facts.confidence == 0.75
    assert facts.flagged is True


def test_facts_reject_a_failed_ai_result():
    assert raises(lm.InvalidAIResultError, lm.AIFacts.from_ai_result,
                  {"status": "error", "reason": "API down"})


def test_facts_reject_missing_data_or_fields():
    assert raises(lm.InvalidAIResultError, lm.AIFacts.from_ai_result, {"status": "success"})
    broken = make_ai_result()
    del broken["data"]["severity"]
    assert raises(lm.InvalidAIResultError, lm.AIFacts.from_ai_result, broken)


def test_facts_reject_a_non_number_confidence():
    assert raises(lm.InvalidAIResultError, lm.AIFacts.from_ai_result,
                  make_ai_result(confidence="very sure"))


def test_facts_reject_something_that_is_not_a_dict():
    assert raises(lm.InvalidAIResultError, lm.AIFacts.from_ai_result, None)
    assert raises(lm.InvalidAIResultError, lm.AIFacts.from_ai_result, "not a dict")


def test_score_adds_severity_and_scope():
    assert rubric.score(make_facts("network", "high", "department")) == 60


def test_cyber_gets_bonus_points():
    assert rubric.score(make_facts("cybersecurity", "medium", "individual")) == 35


def test_score_to_priority_boundaries():
    expected = {70: "P1", 69: "P2", 50: "P2", 49: "P3", 35: "P3",
                34: "P4", 25: "P4", 24: "P5"}
    for score, priority in expected.items():
        assert rubric.priority_for(score) == priority, score


def test_every_priority_is_reachable():
    reached = set()
    for category in ("hardware", "cybersecurity"):
        for severity in rubric.severities:
            for scope in rubric.scopes:
                score = rubric.score(make_facts(category, severity, scope))
                reached.add(rubric.priority_for(score))
    assert reached == {"P1", "P2", "P3", "P4", "P5"}


def test_unknown_severity_or_scope_is_rejected():
    assert raises(lm.InvalidAIResultError, rubric.score, make_facts(severity="extreme"))
    assert raises(lm.InvalidAIResultError, rubric.score, make_facts(scope="planet"))


def test_custom_rubric_without_editing_the_class():
    strict = lm.ScoringRubric(cyber_bonus=0, thresholds=[("P4", 5), ("P1", 90)])
    assert strict.score(make_facts("cybersecurity", "medium", "individual")) == 25
    assert strict.priority_for(95) == "P1"
    assert strict.priority_for(10) == "P4"
    assert strict.priority_for(4) == "P5"


def test_routing():
    router = lm.Router()
    assert router.department_for("cybersecurity") == "Cyber"
    assert router.department_for("hr") == "HR"
    assert router.department_for("finance") == "Finance"
    for category in ("hardware", "software", "network", "account_access", "other"):
        assert router.department_for(category) == "Infra"


def test_departments_are_the_four_in_order():
    assert lm.Router().departments == ["Cyber", "HR", "Finance", "Infra"]


def test_unknown_category_is_rejected():
    assert raises(lm.InvalidAIResultError, lm.Router().department_for, "plumbing")


def test_cyber_beyond_one_person_is_p1():
    rule = lm.CyberMultiUserRule()
    assert rule.adjust("P3", make_facts("cybersecurity", scope="multiple_users")) == "P1"


def test_cyber_for_one_person_keeps_its_priority():
    rule = lm.CyberMultiUserRule()
    assert rule.adjust("P3", make_facts("cybersecurity", scope="individual")) == "P3"


def test_non_cyber_is_never_overridden():
    rule = lm.CyberMultiUserRule()
    assert rule.adjust("P5", make_facts("hardware", scope="organisation")) == "P5"


def test_every_priority_rule_keeps_the_contract():
    valid = {"P1", "P2", "P3", "P4", "P5"}
    for rule in (lm.CyberMultiUserRule(),):
        for category in ("hardware", "cybersecurity"):
            for scope in rubric.scopes:
                for priority in valid:
                    assert rule.adjust(priority, make_facts(category, scope=scope)) in valid


def test_low_confidence_rule():
    rule = lm.LowConfidenceRule()
    assert rule.check(make_facts(confidence=0.59)) == rule.reason
    assert rule.check(make_facts(confidence=0.60)) is None


def test_low_confidence_threshold_can_be_changed():
    rule = lm.LowConfidenceRule(threshold=0.8)
    assert rule.check(make_facts(confidence=0.75)) is not None
    assert "80%" in rule.reason


def test_ai_flagged_rule():
    rule = lm.AIFlaggedRule()
    assert rule.check(make_facts(flagged=True)) == rule.reason
    assert rule.check(make_facts(flagged=False)) is None


def test_abstract_classes_cannot_be_created():
    assert raises(TypeError, lm.PriorityRule)
    assert raises(TypeError, lm.ReviewRule, "some reason")


def test_core_banking_outage_is_p1_infra():
    decision = manager.process(make_ai_result("software", "critical", "organisation"))
    assert decision.priority_score == 85
    assert decision.priority == "P1"
    assert decision.priority_number == 1
    assert decision.department == "Infra"


def test_department_server_down_is_p2():
    decision = manager.process(make_ai_result("network", "high", "department"))
    assert decision.priority_score == 60
    assert decision.priority == "P2"


def test_single_user_phishing_is_p3_cyber():
    decision = manager.process(make_ai_result("cybersecurity", "medium", "individual"))
    assert decision.priority == "P3"
    assert decision.department == "Cyber"


def test_multi_user_phishing_is_forced_to_p1():
    decision = manager.process(make_ai_result("cybersecurity", "medium", "multiple_users"))
    assert decision.priority_score == 45
    assert decision.priority == "P1"


def test_outlook_crash_is_p4():
    assert manager.process(make_ai_result("software", "medium", "individual")).priority == "P4"


def test_jammed_printer_is_p5():
    decision = manager.process(make_ai_result("hardware", "low", "individual"))
    assert decision.priority == "P5"
    assert decision.needs_review is False
    assert decision.review_reasons == ()


def test_low_confidence_is_flagged_for_review():
    decision = manager.process(make_ai_result(confidence=0.4))
    assert decision.needs_review is True
    assert len(decision.review_reasons) == 1


def test_both_review_reasons_are_kept():
    decision = manager.process(make_ai_result(confidence=0.4, flagged=True))
    assert len(decision.review_reasons) == 2


def test_bad_ai_result_raises_a_clear_error():
    assert raises(lm.InvalidAIResultError, manager.process,
                  make_ai_result(category="plumbing"))


def test_new_priority_rule_plugs_in_without_changing_logic_manager():
    class FinanceAlwaysP2(lm.PriorityRule):
        def adjust(self, priority, facts):
            return "P2" if facts.category == "finance" else priority

    custom = lm.LogicManager(priority_rules=[lm.CyberMultiUserRule(), FinanceAlwaysP2()])
    assert custom.process(make_ai_result("finance", "low", "individual")).priority == "P2"
    assert custom.process(make_ai_result("hardware", "low", "individual")).priority == "P5"


def test_no_review_rules_means_no_review():
    custom = lm.LogicManager(review_rules=[])
    assert custom.process(make_ai_result(confidence=0.1, flagged=True)).needs_review is False


def test_to_dict_has_the_phase_1_fields():
    saved = manager.process(make_ai_result("hr", "low", "individual", 0.8)).to_dict()
    phase_1_keys = {"priority", "priority_number", "priority_score", "department",
                    "needs_review", "category", "severity", "affected_scope",
                    "summary", "confidence_score"}
    assert phase_1_keys <= set(saved)
    assert saved["category"] == "hr"
    assert saved["affected_scope"] == "individual"
    assert saved["summary"] == "Test summary."
    assert saved["confidence_score"] == 0.8
    assert saved["review_reasons"] == []


def test_queue_highest_priority_first():
    tickets = [
        make_ticket("C", 3, 40, "2026-10-01T09:00"),
        make_ticket("A", 1, 85, "2026-10-01T10:00"),
        make_ticket("E", 5, 10, "2026-10-01T08:00"),
        make_ticket("B", 2, 60, "2026-10-01T11:00"),
        make_ticket("D", 4, 25, "2026-10-01T07:00"),
    ]
    assert [t["ticket_id"] for t in manager.sort_queue(tickets)] == ["A", "B", "C", "D", "E"]


def test_queue_higher_score_wins_a_tie():
    tickets = [make_ticket("low", 2, 50, "a"), make_ticket("high", 2, 65, "b")]
    assert [t["ticket_id"] for t in manager.sort_queue(tickets)] == ["high", "low"]


def test_queue_older_ticket_wins_a_full_tie():
    tickets = [make_ticket("new", 3, 40, "2026-10-02"), make_ticket("old", 3, 40, "2026-10-01")]
    assert [t["ticket_id"] for t in manager.sort_queue(tickets)] == ["old", "new"]


def test_new_ticket_slots_into_place():
    queue = [make_ticket("P1", 1, 85, "a"), make_ticket("P3", 3, 40, "b")]
    queue.append(make_ticket("new-P2", 2, 60, "c"))
    assert [t["ticket_id"] for t in manager.sort_queue(queue)] == ["P1", "new-P2", "P3"]


def test_queue_puts_a_ticket_with_missing_fields_last():
    tickets = [{"ticket_id": "broken"}, make_ticket("ok", 5, 10, "a")]
    assert [t["ticket_id"] for t in manager.sort_queue(tickets)] == ["ok", "broken"]


def test_queue_does_not_change_the_original_list():
    tickets = [make_ticket("B", 2, 60, "a"), make_ticket("A", 1, 85, "b")]
    manager.sort_queue(tickets)
    assert [t["ticket_id"] for t in tickets] == ["B", "A"]


def test_compat_layer_matches_the_classes():
    ai_result = make_ai_result("cybersecurity", "medium", "multiple_users")
    assert lm.process_ticket(ai_result) == manager.process(ai_result).to_dict()
    assert lm.DEPARTMENTS == ["Cyber", "HR", "Finance", "Infra"]


def decision_for(category, severity, scope="individual", confidence=0.9, flagged=False):
    return manager.process(make_ai_result(category, severity, scope, confidence, flagged)).to_dict()


def test_self_service_for_low_urgency_infra_tickets():
    p5 = decision_for("hardware", "low")        # score 10 -> P5
    p4 = decision_for("network", "medium")      # score 25 -> P4
    assert (p5["priority"], p5["department"]) == ("P5", "Infra")
    assert (p4["priority"], p4["department"]) == ("P4", "Infra")
    assert manager.offers_self_service(p5) and manager.offers_self_service(p4)
    assert lm.offers_self_service(p5)           # module-level function too


def test_no_self_service_for_urgent_infra_tickets():
    p3 = decision_for("software", "high")       # score 40 -> P3
    p1 = decision_for("network", "critical", "organisation")
    assert p3["priority"] == "P3" and p1["priority"] == "P1"
    assert not manager.offers_self_service(p3)
    assert not manager.offers_self_service(p1)


def test_no_self_service_for_other_departments():
    for category in ("hr", "finance", "cybersecurity"):
        low = decision_for(category, "low")
        assert low["priority"] == "P5" and low["department"] != "Infra", category
        assert not manager.offers_self_service(low), category


def test_no_self_service_when_review_is_needed():
    unsure = decision_for("hardware", "low", confidence=0.3)
    flagged = decision_for("account_access", "low", flagged=True)
    for decision in (unsure, flagged):
        assert decision["priority"] == "P5" and decision["department"] == "Infra"
        assert decision["needs_review"] and not manager.offers_self_service(decision)


def test_self_service_policy_can_be_configured():
    unsure = decision_for("hardware", "low", confidence=0.3)
    relaxed = lm.SelfServicePolicy([lm.DepartmentRule(("Infra",)),
                                    lm.PriorityLevelRule(("P5",))])     # no review rule
    assert relaxed.applies_to(unsure)
    assert not relaxed.applies_to(decision_for("network", "medium"))   # P4 not allowed here
    hr_only = lm.LogicManager(self_service=lm.SelfServicePolicy(
        [lm.DepartmentRule(("HR",)), lm.PriorityLevelRule(("P5",))]))
    assert not hr_only.offers_self_service(decision_for("hardware", "low"))
    assert hr_only.offers_self_service(decision_for("hr", "low"))


def test_self_service_rules_share_one_base_class():
    rules = lm.SelfServicePolicy.default_rules()
    assert all(isinstance(rule, lm.SelfServiceRule) for rule in rules)
    assert {type(rule) for rule in rules} == {lm.DepartmentRule, lm.PriorityLevelRule,
                                              lm.NoReviewNeededRule}
    assert all(rule.description for rule in rules)
    assert raises(TypeError, lm.SelfServiceRule, "abstract")


def test_a_new_self_service_rule_needs_no_change_to_the_policy():
    class OnlyHighConfidence(lm.SelfServiceRule):
        def __init__(self):
            super().__init__("AI is very sure")

        def _is_met(self, decision):
            return decision["confidence_score"] >= 0.95

    policy = lm.SelfServicePolicy(lm.SelfServicePolicy.default_rules() + [OnlyHighConfidence()])
    assert not policy.applies_to(decision_for("hardware", "low", confidence=0.9))
    assert policy.applies_to(decision_for("hardware", "low", confidence=0.97))


def run_all_tests():
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