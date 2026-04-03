"""Tests for task graders."""

from __future__ import annotations

import pytest

from src.graders import (
    ClassificationGrader,
    EfficiencyGrader,
    PriorityClassificationGrader,
    get_grader,
)
from src.models import (
    Action,
    ActionTypeEnum,
    CustomerTierEnum,
    DepartmentEnum,
    PriorityEnum,
    Ticket,
)
from src.tasks import TicketGenerator


def make_ticket(
    dept: DepartmentEnum = DepartmentEnum.BILLING,
    priority: PriorityEnum = PriorityEnum.HIGH,
    tier: CustomerTierEnum = CustomerTierEnum.PREMIUM,
    ticket_id: str = "TKT-001",
) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        subject="Test ticket",
        description="Test description",
        sentiment=-0.5,
        customer_tier=tier,
        correct_department=dept,
        correct_priority=priority,
        wait_time_seconds=60,
    )


def make_route_action(dept: DepartmentEnum, confidence: float = 0.9) -> Action:
    return Action(
        action_type=ActionTypeEnum.ROUTE,
        department=dept,
        confidence=confidence,
    )


def make_priority_action(priority: PriorityEnum, confidence: float = 0.9) -> Action:
    return Action(
        action_type=ActionTypeEnum.SET_PRIORITY,
        priority=priority,
        confidence=confidence,
    )


class TestClassificationGrader:
    def test_correct_routing(self):
        grader = ClassificationGrader()
        ticket = make_ticket(dept=DepartmentEnum.BILLING)
        action = make_route_action(DepartmentEnum.BILLING)
        score = grader.grade_action(action, ticket)
        assert score > 0.5
        assert score <= 1.0

    def test_wrong_routing(self):
        grader = ClassificationGrader()
        ticket = make_ticket(dept=DepartmentEnum.BILLING)
        action = make_route_action(DepartmentEnum.TECHNICAL)
        score = grader.grade_action(action, ticket)
        assert score == 0.0

    def test_score_in_range(self):
        grader = ClassificationGrader()
        ticket = make_ticket(dept=DepartmentEnum.GENERAL)
        for dept in DepartmentEnum:
            action = make_route_action(dept)
            score = grader.grade_action(action, ticket)
            assert 0.0 <= score <= 1.0

    def test_high_confidence_boosts_score(self):
        grader = ClassificationGrader()
        ticket = make_ticket(dept=DepartmentEnum.BILLING)
        low_conf = grader.grade_action(make_route_action(DepartmentEnum.BILLING, 0.5), ticket)
        high_conf = grader.grade_action(make_route_action(DepartmentEnum.BILLING, 1.0), ticket)
        assert high_conf >= low_conf

    def test_non_routing_action_scores_zero(self):
        grader = ClassificationGrader()
        ticket = make_ticket()
        action = Action(action_type=ActionTypeEnum.READ)
        assert grader.grade_action(action, ticket) == 0.0

    def test_episode_grading(self):
        grader = ClassificationGrader()
        gen = TicketGenerator(seed=42)
        tickets = gen.generate_episode(3)
        actions = [make_route_action(t.correct_department) for t in tickets]
        score = grader.grade_episode(actions, tickets, 9, 15)
        assert 0.0 <= score <= 1.0

    def test_determinism(self):
        grader = ClassificationGrader()
        gen = TicketGenerator(seed=99)
        tickets = gen.generate_episode(5)
        actions = [make_route_action(t.correct_department) for t in tickets]
        score1 = grader.grade_episode(actions, tickets, 10, 15)
        score2 = grader.grade_episode(actions, tickets, 10, 15)
        assert score1 == score2

    def test_score_not_constant(self):
        grader = ClassificationGrader()
        gen = TicketGenerator(seed=1)
        tickets = gen.generate_episode(5)
        all_correct = [make_route_action(t.correct_department) for t in tickets]
        all_wrong = [make_route_action(DepartmentEnum.GENERAL) for _ in tickets]
        correct_score = grader.grade_episode(all_correct, tickets, 5, 15)
        wrong_score = grader.grade_episode(all_wrong, tickets, 5, 15)
        assert correct_score != wrong_score


class TestPriorityClassificationGrader:
    def test_correct_both(self):
        grader = PriorityClassificationGrader()
        ticket = make_ticket(dept=DepartmentEnum.BILLING, priority=PriorityEnum.HIGH)
        classify = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
            confidence=0.9,
        )
        priority = make_priority_action(PriorityEnum.HIGH)
        score = grader.grade_action(classify, priority, ticket)
        assert score > 0.5

    def test_wrong_department_reduces_score(self):
        grader = PriorityClassificationGrader()
        ticket = make_ticket(dept=DepartmentEnum.BILLING, priority=PriorityEnum.HIGH)
        classify = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.TECHNICAL,
            confidence=0.9,
        )
        priority = make_priority_action(PriorityEnum.HIGH)
        score = grader.grade_action(classify, priority, ticket)
        assert score < 0.5

    def test_adjacent_priority_partial_credit(self):
        grader = PriorityClassificationGrader()
        ticket = make_ticket(dept=DepartmentEnum.BILLING, priority=PriorityEnum.HIGH)
        classify = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
            confidence=0.9,
        )
        adjacent_priority = make_priority_action(PriorityEnum.MEDIUM)
        wrong_score = grader.grade_action(classify, adjacent_priority, ticket)
        correct_priority = make_priority_action(PriorityEnum.HIGH)
        correct_score = grader.grade_action(classify, correct_priority, ticket)
        assert 0.0 <= wrong_score <= correct_score

    def test_score_in_range(self):
        grader = PriorityClassificationGrader()
        gen = TicketGenerator(seed=42)
        tickets = gen.generate_episode(5)
        actions = [make_route_action(t.correct_department) for t in tickets]
        priority_actions = [make_priority_action(t.correct_priority) for t in tickets]
        all_actions = actions + priority_actions
        score = grader.grade_episode(all_actions, tickets, 10, 20)
        assert 0.0 <= score <= 1.0

    def test_determinism(self):
        grader = PriorityClassificationGrader()
        gen = TicketGenerator(seed=7)
        tickets = gen.generate_episode(5)
        actions = [make_route_action(t.correct_department) for t in tickets]
        s1 = grader.grade_episode(actions, tickets, 5, 20)
        s2 = grader.grade_episode(actions, tickets, 5, 20)
        assert s1 == s2

    def test_score_not_constant(self):
        grader = PriorityClassificationGrader()
        gen = TicketGenerator(seed=2)
        tickets = gen.generate_episode(5)
        correct_actions = [make_route_action(t.correct_department) for t in tickets]
        wrong_actions = [make_route_action(DepartmentEnum.GENERAL) for _ in tickets]
        correct_score = grader.grade_episode(correct_actions, tickets, 5, 20)
        wrong_score = grader.grade_episode(wrong_actions, tickets, 5, 20)
        assert correct_score != wrong_score


class TestEfficiencyGrader:
    def test_perfect_episode(self):
        grader = EfficiencyGrader()
        gen = TicketGenerator(seed=42)
        tickets = gen.generate_episode(10)
        actions = [make_route_action(t.correct_department) for t in tickets]
        score = grader.grade_episode(actions, tickets, 10, 30)
        assert score > 0.5

    def test_wrong_routing_reduces_score(self):
        grader = EfficiencyGrader()
        gen = TicketGenerator(seed=42)
        tickets = gen.generate_episode(10)
        all_wrong = [make_route_action(DepartmentEnum.GENERAL) for _ in tickets]
        score = grader.grade_episode(all_wrong, tickets, 10, 30)
        assert 0.0 <= score <= 1.0

    def test_excessive_steps_reduces_score(self):
        grader = EfficiencyGrader()
        gen = TicketGenerator(seed=42)
        tickets = gen.generate_episode(10)
        actions = [make_route_action(t.correct_department) for t in tickets]
        fast_score = grader.grade_episode(actions, tickets, 10, 30)
        slow_score = grader.grade_episode(actions, tickets, 29, 30)
        assert fast_score >= slow_score

    def test_score_in_range(self):
        grader = EfficiencyGrader()
        gen = TicketGenerator(seed=5)
        tickets = gen.generate_episode(10)
        actions = [make_route_action(t.correct_department) for t in tickets]
        score = grader.grade_episode(actions, tickets, 15, 30)
        assert 0.0 <= score <= 1.0

    def test_determinism(self):
        grader = EfficiencyGrader()
        gen = TicketGenerator(seed=42)
        tickets = gen.generate_episode(10)
        actions = [make_route_action(t.correct_department) for t in tickets]
        s1 = grader.grade_episode(actions, tickets, 12, 30)
        s2 = grader.grade_episode(actions, tickets, 12, 30)
        assert s1 == s2

    def test_score_not_constant(self):
        grader = EfficiencyGrader()
        gen = TicketGenerator(seed=3)
        tickets = gen.generate_episode(10)
        correct_actions = [make_route_action(t.correct_department) for t in tickets]
        wrong_actions = [make_route_action(DepartmentEnum.GENERAL) for _ in tickets]
        c_score = grader.grade_episode(correct_actions, tickets, 10, 30)
        w_score = grader.grade_episode(wrong_actions, tickets, 10, 30)
        assert c_score != w_score


class TestGetGrader:
    def test_valid_task_types(self):
        for task_type in ["classification", "priority_classification", "efficiency_triage"]:
            grader = get_grader(task_type)
            assert grader is not None

    def test_invalid_task_type(self):
        with pytest.raises(ValueError):
            get_grader("nonexistent_task")

    def test_classification_grader_type(self):
        grader = get_grader("classification")
        assert isinstance(grader, ClassificationGrader)

    def test_priority_grader_type(self):
        grader = get_grader("priority_classification")
        assert isinstance(grader, PriorityClassificationGrader)

    def test_efficiency_grader_type(self):
        grader = get_grader("efficiency_triage")
        assert isinstance(grader, EfficiencyGrader)
