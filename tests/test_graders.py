"""
Unit tests for graders - determinism and score range tests.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graders import ClassificationGrader, EfficiencyGrader, PriorityClassificationGrader
from src.models import (
    Action,
    ActionTypeEnum,
    CustomerTierEnum,
    DepartmentEnum,
    Observation,
    PriorityEnum,
    TaskTypeEnum,
)


def make_obs(
    department=None,
    tier=CustomerTierEnum.FREE,
    wait_time=10.0,
    sentiment=0.0,
) -> Observation:
    return Observation(
        ticket_id="TKT-001",
        subject="Test ticket",
        description="Test description",
        customer_tier=tier,
        sentiment_score=sentiment,
        wait_time_minutes=wait_time,
        task_type=TaskTypeEnum.CLASSIFICATION,
    )


def make_metadata(dept=DepartmentEnum.BILLING, priority=PriorityEnum.MEDIUM) -> dict:
    return {
        "template_department": dept.value,
        "template_priority": priority.value,
    }


class TestClassificationGrader:
    def setup_method(self):
        self.grader = ClassificationGrader()

    def test_correct_classification_high_score(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
        )
        obs = make_obs()
        meta = make_metadata(dept=DepartmentEnum.BILLING)
        score = self.grader.grade(action, obs, meta)
        assert score > 0.9

    def test_wrong_classification_zero_score(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.TECHNICAL,
        )
        obs = make_obs()
        meta = make_metadata(dept=DepartmentEnum.BILLING)
        score = self.grader.grade(action, obs, meta)
        assert score == 0.0

    def test_non_terminal_action_zero_score(self):
        action = Action(action_type=ActionTypeEnum.READ)
        obs = make_obs()
        meta = make_metadata()
        score = self.grader.grade(action, obs, meta)
        assert score == 0.0

    def test_score_range(self):
        for dept in DepartmentEnum:
            action = Action(
                action_type=ActionTypeEnum.CLASSIFY,
                department=dept,
            )
            obs = make_obs()
            meta = make_metadata(dept=dept)
            score = self.grader.grade(action, obs, meta)
            assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        """Same inputs always produce same score."""
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
        )
        obs = make_obs()
        meta = make_metadata(dept=DepartmentEnum.BILLING)
        scores = [self.grader.grade(action, obs, meta) for _ in range(5)]
        assert all(s == scores[0] for s in scores)

    def test_route_action_also_graded(self):
        action = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.GENERAL,
        )
        obs = make_obs()
        meta = make_metadata(dept=DepartmentEnum.GENERAL)
        score = self.grader.grade(action, obs, meta)
        assert score > 0.9

    def test_no_department_zero_score(self):
        action = Action(action_type=ActionTypeEnum.CLASSIFY, department=None)
        obs = make_obs()
        meta = make_metadata()
        score = self.grader.grade(action, obs, meta)
        assert score == 0.0


class TestPriorityClassificationGrader:
    def setup_method(self):
        self.grader = PriorityClassificationGrader()

    def test_correct_both_high_score(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.TECHNICAL,
            priority=PriorityEnum.HIGH,
        )
        obs = make_obs()
        meta = make_metadata(dept=DepartmentEnum.TECHNICAL, priority=PriorityEnum.HIGH)
        score = self.grader.grade(action, obs, meta)
        assert score > 0.7

    def test_wrong_department_zero_score(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
            priority=PriorityEnum.HIGH,
        )
        obs = make_obs()
        meta = make_metadata(dept=DepartmentEnum.TECHNICAL, priority=PriorityEnum.HIGH)
        score = self.grader.grade(action, obs, meta)
        # Department wrong: department component = 0, but priority may still give partial
        assert score <= 0.5  # Should be significantly penalized

    def test_wrong_priority_partial_score(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.TECHNICAL,
            priority=PriorityEnum.LOW,  # Off by 2 levels from HIGH
        )
        obs = make_obs()
        meta = make_metadata(dept=DepartmentEnum.TECHNICAL, priority=PriorityEnum.HIGH)
        score = self.grader.grade(action, obs, meta)
        # Partial credit for correct dept, penalized for wrong priority
        assert 0.0 < score < 0.7

    def test_score_range(self):
        for dept in DepartmentEnum:
            for priority in PriorityEnum:
                action = Action(
                    action_type=ActionTypeEnum.CLASSIFY,
                    department=dept,
                    priority=priority,
                )
                obs = make_obs()
                meta = make_metadata(dept=DepartmentEnum.BILLING, priority=PriorityEnum.MEDIUM)
                score = self.grader.grade(action, obs, meta)
                assert 0.0 <= score <= 1.0

    def test_missing_priority_zero_score(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
            priority=None,
        )
        obs = make_obs()
        meta = make_metadata()
        score = self.grader.grade(action, obs, meta)
        assert score == 0.0

    def test_deterministic(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
            priority=PriorityEnum.HIGH,
        )
        obs = make_obs()
        meta = make_metadata(dept=DepartmentEnum.BILLING, priority=PriorityEnum.HIGH)
        scores = [self.grader.grade(action, obs, meta) for _ in range(5)]
        assert all(s == scores[0] for s in scores)


class TestEfficiencyGrader:
    def setup_method(self):
        self.grader = EfficiencyGrader()

    def test_fast_correct_high_score(self):
        action = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.BILLING,
        )
        obs = make_obs(wait_time=10.0)  # Fast
        meta = make_metadata(dept=DepartmentEnum.BILLING, priority=PriorityEnum.MEDIUM)
        score = self.grader.grade(action, obs, meta, steps_used=2, max_steps=40)
        assert score > 0.7

    def test_slow_correct_lower_score(self):
        action = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.BILLING,
        )
        fast_obs = make_obs(wait_time=10.0)
        slow_obs = make_obs(wait_time=150.0)
        meta = make_metadata(dept=DepartmentEnum.BILLING)

        fast_score = self.grader.grade(action, fast_obs, meta, steps_used=2, max_steps=40)
        slow_score = self.grader.grade(action, slow_obs, meta, steps_used=2, max_steps=40)
        assert fast_score > slow_score

    def test_wrong_department_zero_score(self):
        action = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.TECHNICAL,
        )
        obs = make_obs(wait_time=5.0)
        meta = make_metadata(dept=DepartmentEnum.BILLING)
        score = self.grader.grade(action, obs, meta, steps_used=1, max_steps=40)
        assert score == 0.0

    def test_score_range(self):
        action = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.BILLING,
        )
        for wait in [5.0, 30.0, 60.0, 120.0, 200.0]:
            obs = make_obs(wait_time=wait)
            meta = make_metadata(dept=DepartmentEnum.BILLING)
            score = self.grader.grade(action, obs, meta, steps_used=5, max_steps=40)
            assert 0.0 <= score <= 1.0

    def test_escalate_action_graded(self):
        action = Action(
            action_type=ActionTypeEnum.ESCALATE,
            department=DepartmentEnum.PREMIUM_SUPPORT,
        )
        obs = make_obs(wait_time=90.0)
        meta = make_metadata(dept=DepartmentEnum.PREMIUM_SUPPORT, priority=PriorityEnum.CRITICAL)
        score = self.grader.grade(action, obs, meta, steps_used=1, max_steps=40)
        assert score > 0.0

    def test_varying_scores_different_inputs(self):
        """Grader should produce varying (not constant) scores."""
        meta_billing = make_metadata(dept=DepartmentEnum.BILLING)
        meta_technical = make_metadata(dept=DepartmentEnum.TECHNICAL)

        action_billing = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.BILLING,
        )
        action_technical = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.TECHNICAL,
        )

        obs = make_obs(wait_time=15.0)

        score_correct = self.grader.grade(action_billing, obs, meta_billing, steps_used=2, max_steps=40)
        score_wrong = self.grader.grade(action_billing, obs, meta_technical, steps_used=2, max_steps=40)

        # Scores should differ
        assert score_correct != score_wrong
