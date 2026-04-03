"""
Unit tests for Pydantic model validation.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    Action,
    ActionTypeEnum,
    CustomerTierEnum,
    DepartmentEnum,
    EnvironmentState,
    Observation,
    PriorityEnum,
    Reward,
    RewardComponents,
    TaskTypeEnum,
)


class TestObservation:
    def test_valid_observation(self):
        obs = Observation(
            ticket_id="TKT-001",
            subject="Test subject",
            description="Test description",
            customer_tier=CustomerTierEnum.FREE,
            sentiment_score=0.5,
            task_type=TaskTypeEnum.CLASSIFICATION,
        )
        assert obs.ticket_id == "TKT-001"
        assert obs.sentiment_score == 0.5

    def test_sentiment_bounds(self):
        with pytest.raises(Exception):
            Observation(
                ticket_id="TKT-001",
                subject="Test",
                description="Desc",
                customer_tier=CustomerTierEnum.FREE,
                sentiment_score=1.5,  # Out of bounds
                task_type=TaskTypeEnum.CLASSIFICATION,
            )

    def test_to_dict(self):
        obs = Observation(
            ticket_id="TKT-001",
            subject="Test",
            description="Desc",
            customer_tier=CustomerTierEnum.PREMIUM,
            sentiment_score=-0.3,
            task_type=TaskTypeEnum.PRIORITY_CLASSIFICATION,
        )
        d = obs.to_dict()
        assert isinstance(d, dict)
        assert d["ticket_id"] == "TKT-001"
        assert d["sentiment_score"] == -0.3

    def test_default_metadata(self):
        obs = Observation(
            ticket_id="TKT-001",
            subject="Test",
            description="Desc",
            customer_tier=CustomerTierEnum.FREE,
            sentiment_score=0.0,
            task_type=TaskTypeEnum.EFFICIENCY,
        )
        assert obs.metadata == {}
        assert obs.wait_time_minutes == 0.0


class TestAction:
    def test_classify_action(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
        )
        assert action.action_type == ActionTypeEnum.CLASSIFY
        assert action.department == DepartmentEnum.BILLING
        assert action.priority is None

    def test_priority_classification_action(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.TECHNICAL,
            priority=PriorityEnum.CRITICAL,
            confidence=0.95,
        )
        assert action.priority == PriorityEnum.CRITICAL
        assert action.confidence == 0.95

    def test_read_action(self):
        action = Action(action_type=ActionTypeEnum.READ)
        assert action.department is None
        assert action.priority is None

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            Action(
                action_type=ActionTypeEnum.CLASSIFY,
                confidence=1.5,  # Out of bounds
            )

    def test_to_dict(self):
        action = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.GENERAL,
        )
        d = action.to_dict()
        assert isinstance(d, dict)
        assert d["action_type"] == ActionTypeEnum.ROUTE


class TestRewardComponents:
    def test_total_computation(self):
        rc = RewardComponents(correctness=0.8, efficiency=0.1, progress=0.05, penalties=-0.1)
        assert abs(rc.total - 0.85) < 1e-9

    def test_to_dict_includes_total(self):
        rc = RewardComponents(correctness=0.5)
        d = rc.to_dict()
        assert "total" in d
        assert d["total"] == 0.5

    def test_defaults_zero(self):
        rc = RewardComponents()
        assert rc.total == 0.0


class TestReward:
    def test_basic_reward(self):
        r = Reward(value=0.75, grader_score=0.8, done=True)
        assert r.value == 0.75
        assert r.done is True

    def test_reward_to_dict(self):
        r = Reward(value=1.0, done=False)
        d = r.to_dict()
        assert "components" in d
        assert "total" in d["components"]


class TestEnvironmentState:
    def test_initial_state(self):
        state = EnvironmentState(task_type=TaskTypeEnum.CLASSIFICATION)
        assert state.episode_number == 0
        assert state.step_number == 0
        assert state.episode_done is False
        assert state.actions_taken == []

    def test_to_dict(self):
        state = EnvironmentState(
            task_type=TaskTypeEnum.EFFICIENCY,
            max_steps=40,
            tickets_total=10,
        )
        d = state.to_dict()
        assert d["max_steps"] == 40
        assert d["tickets_total"] == 10


class TestEnums:
    def test_department_enum_values(self):
        assert DepartmentEnum.BILLING == "billing"
        assert DepartmentEnum.TECHNICAL == "technical"
        assert DepartmentEnum.GENERAL == "general"
        assert DepartmentEnum.PREMIUM_SUPPORT == "premium_support"

    def test_priority_enum_values(self):
        assert PriorityEnum.LOW == "low"
        assert PriorityEnum.CRITICAL == "critical"

    def test_task_type_enum_values(self):
        assert TaskTypeEnum.CLASSIFICATION == "classification"
        assert TaskTypeEnum.EFFICIENCY == "efficiency"

    def test_customer_tier_enum(self):
        assert CustomerTierEnum.FREE == "free"
        assert CustomerTierEnum.ENTERPRISE == "enterprise"
