"""Tests for Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
    Ticket,
)


class TestDepartmentEnum:
    def test_all_values(self):
        assert DepartmentEnum.BILLING == "Billing"
        assert DepartmentEnum.TECHNICAL == "Technical"
        assert DepartmentEnum.GENERAL == "General"
        assert DepartmentEnum.PREMIUM_SUPPORT == "Premium Support"

    def test_from_string(self):
        dept = DepartmentEnum("Billing")
        assert dept == DepartmentEnum.BILLING

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            DepartmentEnum("InvalidDept")


class TestPriorityEnum:
    def test_all_values(self):
        assert PriorityEnum.LOW == "Low"
        assert PriorityEnum.MEDIUM == "Medium"
        assert PriorityEnum.HIGH == "High"
        assert PriorityEnum.CRITICAL == "Critical"

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            PriorityEnum("Urgent")


class TestTicket:
    def test_valid_ticket(self):
        ticket = Ticket(
            ticket_id="TKT-001",
            subject="Test issue",
            description="A test description",
            sentiment=-0.5,
            customer_tier=CustomerTierEnum.PREMIUM,
            correct_department=DepartmentEnum.BILLING,
            correct_priority=PriorityEnum.HIGH,
            wait_time_seconds=120,
        )
        assert ticket.ticket_id == "TKT-001"
        assert ticket.sentiment == -0.5
        assert ticket.customer_tier == CustomerTierEnum.PREMIUM

    def test_sentiment_bounds(self):
        with pytest.raises(ValidationError):
            Ticket(
                ticket_id="TKT-001",
                subject="Test",
                description="Test",
                sentiment=2.0,  # invalid
                customer_tier=CustomerTierEnum.FREE,
                correct_department=DepartmentEnum.GENERAL,
                correct_priority=PriorityEnum.LOW,
                wait_time_seconds=0,
            )

    def test_to_dict(self):
        ticket = Ticket(
            ticket_id="TKT-001",
            subject="Test",
            description="Desc",
            sentiment=0.0,
            customer_tier=CustomerTierEnum.FREE,
            correct_department=DepartmentEnum.GENERAL,
            correct_priority=PriorityEnum.LOW,
            wait_time_seconds=0,
        )
        d = ticket.to_dict()
        assert "ticket_id" in d
        assert "subject" in d
        assert "description" in d
        assert "sentiment" in d
        assert "customer_tier" in d
        # Private fields should not be exposed
        assert "correct_department" not in d
        assert "correct_priority" not in d


class TestObservation:
    def _make_obs(self) -> Observation:
        return Observation(
            ticket_id="TKT-001",
            subject="Test subject",
            description="Test description",
            sentiment=-0.3,
            customer_tier="premium",
            wait_time_seconds=60,
            task_type="classification",
            step=1,
            max_steps=15,
            read_count=0,
            tickets_remaining=4,
            tickets_processed=1,
        )

    def test_creation(self):
        obs = self._make_obs()
        assert obs.ticket_id == "TKT-001"
        assert obs.step == 1
        assert obs.max_steps == 15

    def test_to_dict(self):
        obs = self._make_obs()
        d = obs.to_dict()
        assert isinstance(d, dict)
        assert d["ticket_id"] == "TKT-001"
        assert d["step"] == 1


class TestAction:
    def test_read_action(self):
        action = Action(action_type=ActionTypeEnum.READ)
        assert action.action_type == ActionTypeEnum.READ
        assert action.department is None
        assert action.priority is None
        assert action.confidence == 1.0

    def test_classify_action(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
            confidence=0.9,
        )
        assert action.department == DepartmentEnum.BILLING

    def test_action_from_string_department(self):
        action = Action(
            action_type="route",
            department="Technical",
            confidence=0.8,
        )
        assert action.department == DepartmentEnum.TECHNICAL
        assert action.action_type == ActionTypeEnum.ROUTE

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Action(action_type=ActionTypeEnum.READ, confidence=1.5)

    def test_to_dict(self):
        action = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.BILLING,
            confidence=0.9,
        )
        d = action.to_dict()
        assert d["action_type"] == "route"
        assert d["department"] == "Billing"
        assert d["confidence"] == 0.9


class TestReward:
    def test_creation(self):
        reward = Reward(value=0.5)
        assert reward.value == 0.5
        assert isinstance(reward.components, RewardComponents)

    def test_bounds(self):
        with pytest.raises(ValidationError):
            Reward(value=3.0)  # exceeds max

    def test_to_dict(self):
        reward = Reward(
            value=0.8,
            components=RewardComponents(correctness=0.8),
            message="test",
        )
        d = reward.to_dict()
        assert d["value"] == 0.8
        assert "components" in d
        assert d["message"] == "test"


class TestRewardComponents:
    def test_defaults(self):
        rc = RewardComponents()
        assert rc.correctness == 0.0
        assert rc.efficiency == 0.0
        assert rc.progress == 0.0
        assert rc.penalties == 0.0

    def test_to_dict(self):
        rc = RewardComponents(correctness=0.5, efficiency=-0.1)
        d = rc.to_dict()
        assert d["correctness"] == 0.5
        assert d["efficiency"] == -0.1
