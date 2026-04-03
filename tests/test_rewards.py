"""Tests for reward functions."""

from __future__ import annotations

import pytest

from src.models import (
    Action,
    ActionTypeEnum,
    CustomerTierEnum,
    DepartmentEnum,
    PriorityEnum,
    Reward,
    TaskTypeEnum,
    Ticket,
)
from src.reward_functions import RewardCalculator, RewardModulator


def make_ticket(
    dept: DepartmentEnum = DepartmentEnum.BILLING,
    priority: PriorityEnum = PriorityEnum.HIGH,
) -> Ticket:
    return Ticket(
        ticket_id="TKT-001",
        subject="Test",
        description="Test description",
        sentiment=-0.3,
        customer_tier=CustomerTierEnum.PREMIUM,
        correct_department=dept,
        correct_priority=priority,
        wait_time_seconds=60,
    )


class TestRewardCalculator:
    def _calc(self, task_type: TaskTypeEnum = TaskTypeEnum.CLASSIFICATION) -> RewardCalculator:
        return RewardCalculator(task_type)

    def test_read_gives_positive_reward(self):
        calc = self._calc()
        action = Action(action_type=ActionTypeEnum.READ)
        ticket = make_ticket()
        reward = calc.calculate(
            action=action,
            ticket=ticket,
            step=1,
            max_steps=15,
            read_counts={},
            action_history=[],
            done=False,
            truncated=False,
        )
        assert reward.value > 0.0
        assert reward.components.progress > 0.0

    def test_analyze_gives_positive_reward(self):
        calc = self._calc()
        action = Action(action_type=ActionTypeEnum.ANALYZE)
        ticket = make_ticket()
        reward = calc.calculate(
            action=action,
            ticket=ticket,
            step=1,
            max_steps=15,
            read_counts={},
            action_history=[],
            done=False,
            truncated=False,
        )
        assert reward.value > 0.0
        assert reward.components.progress > 0.0

    def test_analyze_gives_more_than_read(self):
        calc = self._calc()
        ticket = make_ticket()
        base_kwargs = dict(
            ticket=ticket, step=1, max_steps=15, read_counts={},
            action_history=[], done=False, truncated=False
        )
        read_reward = calc.calculate(action=Action(action_type=ActionTypeEnum.READ), **base_kwargs)
        analyze_reward = calc.calculate(action=Action(action_type=ActionTypeEnum.ANALYZE), **base_kwargs)
        assert analyze_reward.value > read_reward.value

    def test_correct_routing_gives_correctness_reward(self):
        calc = self._calc()
        ticket = make_ticket(dept=DepartmentEnum.BILLING)
        action = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.BILLING,
            confidence=0.9,
        )
        reward = calc.calculate(
            action=action,
            ticket=ticket,
            step=5,
            max_steps=15,
            read_counts={},
            action_history=[],
            done=False,
            truncated=False,
        )
        assert reward.components.correctness > 0.0

    def test_wrong_routing_gives_zero_correctness(self):
        calc = self._calc()
        ticket = make_ticket(dept=DepartmentEnum.BILLING)
        action = Action(
            action_type=ActionTypeEnum.ROUTE,
            department=DepartmentEnum.TECHNICAL,
            confidence=0.9,
        )
        reward = calc.calculate(
            action=action,
            ticket=ticket,
            step=5,
            max_steps=15,
            read_counts={},
            action_history=[],
            done=False,
            truncated=False,
        )
        assert reward.components.correctness == 0.0

    def test_repetition_penalty(self):
        calc = self._calc()
        ticket = make_ticket()
        action = Action(action_type=ActionTypeEnum.READ)
        reward = calc.calculate(
            action=action,
            ticket=ticket,
            step=5,
            max_steps=15,
            read_counts={"TKT-001": 5},  # already read 5 times
            action_history=[],
            done=False,
            truncated=False,
        )
        assert reward.components.penalties < 0.0

    def test_timeout_penalty(self):
        calc = self._calc()
        ticket = make_ticket()
        action = Action(action_type=ActionTypeEnum.READ)
        reward = calc.calculate(
            action=action,
            ticket=ticket,
            step=15,
            max_steps=15,
            read_counts={},
            action_history=[],
            done=False,
            truncated=True,  # truncated
        )
        assert reward.components.penalties <= -1.0

    def test_reward_value_in_range(self):
        calc = self._calc()
        ticket = make_ticket()
        for action_type in ActionTypeEnum:
            action = Action(
                action_type=action_type,
                department=DepartmentEnum.BILLING if action_type in (
                    ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE
                ) else None,
                priority=PriorityEnum.HIGH if action_type == ActionTypeEnum.SET_PRIORITY else None,
            )
            reward = calc.calculate(
                action=action,
                ticket=ticket,
                step=5,
                max_steps=15,
                read_counts={},
                action_history=[],
                done=False,
                truncated=False,
            )
            assert -2.0 <= reward.value <= 2.0, f"Out of range for {action_type}: {reward.value}"

    def test_components_structure(self):
        calc = self._calc()
        ticket = make_ticket()
        action = Action(action_type=ActionTypeEnum.READ)
        reward = calc.calculate(
            action=action,
            ticket=ticket,
            step=1,
            max_steps=15,
            read_counts={},
            action_history=[],
            done=False,
            truncated=False,
        )
        components = reward.components
        assert hasattr(components, "correctness")
        assert hasattr(components, "efficiency")
        assert hasattr(components, "progress")
        assert hasattr(components, "penalties")

    def test_priority_correct_reward(self):
        calc = RewardCalculator(TaskTypeEnum.PRIORITY_CLASSIFICATION)
        ticket = make_ticket(priority=PriorityEnum.HIGH)
        action = Action(
            action_type=ActionTypeEnum.SET_PRIORITY,
            priority=PriorityEnum.HIGH,
            confidence=0.9,
        )
        reward = calc.calculate(
            action=action,
            ticket=ticket,
            step=3,
            max_steps=20,
            read_counts={},
            action_history=[],
            done=False,
            truncated=False,
        )
        assert reward.components.correctness > 0.0

    def test_priority_wrong_reward(self):
        calc = RewardCalculator(TaskTypeEnum.PRIORITY_CLASSIFICATION)
        ticket = make_ticket(priority=PriorityEnum.CRITICAL)
        action = Action(
            action_type=ActionTypeEnum.SET_PRIORITY,
            priority=PriorityEnum.LOW,
            confidence=0.9,
        )
        reward = calc.calculate(
            action=action,
            ticket=ticket,
            step=3,
            max_steps=20,
            read_counts={},
            action_history=[],
            done=False,
            truncated=False,
        )
        assert reward.components.correctness == 0.0


class TestRewardModulator:
    def test_scales_by_task(self):
        modulator = RewardModulator()
        reward = Reward(value=1.0)
        for task_type in TaskTypeEnum:
            modulated = modulator.modulate(reward, task_type)
            assert -2.0 <= modulated.value <= 2.0

    def test_classification_scale_is_1(self):
        modulator = RewardModulator()
        reward = Reward(value=0.8)
        modulated = modulator.modulate(reward, TaskTypeEnum.CLASSIFICATION)
        assert modulated.value == pytest.approx(0.8, abs=1e-4)

    def test_harder_tasks_have_lower_scale(self):
        modulator = RewardModulator()
        reward = Reward(value=1.0)
        easy = modulator.modulate(reward, TaskTypeEnum.CLASSIFICATION)
        medium = modulator.modulate(reward, TaskTypeEnum.PRIORITY_CLASSIFICATION)
        hard = modulator.modulate(reward, TaskTypeEnum.EFFICIENCY_TRIAGE)
        assert easy.value >= medium.value >= hard.value
