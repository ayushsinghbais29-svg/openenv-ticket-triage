"""Dense multi-component reward functions for the ticket triage environment."""

from __future__ import annotations

from typing import Dict, List

from .models import (
    Action,
    ActionTypeEnum,
    Reward,
    RewardComponents,
    Ticket,
    TaskTypeEnum,
)

DENSE_REWARDS = {
    ActionTypeEnum.READ: 0.1,
    ActionTypeEnum.ANALYZE: 0.2,
    ActionTypeEnum.CLASSIFY: 0.0,
    ActionTypeEnum.SET_PRIORITY: 0.0,
    ActionTypeEnum.ROUTE: 0.0,
}

CORRECTNESS_BONUS = {
    "department_correct": 0.8,
    "department_correct_with_priority": 1.0,
    "department_wrong": 0.0,
    "priority_correct": 0.4,
    "priority_adjacent": 0.2,
    "priority_wrong": 0.0,
}

EFFICIENCY_PENALTY_RATE = 0.05
REPETITION_THRESHOLD = 2
REPETITION_PENALTY = 0.2
TIMEOUT_PENALTY = 1.0
EFFICIENCY_THRESHOLD_RATIO = 0.6


class RewardCalculator:
    """
    Calculates multi-component rewards for each step.

    Components:
    - progress: Dense signal for exploratory actions (read, analyze)
    - correctness: Bonus for correct routing/classification decisions
    - efficiency: Penalty for excessive steps relative to max
    - penalties: Repetition, timeout, and other negative signals
    """

    def __init__(self, task_type: TaskTypeEnum):
        self.task_type = task_type

    def calculate(
        self,
        action: Action,
        ticket: Ticket,
        step: int,
        max_steps: int,
        read_counts: Dict[str, int],
        action_history: List[Action],
        done: bool,
        truncated: bool,
    ) -> Reward:
        """Calculate reward for a single step."""
        components = RewardComponents()
        messages = []

        dense = DENSE_REWARDS.get(action.action_type, 0.0)
        if dense > 0:
            components.progress = dense
            messages.append(f"+{dense:.1f} for {action.action_type.value}")

        ticket_read_count = read_counts.get(ticket.ticket_id, 0)
        if action.action_type == ActionTypeEnum.READ and ticket_read_count > REPETITION_THRESHOLD:
            penalty = REPETITION_PENALTY
            components.penalties -= penalty
            messages.append(f"-{penalty:.1f} repetition penalty (read {ticket_read_count}x)")

        if action.action_type in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
            components.correctness = self._grade_classification(action, ticket)
            if components.correctness > 0:
                messages.append(f"+{components.correctness:.2f} correctness")

        if action.action_type == ActionTypeEnum.SET_PRIORITY:
            components.correctness = self._grade_priority(action, ticket)
            if components.correctness > 0:
                messages.append(f"+{components.correctness:.2f} priority correctness")

        step_ratio = step / max(max_steps, 1)
        if step_ratio > EFFICIENCY_THRESHOLD_RATIO and action.action_type not in (
            ActionTypeEnum.CLASSIFY,
            ActionTypeEnum.ROUTE,
        ):
            penalty = EFFICIENCY_PENALTY_RATE * step_ratio
            components.efficiency = -round(penalty, 3)

        if truncated:
            components.penalties -= TIMEOUT_PENALTY
            messages.append(f"-{TIMEOUT_PENALTY:.1f} timeout penalty")

        total = (
            components.progress
            + components.correctness
            + components.efficiency
            + components.penalties
        )
        total = max(-2.0, min(2.0, total))

        return Reward(
            value=round(total, 4),
            components=components,
            message="; ".join(messages) if messages else "no reward",
        )

    def _grade_classification(self, action: Action, ticket: Ticket) -> float:
        """Score department classification correctness."""
        if action.department is None:
            return 0.0
        if action.department == ticket.correct_department:
            base = CORRECTNESS_BONUS["department_correct"]
            return round(base * min(1.0, action.confidence), 4)
        return CORRECTNESS_BONUS["department_wrong"]

    def _grade_priority(self, action: Action, ticket: Ticket) -> float:
        """Score priority assignment correctness."""
        if action.priority is None:
            return 0.0

        from .models import PriorityEnum

        PRIORITY_ORDER = [
            PriorityEnum.LOW,
            PriorityEnum.MEDIUM,
            PriorityEnum.HIGH,
            PriorityEnum.CRITICAL,
        ]

        correct_idx = PRIORITY_ORDER.index(ticket.correct_priority)
        actual_idx = PRIORITY_ORDER.index(action.priority)
        diff = abs(correct_idx - actual_idx)

        if diff == 0:
            return CORRECTNESS_BONUS["priority_correct"]
        elif diff == 1:
            return CORRECTNESS_BONUS["priority_adjacent"]
        return CORRECTNESS_BONUS["priority_wrong"]


class RewardModulator:
    """Scales rewards based on task difficulty to maintain balanced learning signal."""

    TASK_SCALES = {
        TaskTypeEnum.CLASSIFICATION: 1.0,
        TaskTypeEnum.PRIORITY_CLASSIFICATION: 0.9,
        TaskTypeEnum.EFFICIENCY_TRIAGE: 0.8,
    }

    def modulate(self, reward: Reward, task_type: TaskTypeEnum) -> Reward:
        """Apply task-difficulty scaling to reward."""
        scale = self.TASK_SCALES.get(task_type, 1.0)
        new_value = round(reward.value * scale, 4)
        new_value = max(-2.0, min(2.0, new_value))
        return Reward(
            value=new_value,
            components=reward.components,
            message=reward.message,
        )
