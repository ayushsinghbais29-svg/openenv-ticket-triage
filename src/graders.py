"""Graders for the three task difficulty levels."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import (
    Action,
    ActionTypeEnum,
    DepartmentEnum,
    PriorityEnum,
    Ticket,
)


class ClassificationGrader:
    """
    Easy Task Grader: Department Classification.

    Scores based on accuracy of routing to the correct department.
    Confidence modulates the score for partial credit.
    """

    TARGET_BASELINE = 0.85

    def grade_action(
        self,
        action: Action,
        ticket: Ticket,
    ) -> float:
        """Grade a single classify/route action. Returns score 0.0–1.0."""
        if action.action_type not in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
            return 0.0

        if action.department is None:
            return 0.0

        correct = action.department == ticket.correct_department
        if correct:
            return min(1.0, 0.5 + action.confidence * 0.5)
        return 0.0

    def grade_episode(
        self,
        actions: List[Action],
        tickets: List[Ticket],
        step_count: int,
        max_steps: int,
    ) -> float:
        """Grade a full episode. Returns aggregate score 0.0–1.0."""
        if not tickets:
            return 0.0

        scores = []
        for ticket in tickets:
            ticket_score = 0.0
            for action in actions:
                if action.action_type in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
                    grade = self.grade_action(action, ticket)
                    if grade > ticket_score:
                        ticket_score = grade
            scores.append(ticket_score)

        return sum(scores) / len(scores)


class PriorityClassificationGrader:
    """
    Medium Task Grader: Priority + Classification.

    Uses F1-like scoring that evaluates both department and priority correctness.
    Enterprise/premium tiers get extra weight since mis-routing them is more costly.
    """

    TARGET_BASELINE = 0.75

    TIER_WEIGHTS = {
        "enterprise": 1.5,
        "premium": 1.2,
        "free": 1.0,
    }

    PRIORITY_ADJACENCY = {
        PriorityEnum.LOW: {PriorityEnum.LOW: 1.0, PriorityEnum.MEDIUM: 0.5},
        PriorityEnum.MEDIUM: {
            PriorityEnum.MEDIUM: 1.0,
            PriorityEnum.LOW: 0.5,
            PriorityEnum.HIGH: 0.5,
        },
        PriorityEnum.HIGH: {
            PriorityEnum.HIGH: 1.0,
            PriorityEnum.MEDIUM: 0.5,
            PriorityEnum.CRITICAL: 0.5,
        },
        PriorityEnum.CRITICAL: {PriorityEnum.CRITICAL: 1.0, PriorityEnum.HIGH: 0.5},
    }

    def grade_action(
        self,
        classify_action: Optional[Action],
        priority_action: Optional[Action],
        ticket: Ticket,
    ) -> float:
        """Grade a combined classification + priority decision."""
        dept_score = 0.0
        if classify_action is not None and classify_action.department is not None:
            if classify_action.department == ticket.correct_department:
                dept_score = min(1.0, 0.5 + classify_action.confidence * 0.5)

        priority_score = 0.0
        if priority_action is not None and priority_action.priority is not None:
            adjacency = self.PRIORITY_ADJACENCY.get(ticket.correct_priority, {})
            priority_score = adjacency.get(priority_action.priority, 0.0)

        if dept_score == 0.0 and priority_score == 0.0:
            return 0.0
        if dept_score == 0.0:
            return priority_score * 0.3
        if priority_score == 0.0:
            return dept_score * 0.7

        precision = (dept_score + priority_score) / 2
        recall = (dept_score + priority_score) / 2
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def grade_episode(
        self,
        actions: List[Action],
        tickets: List[Ticket],
        step_count: int,
        max_steps: int,
    ) -> float:
        """Grade a full episode with weighted F1 scoring.

        Actions are matched to tickets positionally: the Nth classify/route
        action corresponds to the Nth ticket; the Nth set_priority action
        corresponds to the Nth ticket.
        """
        if not tickets:
            return 0.0

        classify_actions: List[Optional[Action]] = [None] * len(tickets)
        priority_actions: List[Optional[Action]] = [None] * len(tickets)

        classify_idx = 0
        priority_idx = 0
        for action in actions:
            if action.action_type in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
                if classify_idx < len(tickets):
                    classify_actions[classify_idx] = action
                    classify_idx += 1
            elif action.action_type == ActionTypeEnum.SET_PRIORITY:
                if priority_idx < len(tickets):
                    priority_actions[priority_idx] = action
                    priority_idx += 1

        total_weight = 0.0
        weighted_score = 0.0

        for i, ticket in enumerate(tickets):
            weight = self.TIER_WEIGHTS.get(ticket.customer_tier.value, 1.0)
            score = self.grade_action(
                classify_actions[i],
                priority_actions[i],
                ticket,
            )
            weighted_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return weighted_score / total_weight


class EfficiencyGrader:
    """
    Hard Task Grader: Efficiency Triage.

    Composite score balancing quality of routing decisions against
    time/step efficiency. Penalizes over-deliberation and rewards
    fast correct routing, especially for enterprise tickets.
    """

    TARGET_BASELINE = 0.65

    ESCALATION_BONUS = 0.15

    def _compute_quality_score(
        self,
        actions: List[Action],
        tickets: List[Ticket],
    ) -> float:
        """Compute quality of routing decisions."""
        if not tickets:
            return 0.0

        correct = 0
        for ticket in tickets:
            for action in actions:
                if action.action_type in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
                    if action.department == ticket.correct_department:
                        correct += 1
                        break

        return correct / len(tickets)

    def _compute_efficiency_score(
        self, step_count: int, max_steps: int, n_tickets: int
    ) -> float:
        """Compute efficiency: steps used relative to optimal."""
        optimal_steps = n_tickets * 2
        if step_count <= optimal_steps:
            return 1.0
        ratio = optimal_steps / max(step_count, 1)
        return max(0.0, ratio)

    def _compute_escalation_score(
        self,
        actions: List[Action],
        tickets: List[Ticket],
    ) -> float:
        """Bonus for correctly handling critical/enterprise tickets."""
        critical_tickets = [
            t
            for t in tickets
            if t.correct_priority == PriorityEnum.CRITICAL
            or t.customer_tier.value == "enterprise"
        ]
        if not critical_tickets:
            return 0.0

        correct = 0
        for ticket in critical_tickets:
            for action in actions:
                if action.action_type in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
                    if action.department == ticket.correct_department:
                        correct += 1
                        break

        return correct / len(critical_tickets)

    def grade_episode(
        self,
        actions: List[Action],
        tickets: List[Ticket],
        step_count: int,
        max_steps: int,
    ) -> float:
        """Grade a full efficiency episode. Returns composite score 0.0–1.0."""
        if not tickets:
            return 0.0

        quality = self._compute_quality_score(actions, tickets)
        efficiency = self._compute_efficiency_score(step_count, max_steps, len(tickets))
        escalation = self._compute_escalation_score(actions, tickets)

        composite = 0.5 * quality + 0.35 * efficiency + 0.15 * escalation
        return min(1.0, composite)


def get_grader(task_type: str) -> Any:
    """Factory to get grader by task type string."""
    mapping = {
        "classification": ClassificationGrader(),
        "priority_classification": PriorityClassificationGrader(),
        "efficiency_triage": EfficiencyGrader(),
    }
    if task_type not in mapping:
        raise ValueError(f"Unknown task type: {task_type}. Must be one of {list(mapping)}")
    return mapping[task_type]
