"""
Deterministic graders for OpenEnv Ticket Triage tasks.
Each grader returns a score in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .models import (
    Action,
    ActionTypeEnum,
    CustomerTierEnum,
    DepartmentEnum,
    Observation,
    PriorityEnum,
    TaskTypeEnum,
)


# ---------------------------------------------------------------------------
# Tier weights for priority classification scoring
# ---------------------------------------------------------------------------

TIER_WEIGHTS: Dict[CustomerTierEnum, float] = {
    CustomerTierEnum.FREE: 1.0,
    CustomerTierEnum.PREMIUM: 1.2,
    CustomerTierEnum.ENTERPRISE: 1.5,
}

PRIORITY_SCORES: Dict[PriorityEnum, int] = {
    PriorityEnum.LOW: 0,
    PriorityEnum.MEDIUM: 1,
    PriorityEnum.HIGH: 2,
    PriorityEnum.CRITICAL: 3,
}


def _priority_distance(predicted: PriorityEnum, actual: PriorityEnum) -> int:
    """Return ordinal distance between two priority levels."""
    return abs(PRIORITY_SCORES[predicted] - PRIORITY_SCORES[actual])


# ---------------------------------------------------------------------------
# ClassificationGrader
# ---------------------------------------------------------------------------


class ClassificationGrader:
    """
    Grades department classification correctness.
    Returns 1.0 for exact match, 0.0 otherwise.
    Modulated slightly by agent confidence.
    """

    def grade(
        self,
        action: Action,
        observation: Observation,
        ticket_metadata: Dict[str, Any],
    ) -> float:
        """
        Grade a classification action.

        Args:
            action: Agent action containing department prediction.
            observation: Current ticket observation.
            ticket_metadata: Ground-truth metadata from ticket generation.

        Returns:
            Score in [0.0, 1.0].
        """
        if action.action_type not in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
            return 0.0

        predicted_dept = action.department
        if predicted_dept is None:
            return 0.0

        correct_dept = DepartmentEnum(ticket_metadata.get("template_department", "general"))

        if predicted_dept == correct_dept:
            # Slight modulation by confidence (0.95 base, up to 1.0)
            confidence = getattr(action, "confidence", 1.0)
            return min(1.0, 0.95 + 0.05 * confidence)
        else:
            return 0.0


# ---------------------------------------------------------------------------
# PriorityClassificationGrader
# ---------------------------------------------------------------------------


class PriorityClassificationGrader:
    """
    Grades priority + department classification.
    Uses F1-style scoring with customer tier weighting.
    Department wrong = 0.0; priority graded by ordinal distance.
    """

    def grade(
        self,
        action: Action,
        observation: Observation,
        ticket_metadata: Dict[str, Any],
    ) -> float:
        """
        Grade a priority classification action.

        Args:
            action: Agent action with department and priority predictions.
            observation: Current ticket observation.
            ticket_metadata: Ground-truth metadata from ticket generation.

        Returns:
            Score in [0.0, 1.0].
        """
        if action.action_type not in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
            return 0.0

        predicted_dept = action.department
        predicted_priority = action.priority

        if predicted_dept is None or predicted_priority is None:
            return 0.0

        correct_dept = DepartmentEnum(ticket_metadata.get("template_department", "general"))
        correct_priority = PriorityEnum(ticket_metadata.get("template_priority", "medium"))

        # Department must be correct for any credit
        dept_correct = predicted_dept == correct_dept
        dept_score = 1.0 if dept_correct else 0.0

        # Priority scored by distance: 0 distance = 1.0, max distance (3) = 0.0
        dist = _priority_distance(predicted_priority, correct_priority)
        priority_score = max(0.0, 1.0 - (dist / 3.0))

        # Combined score: 0.5 department + 0.5 priority
        combined = 0.5 * dept_score + 0.5 * priority_score

        # Apply tier weight modulation
        tier = CustomerTierEnum(observation.customer_tier)
        tier_weight = TIER_WEIGHTS.get(tier, 1.0)
        # Normalize so tier_weight doesn't exceed 1.0
        modulated = combined * (1.0 if tier_weight == 1.0 else min(1.0, combined * (tier_weight / 1.0)))

        # Ensure score remains in [0, 1]
        return min(1.0, max(0.0, modulated))


# ---------------------------------------------------------------------------
# EfficiencyGrader
# ---------------------------------------------------------------------------


class EfficiencyGrader:
    """
    Grades triage efficiency: quality × speed bonus.
    Penalizes excessive wait times; rewards fast accurate routing.
    """

    FAST_THRESHOLD_MINUTES = 30.0
    SLOW_THRESHOLD_MINUTES = 120.0
    ESCALATION_BONUS_TIERS: Dict[PriorityEnum, float] = {
        PriorityEnum.LOW: 1.0,
        PriorityEnum.MEDIUM: 1.0,
        PriorityEnum.HIGH: 1.1,
        PriorityEnum.CRITICAL: 1.3,
    }

    def grade(
        self,
        action: Action,
        observation: Observation,
        ticket_metadata: Dict[str, Any],
        steps_used: int = 1,
        max_steps: int = 40,
    ) -> float:
        """
        Grade efficiency of triage action.

        Args:
            action: Agent action.
            observation: Current observation with wait time.
            ticket_metadata: Ground-truth metadata.
            steps_used: Number of steps used for this ticket.
            max_steps: Maximum allowed steps in episode.

        Returns:
            Score in [0.0, 1.0].
        """
        if action.action_type not in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE, ActionTypeEnum.ESCALATE):
            return 0.0

        predicted_dept = action.department
        if predicted_dept is None:
            return 0.0

        correct_dept = DepartmentEnum(ticket_metadata.get("template_department", "general"))
        correct_priority = PriorityEnum(ticket_metadata.get("template_priority", "medium"))

        # Base quality score (department correctness)
        quality = 1.0 if predicted_dept == correct_dept else 0.0

        # Speed bonus/penalty based on wait time
        wait = observation.wait_time_minutes
        if wait <= self.FAST_THRESHOLD_MINUTES:
            speed_factor = 1.1  # bonus for fast routing
        elif wait >= self.SLOW_THRESHOLD_MINUTES:
            speed_factor = 0.7  # penalty for very slow routing
        else:
            # Linear interpolation between thresholds
            ratio = (wait - self.FAST_THRESHOLD_MINUTES) / (
                self.SLOW_THRESHOLD_MINUTES - self.FAST_THRESHOLD_MINUTES
            )
            speed_factor = 1.1 - (0.4 * ratio)

        # Escalation bonus for critical/enterprise tickets
        escalation_bonus = self.ESCALATION_BONUS_TIERS.get(correct_priority, 1.0)

        # Step efficiency: bonus for fewer steps used
        step_efficiency = max(0.5, 1.0 - (steps_used / max_steps) * 0.5)

        score = quality * speed_factor * escalation_bonus * step_efficiency
        return min(1.0, max(0.0, score))
