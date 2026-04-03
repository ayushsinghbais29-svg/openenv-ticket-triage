"""
Validation and helper utilities for the OpenEnv Ticket Triage environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .models import (
    Action,
    ActionTypeEnum,
    DepartmentEnum,
    Observation,
    PriorityEnum,
    TaskTypeEnum,
)


def validate_action(action: Union[Action, Dict[str, Any]]) -> Action:
    """
    Validate and coerce an action to the Action model.

    Args:
        action: Action instance or dict.

    Returns:
        Validated Action instance.

    Raises:
        ValueError: If action is invalid.
    """
    if isinstance(action, dict):
        action = Action(**action)
    if not isinstance(action, Action):
        raise TypeError(f"Expected Action or dict, got {type(action)}")
    return action


def validate_observation(obs: Union[Observation, Dict[str, Any]]) -> Observation:
    """
    Validate and coerce an observation to the Observation model.

    Args:
        obs: Observation instance or dict.

    Returns:
        Validated Observation instance.
    """
    if isinstance(obs, dict):
        obs = Observation(**obs)
    if not isinstance(obs, Observation):
        raise TypeError(f"Expected Observation or dict, got {type(obs)}")
    return obs


def action_requires_department(action: Action, task_type: TaskTypeEnum) -> bool:
    """
    Check whether the given action requires a department to be specified.

    Args:
        action: The action.
        task_type: Current task type.

    Returns:
        True if department is required.
    """
    terminal_types = {ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE, ActionTypeEnum.ESCALATE}
    return ActionTypeEnum(action.action_type) in terminal_types


def action_requires_priority(action: Action, task_type: TaskTypeEnum) -> bool:
    """
    Check whether the given action requires a priority to be specified.

    Args:
        action: The action.
        task_type: Current task type.

    Returns:
        True if priority is required for this task type.
    """
    if task_type != TaskTypeEnum.PRIORITY_CLASSIFICATION:
        return False
    terminal_types = {ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE}
    return ActionTypeEnum(action.action_type) in terminal_types


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length characters, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_ticket_for_prompt(observation: Observation) -> str:
    """
    Format a ticket observation as a human-readable string for LLM prompts.

    Args:
        observation: The ticket observation.

    Returns:
        Formatted string representation.
    """
    lines = [
        f"Ticket ID: {observation.ticket_id}",
        f"Subject: {observation.subject}",
        f"Description: {truncate_text(observation.description, 400)}",
        f"Customer Tier: {observation.customer_tier}",
        f"Sentiment Score: {observation.sentiment_score:.2f}",
        f"Wait Time: {observation.wait_time_minutes:.1f} minutes",
    ]
    return "\n".join(lines)


def compute_episode_stats(grader_scores: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for an episode's grader scores.

    Args:
        grader_scores: List of grader scores from the episode.

    Returns:
        Dict with mean, min, max, and count.
    """
    if not grader_scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": round(sum(grader_scores) / len(grader_scores), 4),
        "min": round(min(grader_scores), 4),
        "max": round(max(grader_scores), 4),
        "count": len(grader_scores),
    }


def department_label(dept: Optional[Union[DepartmentEnum, str]]) -> str:
    """Convert a department enum to a human-readable label."""
    if dept is None:
        return "Unknown"
    labels = {
        DepartmentEnum.BILLING: "Billing",
        DepartmentEnum.TECHNICAL: "Technical Support",
        DepartmentEnum.GENERAL: "General Support",
        DepartmentEnum.PREMIUM_SUPPORT: "Premium Support",
        "billing": "Billing",
        "technical": "Technical Support",
        "general": "General Support",
        "premium_support": "Premium Support",
    }
    return labels.get(dept, str(dept).replace("_", " ").title())


def priority_label(priority: Optional[Union[PriorityEnum, str]]) -> str:
    """Convert a priority enum to a human-readable label with emoji."""
    if priority is None:
        return "Unknown"
    labels = {
        PriorityEnum.LOW: "🟢 Low",
        PriorityEnum.MEDIUM: "🟡 Medium",
        PriorityEnum.HIGH: "🟠 High",
        PriorityEnum.CRITICAL: "🔴 Critical",
        "low": "🟢 Low",
        "medium": "🟡 Medium",
        "high": "🟠 High",
        "critical": "🔴 Critical",
    }
    return labels.get(priority, str(priority).title())
