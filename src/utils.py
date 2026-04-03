"""Utility functions for validation, seeding, and helpers."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .models import ActionTypeEnum, DepartmentEnum, PriorityEnum, TaskTypeEnum


TASK_CONFIG = {
    TaskTypeEnum.CLASSIFICATION: {
        "n_tickets": 5,
        "max_steps": 15,
        "available_actions": [
            ActionTypeEnum.READ,
            ActionTypeEnum.ANALYZE,
            ActionTypeEnum.CLASSIFY,
            ActionTypeEnum.ROUTE,
        ],
        "description": "Route tickets to the correct department",
    },
    TaskTypeEnum.PRIORITY_CLASSIFICATION: {
        "n_tickets": 5,
        "max_steps": 20,
        "available_actions": [
            ActionTypeEnum.READ,
            ActionTypeEnum.ANALYZE,
            ActionTypeEnum.CLASSIFY,
            ActionTypeEnum.SET_PRIORITY,
            ActionTypeEnum.ROUTE,
        ],
        "description": "Classify department AND set priority for each ticket",
    },
    TaskTypeEnum.EFFICIENCY_TRIAGE: {
        "n_tickets": 10,
        "max_steps": 30,
        "available_actions": [
            ActionTypeEnum.READ,
            ActionTypeEnum.ANALYZE,
            ActionTypeEnum.CLASSIFY,
            ActionTypeEnum.SET_PRIORITY,
            ActionTypeEnum.ROUTE,
        ],
        "description": "Route 10 tickets efficiently, balancing quality vs. speed",
    },
}


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)


def validate_action(action_dict: Dict[str, Any], task_type: TaskTypeEnum) -> Dict[str, str]:
    """
    Validate an action dict, returning a dict of field -> error messages.
    Empty dict means valid.
    """
    errors: Dict[str, str] = {}
    config = TASK_CONFIG.get(task_type, {})
    allowed = config.get("available_actions", [])

    action_type_str = action_dict.get("action_type")
    if action_type_str is None:
        errors["action_type"] = "action_type is required"
        return errors

    try:
        action_type = ActionTypeEnum(action_type_str)
    except ValueError:
        errors["action_type"] = (
            f"Invalid action_type '{action_type_str}'. "
            f"Must be one of {[a.value for a in ActionTypeEnum]}"
        )
        return errors

    if action_type not in allowed:
        errors["action_type"] = (
            f"Action '{action_type_str}' not allowed for task '{task_type.value}'. "
            f"Allowed: {[a.value for a in allowed]}"
        )

    if action_type in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
        dept_str = action_dict.get("department")
        if dept_str is not None:
            try:
                DepartmentEnum(dept_str)
            except ValueError:
                errors["department"] = (
                    f"Invalid department '{dept_str}'. "
                    f"Must be one of {[d.value for d in DepartmentEnum]}"
                )

    if action_type == ActionTypeEnum.SET_PRIORITY:
        priority_str = action_dict.get("priority")
        if priority_str is not None:
            try:
                PriorityEnum(priority_str)
            except ValueError:
                errors["priority"] = (
                    f"Invalid priority '{priority_str}'. "
                    f"Must be one of {[p.value for p in PriorityEnum]}"
                )

    confidence = action_dict.get("confidence", 1.0)
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        errors["confidence"] = f"confidence must be between 0.0 and 1.0, got {confidence}"

    return errors


def compute_episode_score(
    grader: Any,
    actions: List[Any],
    tickets: List[Any],
    step_count: int,
    max_steps: int,
) -> float:
    """Compute final episode score using the appropriate grader."""
    return grader.grade_episode(actions, tickets, step_count, max_steps)


def format_observation_text(obs: Dict[str, Any]) -> str:
    """Format an observation as readable text for the agent."""
    lines = [
        f"Ticket ID: {obs.get('ticket_id', 'N/A')}",
        f"Subject: {obs.get('subject', 'N/A')}",
        f"Description: {obs.get('description', 'N/A')}",
        f"Customer Tier: {obs.get('customer_tier', 'N/A')}",
        f"Sentiment: {obs.get('sentiment', 0.0):.2f}",
        f"Wait Time: {obs.get('wait_time_seconds', 0)}s",
        f"Task Type: {obs.get('task_type', 'N/A')}",
        f"Step: {obs.get('step', 0)}/{obs.get('max_steps', 0)}",
        f"Tickets Remaining: {obs.get('tickets_remaining', 0)}",
        f"Available Actions: {', '.join(obs.get('available_actions', []))}",
    ]
    return "\n".join(lines)


def get_task_type(task_type_str: str) -> TaskTypeEnum:
    """Parse task type string to enum."""
    mapping = {
        "classification": TaskTypeEnum.CLASSIFICATION,
        "priority_classification": TaskTypeEnum.PRIORITY_CLASSIFICATION,
        "efficiency_triage": TaskTypeEnum.EFFICIENCY_TRIAGE,
    }
    if task_type_str not in mapping:
        raise ValueError(
            f"Unknown task type: '{task_type_str}'. Must be one of {list(mapping)}"
        )
    return mapping[task_type_str]
