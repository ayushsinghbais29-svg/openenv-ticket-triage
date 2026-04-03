"""
Typed Pydantic models for OpenEnv Ticket Triage environment.
Includes Observation, Action, Reward, RewardComponents, EnvironmentState, and Enums.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DepartmentEnum(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    PREMIUM_SUPPORT = "premium_support"


class PriorityEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionTypeEnum(str, Enum):
    READ = "read"
    ANALYZE = "analyze"
    CLASSIFY = "classify"
    ROUTE = "route"
    ESCALATE = "escalate"
    CLOSE = "close"


class TaskTypeEnum(str, Enum):
    CLASSIFICATION = "classification"
    PRIORITY_CLASSIFICATION = "priority_classification"
    EFFICIENCY = "efficiency"


class CustomerTierEnum(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


# ---------------------------------------------------------------------------
# Observation Model
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """Observation returned by the environment after each step."""

    model_config = ConfigDict(use_enum_values=True)

    ticket_id: str = Field(..., description="Unique identifier for the ticket")
    subject: str = Field(..., description="Ticket subject / title")
    description: str = Field(..., description="Full ticket description body")
    customer_tier: CustomerTierEnum = Field(..., description="Customer subscription tier")
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Sentiment score from -1.0 (negative) to 1.0 (positive)"
    )
    wait_time_minutes: float = Field(
        default=0.0, ge=0.0, description="Time ticket has been waiting (minutes)"
    )
    task_type: TaskTypeEnum = Field(..., description="Current task type")
    step_number: int = Field(default=0, ge=0, description="Current step number in episode")
    episode_number: int = Field(default=0, ge=0, description="Current episode number")
    tickets_remaining: int = Field(default=0, ge=0, description="Number of tickets remaining in episode")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """Action taken by the agent in the environment."""

    model_config = ConfigDict(use_enum_values=True)

    action_type: ActionTypeEnum = Field(..., description="Type of action to perform")
    department: Optional[DepartmentEnum] = Field(
        None, description="Target department for classify/route actions"
    )
    priority: Optional[PriorityEnum] = Field(
        None, description="Priority level for priority_classification task"
    )
    reasoning: Optional[str] = Field(None, description="Agent's chain-of-thought reasoning")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Agent's confidence in this action"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Reward Models
# ---------------------------------------------------------------------------


class RewardComponents(BaseModel):
    """Breakdown of reward into interpretable components."""

    correctness: float = Field(default=0.0, description="Score for correct classification")
    efficiency: float = Field(default=0.0, description="Score for time efficiency")
    progress: float = Field(default=0.0, description="Score for intermediate progress steps")
    penalties: float = Field(default=0.0, description="Accumulated penalties (negative)")

    @property
    def total(self) -> float:
        return self.correctness + self.efficiency + self.progress + self.penalties

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["total"] = self.total
        return d


class Reward(BaseModel):
    """Reward signal returned by the environment."""

    value: float = Field(..., description="Scalar reward value")
    components: RewardComponents = Field(
        default_factory=RewardComponents, description="Breakdown of reward components"
    )
    grader_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Raw grader score (0.0-1.0)"
    )
    done: bool = Field(default=False, description="Whether episode is complete")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional information")

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["components"] = self.components.to_dict()
        return d


# ---------------------------------------------------------------------------
# Environment State Model
# ---------------------------------------------------------------------------


class EnvironmentState(BaseModel):
    """Full state of the environment at any point in time."""

    model_config = ConfigDict(use_enum_values=True)

    task_type: TaskTypeEnum = Field(..., description="Active task type")
    episode_number: int = Field(default=0, ge=0, description="Current episode index")
    step_number: int = Field(default=0, ge=0, description="Current step within episode")
    max_steps: int = Field(default=15, ge=1, description="Maximum steps allowed per episode")
    tickets_total: int = Field(default=5, ge=1, description="Total tickets in this episode")
    tickets_processed: int = Field(default=0, ge=0, description="Tickets processed so far")
    current_ticket_index: int = Field(default=0, ge=0, description="Index of current ticket")
    cumulative_reward: float = Field(default=0.0, description="Cumulative reward this episode")
    episode_done: bool = Field(default=False, description="Whether episode has ended")
    actions_taken: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of actions in this episode"
    )
    grader_scores: List[float] = Field(
        default_factory=list, description="Grader scores per ticket"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional state metadata")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
