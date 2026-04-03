"""Pydantic models for the OpenEnv Ticket Triage environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DepartmentEnum(str, Enum):
    BILLING = "Billing"
    TECHNICAL = "Technical"
    GENERAL = "General"
    PREMIUM_SUPPORT = "Premium Support"


class PriorityEnum(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ActionTypeEnum(str, Enum):
    READ = "read"
    ANALYZE = "analyze"
    CLASSIFY = "classify"
    SET_PRIORITY = "set_priority"
    ROUTE = "route"


class TaskTypeEnum(str, Enum):
    CLASSIFICATION = "classification"
    PRIORITY_CLASSIFICATION = "priority_classification"
    EFFICIENCY_TRIAGE = "efficiency_triage"


class CustomerTierEnum(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class SentimentEnum(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    description: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    customer_tier: CustomerTierEnum
    correct_department: DepartmentEnum
    correct_priority: PriorityEnum
    wait_time_seconds: int = Field(ge=0)
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "subject": self.subject,
            "description": self.description,
            "sentiment": self.sentiment,
            "customer_tier": self.customer_tier.value,
            "wait_time_seconds": self.wait_time_seconds,
        }


class Observation(BaseModel):
    ticket_id: str
    subject: str
    description: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    customer_tier: str
    wait_time_seconds: int
    task_type: str
    step: int
    max_steps: int
    read_count: int
    tickets_remaining: int
    tickets_processed: int
    current_score: float = 0.0
    available_actions: List[str] = Field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class Action(BaseModel):
    action_type: ActionTypeEnum
    department: Optional[DepartmentEnum] = None
    priority: Optional[PriorityEnum] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: Optional[str] = None

    @field_validator("department", mode="before")
    @classmethod
    def validate_department(cls, v: Any) -> Optional[DepartmentEnum]:
        if v is None:
            return None
        if isinstance(v, DepartmentEnum):
            return v
        return DepartmentEnum(v)

    @field_validator("priority", mode="before")
    @classmethod
    def validate_priority(cls, v: Any) -> Optional[PriorityEnum]:
        if v is None:
            return None
        if isinstance(v, PriorityEnum):
            return v
        return PriorityEnum(v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "department": self.department.value if self.department else None,
            "priority": self.priority.value if self.priority else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class RewardComponents(BaseModel):
    correctness: float = 0.0
    efficiency: float = 0.0
    progress: float = 0.0
    penalties: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


class Reward(BaseModel):
    value: float = Field(ge=-2.0, le=2.0)
    components: RewardComponents = Field(default_factory=RewardComponents)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "components": self.components.to_dict(),
            "message": self.message,
        }


class EnvironmentState(BaseModel):
    task_type: TaskTypeEnum
    step: int
    max_steps: int
    tickets: List[Dict[str, Any]]
    current_ticket_index: int
    action_history: List[Dict[str, Any]]
    total_reward: float
    done: bool
    truncated: bool
    episode_score: float
    read_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
