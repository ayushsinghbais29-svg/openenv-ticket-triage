"""
TicketTriageEnv: OpenEnv-compliant customer support ticket triage environment.

Implements the OpenEnv interface:
  - reset() → initial observation
  - step(action) → observation, reward, done, truncated, info
  - state() → full environment state
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .graders import get_grader
from .models import (
    Action,
    ActionTypeEnum,
    EnvironmentState,
    Observation,
    Reward,
    TaskTypeEnum,
    Ticket,
)
from .reward_functions import RewardCalculator, RewardModulator
from .tasks import TicketGenerator
from .utils import TASK_CONFIG, validate_action


class TicketTriageEnv:
    """
    OpenEnv environment for customer support ticket triage.

    Supports three tasks of increasing difficulty:
      1. classification       (Easy)   - Route to correct department
      2. priority_classification (Medium) - Classify + set priority
      3. efficiency_triage    (Hard)   - Route 10 tickets efficiently

    Usage:
        env = TicketTriageEnv(task_type="classification", seed=42)
        obs = env.reset()
        action = {"action_type": "classify", "department": "Billing", "confidence": 0.9}
        obs, reward, done, truncated, info = env.step(action)
        state = env.state()
    """

    def __init__(
        self,
        task_type: str = "classification",
        seed: Optional[int] = None,
    ):
        """
        Initialize the environment.

        Args:
            task_type: One of "classification", "priority_classification",
                       "efficiency_triage"
            seed: Optional random seed for reproducibility
        """
        from .utils import get_task_type

        self.task_type = get_task_type(task_type)
        self.seed = seed
        self._config = TASK_CONFIG[self.task_type]
        self._generator = TicketGenerator(seed=seed)
        self._grader = get_grader(task_type)
        self._reward_calc = RewardCalculator(self.task_type)
        self._modulator = RewardModulator()

        self._tickets: List[Ticket] = []
        self._current_idx: int = 0
        self._step: int = 0
        self._done: bool = False
        self._truncated: bool = False
        self._total_reward: float = 0.0
        self._action_history: List[Dict[str, Any]] = []
        self._read_counts: Dict[str, int] = {}
        self._episode_actions: List[Action] = []
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # OpenEnv Interface
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment and return the initial observation.

        Returns:
            dict: Initial observation for the first ticket.
        """
        n_tickets = self._config["n_tickets"]
        self._generator = TicketGenerator(seed=self.seed)
        self._tickets = self._generator.generate_episode(n_tickets)
        self._current_idx = 0
        self._step = 0
        self._done = False
        self._truncated = False
        self._total_reward = 0.0
        self._action_history = []
        self._read_counts = {}
        self._episode_actions = []
        self._initialized = True

        return self._make_observation().to_dict()

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], bool, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.

        Args:
            action: Dict with keys:
                - action_type: str (read, analyze, classify, set_priority, route)
                - department: Optional[str] for classify/route actions
                - priority: Optional[str] for set_priority actions
                - confidence: float 0.0–1.0 (default 1.0)
                - reasoning: Optional[str]

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before step()")

        if self._done or self._truncated:
            obs = self._make_observation()
            return (
                obs.to_dict(),
                Reward(value=0.0, message="episode already ended").to_dict(),
                self._done,
                self._truncated,
                self._make_info(),
            )

        errors = validate_action(action, self.task_type)
        if errors:
            obs = self._make_observation()
            reward = Reward(value=-0.1, message=f"invalid action: {errors}")
            return obs.to_dict(), reward.to_dict(), False, False, self._make_info()

        parsed_action = Action(**action)
        self._step += 1
        self._episode_actions.append(parsed_action)
        self._action_history.append(
            {**parsed_action.to_dict(), "step": self._step, "ticket_id": self._current_ticket.ticket_id}
        )

        if parsed_action.action_type == ActionTypeEnum.READ:
            tid = self._current_ticket.ticket_id
            self._read_counts[tid] = self._read_counts.get(tid, 0) + 1

        truncated = self._step >= self._config["max_steps"]
        done = False

        if parsed_action.action_type in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE):
            self._current_idx += 1
            if self._current_idx >= len(self._tickets):
                done = True

        self._done = done
        self._truncated = truncated and not done

        reward = self._reward_calc.calculate(
            action=parsed_action,
            ticket=self._current_ticket if not done else self._tickets[-1],
            step=self._step,
            max_steps=self._config["max_steps"],
            read_counts=self._read_counts,
            action_history=self._episode_actions,
            done=done,
            truncated=self._truncated,
        )
        reward = self._modulator.modulate(reward, self.task_type)
        self._total_reward += reward.value

        obs = self._make_observation()
        info = self._make_info()
        if done or self._truncated:
            info["episode_score"] = self._grader.grade_episode(
                self._episode_actions,
                self._tickets,
                self._step,
                self._config["max_steps"],
            )

        return obs.to_dict(), reward.to_dict(), done, self._truncated, info

    def state(self) -> Dict[str, Any]:
        """
        Return complete environment state.

        Returns:
            dict: Full EnvironmentState representation.
        """
        env_state = EnvironmentState(
            task_type=self.task_type,
            step=self._step,
            max_steps=self._config["max_steps"],
            tickets=[t.to_dict() for t in self._tickets],
            current_ticket_index=self._current_idx,
            action_history=deepcopy(self._action_history),
            total_reward=self._total_reward,
            done=self._done,
            truncated=self._truncated,
            episode_score=(
                self._grader.grade_episode(
                    self._episode_actions,
                    self._tickets,
                    self._step,
                    self._config["max_steps"],
                )
                if self._initialized
                else 0.0
            ),
            read_counts=deepcopy(self._read_counts),
        )
        return env_state.to_dict()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _current_ticket(self) -> Ticket:
        idx = min(self._current_idx, len(self._tickets) - 1)
        return self._tickets[idx]

    def _make_observation(self) -> Observation:
        ticket = self._current_ticket
        available = [a.value for a in self._config["available_actions"]]
        tickets_remaining = max(0, len(self._tickets) - self._current_idx)

        return Observation(
            ticket_id=ticket.ticket_id,
            subject=ticket.subject,
            description=ticket.description,
            sentiment=ticket.sentiment,
            customer_tier=ticket.customer_tier.value,
            wait_time_seconds=ticket.wait_time_seconds,
            task_type=self.task_type.value,
            step=self._step,
            max_steps=self._config["max_steps"],
            read_count=self._read_counts.get(ticket.ticket_id, 0),
            tickets_remaining=tickets_remaining,
            tickets_processed=self._current_idx,
            current_score=round(self._total_reward, 4),
            available_actions=available,
            message="" if not self._done else "Episode complete",
        )

    def _make_info(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "max_steps": self._config["max_steps"],
            "total_reward": round(self._total_reward, 4),
            "current_ticket_index": self._current_idx,
            "tickets_total": len(self._tickets),
            "task_type": self.task_type.value,
        }
