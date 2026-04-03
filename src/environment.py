"""
TicketTriageEnv: Gymnasium-compatible OpenEnv environment for customer support ticket triage.
Supports three task types: classification, priority_classification, and efficiency.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

from .graders import ClassificationGrader, EfficiencyGrader, PriorityClassificationGrader
from .models import (
    Action,
    ActionTypeEnum,
    EnvironmentState,
    Observation,
    Reward,
    TaskTypeEnum,
)
from .reward_functions import RewardCalculator, RewardModulator
from .tasks import TicketGenerator

# Task configuration
TASK_CONFIG: Dict[TaskTypeEnum, Dict[str, Any]] = {
    TaskTypeEnum.CLASSIFICATION: {
        "tickets_per_episode": 5,
        "max_steps": 15,
        "description": "Route tickets to the correct department",
    },
    TaskTypeEnum.PRIORITY_CLASSIFICATION: {
        "tickets_per_episode": 5,
        "max_steps": 20,
        "description": "Classify tickets by department AND priority level",
    },
    TaskTypeEnum.EFFICIENCY: {
        "tickets_per_episode": 10,
        "max_steps": 40,
        "description": "Route 10-ticket stream optimizing quality vs. latency",
    },
}


class TicketTriageEnv:
    """
    OpenEnv-compliant environment for customer support ticket triage.

    Supports three graduated-difficulty tasks:
        - classification (Easy): Route to correct department
        - priority_classification (Medium): Classify department + priority
        - efficiency (Hard): Process 10-ticket stream with latency optimization

    Usage:
        env = TicketTriageEnv(task_type="classification", seed=42)
        obs = env.reset()
        done = False
        while not done:
            action = agent.decide(obs)
            obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        task_type: Union[TaskTypeEnum, str] = TaskTypeEnum.CLASSIFICATION,
        seed: Optional[int] = None,
    ) -> None:
        if isinstance(task_type, str):
            task_type = TaskTypeEnum(task_type)
        self.task_type = task_type

        config = TASK_CONFIG[task_type]
        self._tickets_per_episode: int = config["tickets_per_episode"]
        self._max_steps: int = config["max_steps"]

        self._seed = seed
        self._rng = random.Random(seed)
        self._ticket_generator = TicketGenerator(seed=seed)
        self._reward_calculator = RewardCalculator(task_type)
        self._reward_modulator = RewardModulator()

        # Select grader
        if task_type == TaskTypeEnum.CLASSIFICATION:
            self._grader = ClassificationGrader()
        elif task_type == TaskTypeEnum.PRIORITY_CLASSIFICATION:
            self._grader = PriorityClassificationGrader()
        else:
            self._grader = EfficiencyGrader()

        self._state: Optional[EnvironmentState] = None
        self._tickets: List[Dict[str, Any]] = []
        self._current_observation: Optional[Observation] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment to the start of a new episode.

        Returns:
            Initial observation for the first ticket.
        """
        episode_num = (self._state.episode_number + 1) if self._state else 0

        self._state = EnvironmentState(
            task_type=self.task_type,
            episode_number=episode_num,
            step_number=0,
            max_steps=self._max_steps,
            tickets_total=self._tickets_per_episode,
            tickets_processed=0,
            current_ticket_index=0,
            cumulative_reward=0.0,
            episode_done=False,
        )

        self._tickets = self._ticket_generator.generate_episode_tickets(
            task_type=self.task_type.value,
            num_tickets=self._tickets_per_episode,
        )

        self._reward_modulator.reset()
        self._current_observation = self._build_observation(0)
        return self._current_observation

    def step(
        self, action: Union[Action, Dict[str, Any]]
    ) -> Tuple[Optional[Observation], Reward, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.

        Args:
            action: Agent action (Action instance or dict).

        Returns:
            Tuple of (observation, reward, done, info).
            observation is None if episode is done.
        """
        if self._state is None or self._state.episode_done:
            raise RuntimeError("Call reset() before step().")

        if isinstance(action, dict):
            action = Action(**action)

        self._state.step_number += 1
        action_type = ActionTypeEnum(action.action_type)

        # Grade terminal actions
        grader_score = 0.0
        ticket_done = False

        if action_type in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE, ActionTypeEnum.ESCALATE):
            ticket_idx = self._state.current_ticket_index
            ticket_meta = self._tickets[ticket_idx].get("metadata", {})
            obs = self._current_observation

            if self.task_type == TaskTypeEnum.EFFICIENCY:
                grader_score = self._grader.grade(
                    action=action,
                    observation=obs,
                    ticket_metadata=ticket_meta,
                    steps_used=self._state.step_number,
                    max_steps=self._max_steps,
                )
            else:
                grader_score = self._grader.grade(
                    action=action,
                    observation=obs,
                    ticket_metadata=ticket_meta,
                )

            self._state.grader_scores.append(grader_score)
            self._state.tickets_processed += 1
            ticket_done = True

        # Record action in history
        # Use str() to handle both plain-string and enum values (Pydantic v1/v2 compat)
        self._state.actions_taken.append(
            {
                "step": self._state.step_number,
                "action_type": str(action_type.value) if hasattr(action_type, "value") else str(action_type),
                "department": str(action.department) if action.department else None,
                "priority": str(action.priority) if action.priority else None,
            }
        )

        # Calculate reward
        episode_progress = self._state.tickets_processed / self._state.tickets_total
        reward = self._reward_calculator.calculate(
            action=action,
            observation=self._current_observation,
            grader_score=grader_score,
            done=ticket_done,
            steps_used=self._state.step_number,
            max_steps=self._max_steps,
            action_history=[a["action_type"] for a in self._state.actions_taken[:-1]],
        )
        reward = self._reward_modulator.modulate(reward, episode_progress)
        self._state.cumulative_reward += reward.value

        # Advance to next ticket or end episode
        done = False
        next_obs = None

        if ticket_done:
            next_idx = self._state.current_ticket_index + 1
            if next_idx >= self._tickets_per_episode or self._state.step_number >= self._max_steps:
                # Episode complete
                done = True
                self._state.episode_done = True
                self._reward_modulator.end_episode()
            else:
                # Move to next ticket
                self._state.current_ticket_index = next_idx
                self._current_observation = self._build_observation(next_idx)
                next_obs = self._current_observation
        elif self._state.step_number >= self._max_steps:
            # Timeout
            done = True
            self._state.episode_done = True
        else:
            next_obs = self._current_observation

        info = {
            "step": self._state.step_number,
            "ticket_index": self._state.current_ticket_index,
            "grader_score": grader_score,
            "cumulative_reward": round(self._state.cumulative_reward, 4),
            "tickets_processed": self._state.tickets_processed,
            "task_type": self.task_type.value,
        }

        return next_obs, reward, done, info

    def state(self) -> EnvironmentState:
        """Return current environment state."""
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self, ticket_idx: int) -> Observation:
        """Build an Observation from a ticket dict."""
        ticket = self._tickets[ticket_idx]
        return Observation(
            ticket_id=ticket["ticket_id"],
            subject=ticket["subject"],
            description=ticket["description"],
            customer_tier=ticket["customer_tier"],
            sentiment_score=ticket["sentiment_score"],
            wait_time_minutes=ticket["wait_time_minutes"],
            task_type=self.task_type,
            step_number=self._state.step_number if self._state else 0,
            episode_number=self._state.episode_number if self._state else 0,
            tickets_remaining=(
                self._tickets_per_episode - ticket_idx - 1
            ),
            metadata=ticket.get("metadata", {}),
        )
