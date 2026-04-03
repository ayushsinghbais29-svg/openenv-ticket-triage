"""
Multi-component reward functions for the OpenEnv Ticket Triage environment.
Provides dense reward signals throughout agent trajectories.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .models import (
    Action,
    ActionTypeEnum,
    CustomerTierEnum,
    Observation,
    Reward,
    RewardComponents,
    TaskTypeEnum,
)


# ---------------------------------------------------------------------------
# Per-action reward constants
# ---------------------------------------------------------------------------

READ_REWARD = 0.05
ANALYZE_REWARD = 0.10
REPETITION_PENALTY = -0.10
TIMEOUT_PENALTY = -1.0
ESCALATION_BONUS = 0.15

TIER_MULTIPLIERS: Dict[CustomerTierEnum, float] = {
    CustomerTierEnum.FREE: 1.0,
    CustomerTierEnum.PREMIUM: 1.1,
    CustomerTierEnum.ENTERPRISE: 1.25,
}


# ---------------------------------------------------------------------------
# RewardCalculator
# ---------------------------------------------------------------------------


class RewardCalculator:
    """
    Calculates scalar reward for an action given context.
    Supports dense intermediate rewards and terminal correctness rewards.
    """

    def __init__(self, task_type: TaskTypeEnum) -> None:
        self.task_type = task_type

    def calculate(
        self,
        action: Action,
        observation: Observation,
        grader_score: float,
        done: bool,
        steps_used: int,
        max_steps: int,
        action_history: Optional[list] = None,
    ) -> Reward:
        """
        Calculate reward for a single action.

        Args:
            action: The action taken by the agent.
            observation: Current observation.
            grader_score: Score from the task grader (0.0-1.0).
            done: Whether the episode is done.
            steps_used: Steps used so far.
            max_steps: Maximum allowed steps.
            action_history: List of previous action types.

        Returns:
            Reward object with components and scalar value.
        """
        action_history = action_history or []
        components = RewardComponents()

        action_type = ActionTypeEnum(action.action_type)

        # --- Intermediate action rewards ---
        if action_type == ActionTypeEnum.READ:
            # Count consecutive reads to penalize repetition
            recent_reads = sum(
                1 for a in action_history[-5:] if a == ActionTypeEnum.READ.value
            )
            if recent_reads >= 2:
                components.penalties += REPETITION_PENALTY
            else:
                components.progress += READ_REWARD

        elif action_type == ActionTypeEnum.ANALYZE:
            components.progress += ANALYZE_REWARD

        # --- Terminal action rewards ---
        elif action_type in (ActionTypeEnum.CLASSIFY, ActionTypeEnum.ROUTE, ActionTypeEnum.ESCALATE):
            # Correctness component
            components.correctness = grader_score

            # Efficiency component: bonus for speed, penalty for slowness
            step_ratio = steps_used / max_steps
            if step_ratio <= 0.25:
                components.efficiency += 0.15  # Fast bonus
            elif step_ratio >= 0.75:
                components.efficiency -= 0.10  # Slow penalty

            # Escalation bonus for critical tickets
            wait = observation.wait_time_minutes
            if wait > 60.0 and action_type == ActionTypeEnum.ESCALATE:
                components.efficiency += ESCALATION_BONUS

            # Tier multiplier applied to correctness
            tier = CustomerTierEnum(observation.customer_tier)
            tier_mult = TIER_MULTIPLIERS.get(tier, 1.0)
            components.correctness = min(1.0, components.correctness * tier_mult)

        elif action_type == ActionTypeEnum.CLOSE:
            # Closing without routing is slightly penalized
            components.penalties -= 0.05

        # --- Timeout penalty ---
        if steps_used >= max_steps and not done:
            components.penalties += TIMEOUT_PENALTY

        scalar = components.total

        return Reward(
            value=round(scalar, 4),
            components=components,
            grader_score=round(grader_score, 4),
            done=done,
            info={
                "task_type": self.task_type.value,
                "action_type": action_type.value,
                "steps_used": steps_used,
                "step_ratio": round(steps_used / max_steps, 3),
            },
        )


# ---------------------------------------------------------------------------
# RewardModulator
# ---------------------------------------------------------------------------


class RewardModulator:
    """
    Modulates reward based on episode-level context.
    Tracks cumulative episode rewards and applies bonuses/penalties.
    """

    def __init__(self) -> None:
        self._episode_rewards: list = []
        self._episode_count = 0

    def reset(self) -> None:
        """Reset modulator state for new episode."""
        self._episode_rewards = []

    def modulate(self, reward: Reward, episode_progress: float) -> Reward:
        """
        Apply episode-level modulation to reward.

        Args:
            reward: Base reward from RewardCalculator.
            episode_progress: Float in [0, 1] representing episode completion.

        Returns:
            Modulated Reward object.
        """
        self._episode_rewards.append(reward.value)

        # Streak bonus: consecutive correct actions get a small bonus
        recent = self._episode_rewards[-3:]
        streak_bonus = 0.0
        if len(recent) == 3 and all(r > 0.5 for r in recent):
            streak_bonus = 0.05

        # End-of-episode bonus if high average performance
        episode_bonus = 0.0
        if episode_progress >= 0.95 and len(self._episode_rewards) > 0:
            avg = sum(self._episode_rewards) / len(self._episode_rewards)
            if avg > 0.7:
                episode_bonus = 0.10

        modulated_value = reward.value + streak_bonus + episode_bonus

        return Reward(
            value=round(modulated_value, 4),
            components=reward.components,
            grader_score=reward.grader_score,
            done=reward.done,
            info={
                **reward.info,
                "streak_bonus": streak_bonus,
                "episode_bonus": episode_bonus,
            },
        )

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def end_episode(self) -> float:
        """Finalize episode and return mean episode reward."""
        self._episode_count += 1
        if not self._episode_rewards:
            return 0.0
        mean = sum(self._episode_rewards) / len(self._episode_rewards)
        self.reset()
        return mean
