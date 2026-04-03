"""
Unit tests for reward calculation and modulation.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    Action,
    ActionTypeEnum,
    CustomerTierEnum,
    DepartmentEnum,
    Observation,
    PriorityEnum,
    Reward,
    RewardComponents,
    TaskTypeEnum,
)
from src.reward_functions import RewardCalculator, RewardModulator


def make_obs(tier=CustomerTierEnum.FREE, wait_time=10.0) -> Observation:
    return Observation(
        ticket_id="TKT-001",
        subject="Test",
        description="Test description",
        customer_tier=tier,
        sentiment_score=0.0,
        wait_time_minutes=wait_time,
        task_type=TaskTypeEnum.CLASSIFICATION,
    )


class TestRewardCalculator:
    def setup_method(self):
        self.calculator = RewardCalculator(task_type=TaskTypeEnum.CLASSIFICATION)

    def test_read_action_gives_progress_reward(self):
        action = Action(action_type=ActionTypeEnum.READ)
        obs = make_obs()
        reward = self.calculator.calculate(
            action=action, observation=obs, grader_score=0.0,
            done=False, steps_used=1, max_steps=15,
        )
        assert reward.value > 0
        assert reward.components.progress > 0

    def test_analyze_action_reward(self):
        action = Action(action_type=ActionTypeEnum.ANALYZE)
        obs = make_obs()
        reward = self.calculator.calculate(
            action=action, observation=obs, grader_score=0.0,
            done=False, steps_used=1, max_steps=15,
        )
        assert reward.components.progress > 0

    def test_classify_correct_high_reward(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
        )
        obs = make_obs()
        reward = self.calculator.calculate(
            action=action, observation=obs, grader_score=0.95,
            done=True, steps_used=3, max_steps=15,
        )
        assert reward.value > 0.5
        assert reward.grader_score == 0.95

    def test_classify_wrong_low_reward(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.TECHNICAL,
        )
        obs = make_obs()
        reward = self.calculator.calculate(
            action=action, observation=obs, grader_score=0.0,
            done=True, steps_used=3, max_steps=15,
        )
        assert reward.components.correctness == 0.0

    def test_timeout_penalty(self):
        action = Action(action_type=ActionTypeEnum.READ)
        obs = make_obs()
        reward = self.calculator.calculate(
            action=action, observation=obs, grader_score=0.0,
            done=False, steps_used=15, max_steps=15,  # At max steps
        )
        assert reward.components.penalties < 0

    def test_repetitive_reads_penalized(self):
        action = Action(action_type=ActionTypeEnum.READ)
        obs = make_obs()
        # History has 2 reads already
        history = ["read", "read"]
        reward = self.calculator.calculate(
            action=action, observation=obs, grader_score=0.0,
            done=False, steps_used=3, max_steps=15,
            action_history=history,
        )
        # Should have penalty for repetition
        assert reward.components.penalties < 0

    def test_fast_classify_efficiency_bonus(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
        )
        obs = make_obs()
        reward = self.calculator.calculate(
            action=action, observation=obs, grader_score=0.9,
            done=True, steps_used=2, max_steps=15,  # Very fast
        )
        assert reward.components.efficiency > 0

    def test_enterprise_tier_multiplier(self):
        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.BILLING,
        )
        free_obs = make_obs(tier=CustomerTierEnum.FREE)
        enterprise_obs = make_obs(tier=CustomerTierEnum.ENTERPRISE)

        free_reward = self.calculator.calculate(
            action=action, observation=free_obs, grader_score=0.8,
            done=True, steps_used=3, max_steps=15,
        )
        enterprise_reward = self.calculator.calculate(
            action=action, observation=enterprise_obs, grader_score=0.8,
            done=True, steps_used=3, max_steps=15,
        )
        # Enterprise should get higher or equal correctness
        assert enterprise_reward.components.correctness >= free_reward.components.correctness

    def test_reward_info_contains_task_type(self):
        action = Action(action_type=ActionTypeEnum.READ)
        obs = make_obs()
        reward = self.calculator.calculate(
            action=action, observation=obs, grader_score=0.0,
            done=False, steps_used=1, max_steps=15,
        )
        assert "task_type" in reward.info

    def test_different_task_types(self):
        for task_type in TaskTypeEnum:
            calc = RewardCalculator(task_type=task_type)
            action = Action(
                action_type=ActionTypeEnum.CLASSIFY,
                department=DepartmentEnum.GENERAL,
                priority=PriorityEnum.MEDIUM,
            )
            obs = Observation(
                ticket_id="TKT-001",
                subject="Test",
                description="Test",
                customer_tier=CustomerTierEnum.FREE,
                sentiment_score=0.0,
                task_type=task_type,
            )
            reward = calc.calculate(
                action=action, observation=obs, grader_score=0.7,
                done=True, steps_used=5, max_steps=20,
            )
            assert isinstance(reward, Reward)


class TestRewardModulator:
    def setup_method(self):
        self.modulator = RewardModulator()

    def test_modulate_returns_reward(self):
        r = Reward(value=0.5, done=False)
        modulated = self.modulator.modulate(r, episode_progress=0.5)
        assert isinstance(modulated, Reward)

    def test_streak_bonus_for_good_performance(self):
        self.modulator.reset()
        r = Reward(value=0.8, done=False)
        # Add three high rewards to history
        for _ in range(3):
            self.modulator.modulate(r, episode_progress=0.5)

        # Fourth call should potentially include streak bonus
        result = self.modulator.modulate(r, episode_progress=0.8)
        assert "streak_bonus" in result.info

    def test_end_episode_returns_mean(self):
        self.modulator.reset()
        r = Reward(value=0.8, done=False)
        self.modulator.modulate(r, episode_progress=0.5)
        self.modulator.modulate(r, episode_progress=0.7)
        mean = self.modulator.end_episode()
        assert 0.0 <= mean <= 2.0  # mean of rewards

    def test_reset_clears_history(self):
        r = Reward(value=0.9, done=False)
        self.modulator.modulate(r, episode_progress=0.5)
        self.modulator.reset()
        mean = self.modulator.end_episode()
        assert mean == 0.0

    def test_episode_count_increments(self):
        initial = self.modulator.episode_count
        self.modulator.end_episode()
        assert self.modulator.episode_count == initial + 1
