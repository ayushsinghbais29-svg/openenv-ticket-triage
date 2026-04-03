"""
Unit tests for TicketTriageEnv - reset, step, and state methods.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment import TicketTriageEnv
from src.models import (
    Action,
    ActionTypeEnum,
    DepartmentEnum,
    Observation,
    PriorityEnum,
    TaskTypeEnum,
)


CLASSIFY_ACTION = Action(
    action_type=ActionTypeEnum.CLASSIFY,
    department=DepartmentEnum.BILLING,
    confidence=0.9,
)

READ_ACTION = Action(action_type=ActionTypeEnum.READ)


class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        obs = env.reset()
        assert isinstance(obs, Observation)

    def test_reset_initializes_state(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        state = env.state()
        assert state.step_number == 0
        assert state.episode_done is False
        assert state.tickets_processed == 0

    def test_reset_produces_valid_ticket(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        obs = env.reset()
        assert obs.ticket_id.startswith("TKT-")
        assert len(obs.subject) > 0
        assert len(obs.description) > 0
        assert -1.0 <= obs.sentiment_score <= 1.0

    def test_multiple_resets_increment_episode(self):
        env = TicketTriageEnv(seed=42)
        env.reset()
        state1 = env.state()
        ep1 = state1.episode_number
        env.reset()
        state2 = env.state()
        assert state2.episode_number == ep1 + 1

    def test_step_before_reset_raises(self):
        env = TicketTriageEnv(seed=42)
        with pytest.raises(RuntimeError):
            env.step(CLASSIFY_ACTION)


class TestEnvironmentStep:
    def test_step_returns_tuple(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        result = env.step(READ_ACTION)
        assert len(result) == 4  # obs, reward, done, info

    def test_read_action_gives_progress_reward(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        _, reward, done, _ = env.step(READ_ACTION)
        assert reward.value > 0
        assert done is False

    def test_classify_action_terminates_ticket(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        _, reward, done, info = env.step(CLASSIFY_ACTION)
        assert info["tickets_processed"] == 1

    def test_full_episode_classification(self):
        """Run through a complete classification episode."""
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        obs = env.reset()
        done = False
        steps = 0
        while not done and obs is not None:
            action = Action(
                action_type=ActionTypeEnum.CLASSIFY,
                department=DepartmentEnum.BILLING,
            )
            obs, reward, done, info = env.step(action)
            steps += 1
            assert steps <= 20, "Episode should end before max steps"
        assert done is True

    def test_step_accepts_dict_action(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        action_dict = {
            "action_type": "classify",
            "department": "billing",
        }
        obs, reward, done, info = env.step(action_dict)
        assert reward is not None

    def test_grader_score_in_reward(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        _, reward, _, _ = env.step(CLASSIFY_ACTION)
        assert 0.0 <= reward.grader_score <= 1.0

    def test_info_contains_expected_keys(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        _, _, _, info = env.step(CLASSIFY_ACTION)
        assert "step" in info
        assert "grader_score" in info
        assert "cumulative_reward" in info


class TestEnvironmentState:
    def test_state_tracks_steps(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        env.step(READ_ACTION)
        env.step(READ_ACTION)
        state = env.state()
        assert state.step_number == 2

    def test_state_tracks_processed_tickets(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        env.step(CLASSIFY_ACTION)
        state = env.state()
        assert state.tickets_processed == 1

    def test_cumulative_reward_increases(self):
        env = TicketTriageEnv(task_type=TaskTypeEnum.CLASSIFICATION, seed=42)
        env.reset()
        env.step(READ_ACTION)
        env.step(READ_ACTION)
        state = env.state()
        assert state.cumulative_reward != 0.0

    def test_state_before_reset_raises(self):
        env = TicketTriageEnv(seed=42)
        with pytest.raises(RuntimeError):
            env.state()


class TestAllTaskTypes:
    @pytest.mark.parametrize("task_type", [
        TaskTypeEnum.CLASSIFICATION,
        TaskTypeEnum.PRIORITY_CLASSIFICATION,
        TaskTypeEnum.EFFICIENCY,
    ])
    def test_task_type_resets_and_steps(self, task_type):
        env = TicketTriageEnv(task_type=task_type, seed=42)
        obs = env.reset()
        assert obs is not None
        assert obs.task_type == task_type

        action = Action(
            action_type=ActionTypeEnum.CLASSIFY,
            department=DepartmentEnum.GENERAL,
            priority=PriorityEnum.MEDIUM,
        )
        _, reward, _, _ = env.step(action)
        assert reward is not None
