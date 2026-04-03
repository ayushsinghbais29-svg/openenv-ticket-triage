"""Tests for the TicketTriageEnv environment."""

from __future__ import annotations

import pytest

from src.environment import TicketTriageEnv
from src.models import ActionTypeEnum, DepartmentEnum, PriorityEnum


class TestReset:
    def test_reset_returns_observation(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "ticket_id" in obs
        assert "subject" in obs
        assert "description" in obs
        assert "step" in obs
        assert obs["step"] == 0

    def test_reset_idempotent(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        obs1 = env.reset()
        obs2 = env.reset()
        assert obs1["ticket_id"] == obs2["ticket_id"]
        assert obs1["subject"] == obs2["subject"]

    def test_reset_clears_state(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        env.step({"action_type": "read"})
        env.reset()
        assert env._step == 0
        assert env._current_idx == 0
        assert env._done is False
        assert env._total_reward == 0.0

    def test_reset_all_tasks(self):
        for task_type in ["classification", "priority_classification", "efficiency_triage"]:
            env = TicketTriageEnv(task_type=task_type, seed=1)
            obs = env.reset()
            assert obs["task_type"] == task_type

    def test_reset_generates_correct_ticket_count(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        assert len(env._tickets) == 5  # classification: 5 tickets

        env2 = TicketTriageEnv(task_type="efficiency_triage", seed=42)
        env2.reset()
        assert len(env2._tickets) == 10  # efficiency: 10 tickets


class TestStep:
    def test_step_requires_reset(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        with pytest.raises(RuntimeError):
            env.step({"action_type": "read"})

    def test_step_returns_tuple(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        result = env.step({"action_type": "read"})
        assert len(result) == 5
        obs, reward, done, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, dict)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_increments_step(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        obs, _, _, _, _ = env.step({"action_type": "read"})
        assert obs["step"] == 1

    def test_read_action_reward(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        _, reward, _, _, _ = env.step({"action_type": "read"})
        assert reward["value"] > 0  # reading gives positive reward

    def test_invalid_action_penalized(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        _, reward, done, truncated, _ = env.step(
            {"action_type": "route", "department": "InvalidDept"}
        )
        assert reward["value"] < 0  # invalid action penalized
        assert done is False

    def test_route_advances_ticket(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        obs = env.reset()
        initial_processed = obs["tickets_processed"]
        obs, _, _, _, _ = env.step(
            {"action_type": "route", "department": "Billing", "confidence": 0.9}
        )
        assert obs["tickets_processed"] > initial_processed

    def test_episode_ends_after_all_tickets(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            _, _, done, truncated, info = env.step(
                {"action_type": "route", "department": "Billing", "confidence": 0.9}
            )
            steps += 1
            if truncated:
                break
        assert done or truncated

    def test_episode_score_in_info_on_done(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        done = False
        truncated = False
        info = {}
        steps = 0
        while not done and not truncated and steps < 50:
            _, _, done, truncated, info = env.step(
                {"action_type": "route", "department": "Billing", "confidence": 0.9}
            )
            steps += 1
        assert "episode_score" in info
        assert 0.0 <= info["episode_score"] <= 1.0

    def test_truncation_on_max_steps(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        truncated = False
        done = False
        steps = 0
        while not done and not truncated:
            _, _, done, truncated, _ = env.step({"action_type": "read"})
            steps += 1
            if steps > 100:
                break
        assert truncated or done

    def test_reward_value_in_range(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        for _ in range(5):
            _, reward, done, truncated, _ = env.step({"action_type": "read"})
            assert -2.0 <= reward["value"] <= 2.0
            if done or truncated:
                break


class TestState:
    def test_state_before_reset(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        state = env.state()
        assert isinstance(state, dict)

    def test_state_after_reset(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        state = env.state()
        assert state["step"] == 0
        assert state["done"] is False
        assert state["total_reward"] == 0.0
        assert isinstance(state["tickets"], list)

    def test_state_consistency(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        env.reset()
        env.step({"action_type": "read"})
        state = env.state()
        assert state["step"] == 1
        assert len(state["action_history"]) == 1

    def test_state_task_type(self):
        for task_type in ["classification", "priority_classification", "efficiency_triage"]:
            env = TicketTriageEnv(task_type=task_type, seed=1)
            env.reset()
            state = env.state()
            assert state["task_type"] == task_type


class TestFullEpisode:
    def test_classification_episode(self):
        env = TicketTriageEnv(task_type="classification", seed=42)
        obs = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        while not done and not truncated and steps < 50:
            action = {"action_type": "route", "department": "Billing", "confidence": 0.9}
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward["value"]
            steps += 1

        assert done or truncated
        if done:
            assert info.get("episode_score", -1) >= 0.0

    def test_priority_episode(self):
        env = TicketTriageEnv(task_type="priority_classification", seed=42)
        env.reset()
        done = False
        truncated = False
        steps = 0
        classify_step = True

        while not done and not truncated and steps < 50:
            if classify_step:
                action = {"action_type": "classify", "department": "Billing", "confidence": 0.8}
            else:
                action = {"action_type": "set_priority", "priority": "High", "confidence": 0.8}
            obs, reward, done, truncated, info = env.step(action)
            classify_step = not classify_step
            steps += 1

        assert done or truncated

    def test_efficiency_episode(self):
        env = TicketTriageEnv(task_type="efficiency_triage", seed=42)
        env.reset()
        done = False
        truncated = False
        steps = 0

        while not done and not truncated and steps < 50:
            action = {"action_type": "route", "department": "Technical", "confidence": 0.85}
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

        assert done or truncated
