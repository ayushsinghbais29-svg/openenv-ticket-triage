"""
Full evaluation harness for the GPT-4 baseline agent.
Runs all three task types and reports per-task scores.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment import TicketTriageEnv
from src.models import TaskTypeEnum
from baseline.agent import GPT4Agent

SEED = 42
NUM_EPISODES = 5  # Episodes per task type

TASK_CONFIGS = [
    {"task_type": TaskTypeEnum.CLASSIFICATION, "label": "Task 1: Classification (Easy)"},
    {"task_type": TaskTypeEnum.PRIORITY_CLASSIFICATION, "label": "Task 2: Priority Classification (Medium)"},
    {"task_type": TaskTypeEnum.EFFICIENCY, "label": "Task 3: Efficiency Triage (Hard)"},
]


def run_episode(env: TicketTriageEnv, agent: GPT4Agent) -> Dict[str, Any]:
    """Run a single episode and return results."""
    obs = env.reset()
    done = False
    episode_reward = 0.0
    grader_scores = []
    steps = 0

    while not done and obs is not None:
        action = agent.decide(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward.value
        steps += 1
        if reward.grader_score > 0:
            grader_scores.append(reward.grader_score)

    avg_score = sum(grader_scores) / len(grader_scores) if grader_scores else 0.0

    return {
        "episode_reward": round(episode_reward, 4),
        "avg_grader_score": round(avg_score, 4),
        "grader_scores": [round(s, 4) for s in grader_scores],
        "steps": steps,
    }


def evaluate_task(task_type: TaskTypeEnum, label: str, agent: GPT4Agent) -> Dict[str, Any]:
    """Evaluate agent on a specific task type."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    env = TicketTriageEnv(task_type=task_type, seed=SEED)
    episode_results = []

    for ep in range(NUM_EPISODES):
        print(f"  Episode {ep + 1}/{NUM_EPISODES}...", end=" ", flush=True)
        result = run_episode(env, agent)
        episode_results.append(result)
        print(f"Score: {result['avg_grader_score']:.3f} | Reward: {result['episode_reward']:.3f}")

    scores = [r["avg_grader_score"] for r in episode_results]
    mean_score = sum(scores) / len(scores)
    mean_reward = sum(r["episode_reward"] for r in episode_results) / len(episode_results)

    print(f"\n  Mean Score: {mean_score:.3f} | Mean Reward: {mean_reward:.3f}")

    return {
        "task_type": task_type.value,
        "label": label,
        "num_episodes": NUM_EPISODES,
        "mean_score": round(mean_score, 4),
        "mean_reward": round(mean_reward, 4),
        "episodes": episode_results,
    }


def main() -> None:
    print("\n" + "=" * 60)
    print("  OpenEnv Ticket Triage - Baseline Evaluation")
    print("  Model: GPT-4 | Seed: 42 | Episodes per task: 5")
    print("=" * 60)

    agent = GPT4Agent(model="gpt-4o-mini", temperature=0.0, seed=SEED)
    all_results = []
    start_time = time.time()

    for config in TASK_CONFIGS:
        result = evaluate_task(
            task_type=config["task_type"],
            label=config["label"],
            agent=agent,
        )
        all_results.append(result)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['label']}: {r['mean_score']:.1%}")
    print(f"\n  Total evaluation time: {elapsed:.1f}s")

    # Save results
    output = {
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "evaluation_time_seconds": round(elapsed, 1),
        "results": all_results,
    }

    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
