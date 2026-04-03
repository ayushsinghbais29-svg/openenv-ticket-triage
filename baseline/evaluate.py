"""Evaluation harness for the baseline agent."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment import TicketTriageEnv
from baseline.agent import GPT4Agent


def evaluate_task(
    task_type: str,
    agent: GPT4Agent,
    n_episodes: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run multiple episodes and collect statistics.

    Args:
        task_type: One of classification, priority_classification, efficiency_triage
        agent: Initialized GPT4Agent instance
        n_episodes: Number of episodes to run
        seed: Base random seed (incremented per episode)
        verbose: Print step-by-step details

    Returns:
        Dict with mean_score, std_score, episode_scores, trajectories
    """
    episode_scores: List[float] = []
    trajectories: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        episode_seed = seed + ep
        env = TicketTriageEnv(task_type=task_type, seed=episode_seed)
        agent.reset()

        obs = env.reset()
        trajectory: List[Dict[str, Any]] = []
        episode_reward = 0.0
        done = False
        truncated = False
        score = 0.0

        if verbose:
            print(f"\n--- Episode {ep + 1}/{n_episodes} ({task_type}) ---")

        while not done and not truncated:
            action = agent.decide(obs)

            if verbose:
                print(
                    f"  Step {obs['step']}: {action.get('action_type')} -> "
                    f"dept={action.get('department')} pri={action.get('priority')}"
                )

            try:
                next_obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward.get("value", 0.0)

                trajectory.append(
                    {
                        "step": obs["step"],
                        "ticket_id": obs["ticket_id"],
                        "action": action,
                        "reward": reward,
                    }
                )

                if done or truncated:
                    score = info.get("episode_score", 0.0)

                obs = next_obs
            except Exception as e:
                if verbose:
                    print(f"  Error on step: {e}")
                break

        episode_scores.append(score)
        trajectories.append(
            {
                "episode": ep + 1,
                "score": score,
                "total_reward": episode_reward,
                "steps": len(trajectory),
                "trajectory": trajectory,
            }
        )

        if verbose:
            print(f"  Episode score: {score:.3f}, Total reward: {episode_reward:.3f}")

    import statistics

    mean_score = statistics.mean(episode_scores) if episode_scores else 0.0
    std_score = statistics.stdev(episode_scores) if len(episode_scores) > 1 else 0.0

    return {
        "task_type": task_type,
        "n_episodes": n_episodes,
        "mean_score": round(mean_score, 4),
        "std_score": round(std_score, 4),
        "episode_scores": [round(s, 4) for s in episode_scores],
        "trajectories": trajectories,
    }


def run_full_evaluation(
    n_episodes: int = 5,
    seed: int = 42,
    verbose: bool = True,
    save_results: bool = True,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run evaluation across all three tasks.

    Args:
        n_episodes: Episodes per task
        seed: Base random seed
        verbose: Print progress
        save_results: Save results to JSON
        output_path: Custom path for results JSON

    Returns:
        Dict with results per task and aggregate summary
    """
    task_types = [
        "classification",
        "priority_classification",
        "efficiency_triage",
    ]

    agent = GPT4Agent(seed=seed)
    all_results: Dict[str, Any] = {}

    if verbose:
        print("=" * 60)
        print("OpenEnv Ticket Triage — Baseline Evaluation")
        print(f"Model: {agent.model} | Episodes per task: {n_episodes}")
        print("=" * 60)

    start_time = time.time()

    for task_type in task_types:
        if verbose:
            print(f"\nEvaluating task: {task_type}")

        result = evaluate_task(
            task_type=task_type,
            agent=agent,
            n_episodes=n_episodes,
            seed=seed,
            verbose=verbose,
        )
        all_results[task_type] = result

        if verbose:
            print(
                f"  Mean score: {result['mean_score']:.4f} "
                f"± {result['std_score']:.4f}"
            )
            print(f"  Episode scores: {result['episode_scores']}")

    elapsed = time.time() - start_time
    summary = {
        "classification_score": all_results["classification"]["mean_score"],
        "priority_classification_score": all_results["priority_classification"]["mean_score"],
        "efficiency_triage_score": all_results["efficiency_triage"]["mean_score"],
        "overall_mean": round(
            (
                all_results["classification"]["mean_score"]
                + all_results["priority_classification"]["mean_score"]
                + all_results["efficiency_triage"]["mean_score"]
            )
            / 3,
            4,
        ),
        "evaluation_time_seconds": round(elapsed, 2),
        "model": agent.model,
        "seed": seed,
        "n_episodes": n_episodes,
    }

    output = {
        "summary": summary,
        "results": all_results,
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": agent.model,
            "seed": seed,
        },
    }

    if save_results:
        path = output_path or os.path.join(os.path.dirname(__file__), "results.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"\nResults saved to {path}")

    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Classification (Easy):        {summary['classification_score']:.4f}")
        print(f"  Priority+Class (Medium):      {summary['priority_classification_score']:.4f}")
        print(f"  Efficiency Triage (Hard):     {summary['efficiency_triage_score']:.4f}")
        print(f"  Overall Mean:                 {summary['overall_mean']:.4f}")
        print(f"  Total time: {summary['evaluation_time_seconds']}s")
        print("=" * 60)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--n-episodes", type=int, default=5, help="Episodes per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    run_full_evaluation(
        n_episodes=args.n_episodes,
        seed=args.seed,
        verbose=not args.quiet,
        save_results=True,
        output_path=args.output,
    )
