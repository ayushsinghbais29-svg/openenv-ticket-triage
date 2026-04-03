"""
OpenEnv Ticket Triage - Beautiful Gradio Dashboard
Meta PyTorch OpenEnv Hackathon 2026

A professional, interactive UI for exploring the ticket triage environment.
Features:
  - Gradient header with professional branding
  - Task selection radio buttons
  - Real-time ticket display
  - AI suggestion panel with mock reasoning
  - Action execution controls
  - Real-time Plotly visualizations (reward chart, grader score chart)
  - Episode history table
  - Reward component breakdown
  - Professional CSS styling
  - Responsive design
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure the src package is importable from the deployment directory
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from src.environment import TicketTriageEnv
from src.models import (
    Action,
    ActionTypeEnum,
    CustomerTierEnum,
    DepartmentEnum,
    Observation,
    PriorityEnum,
    TaskTypeEnum,
)
from src.utils import department_label, format_ticket_for_prompt, priority_label

# ---------------------------------------------------------------------------
# Global session state (per Gradio session via gr.State)
# ---------------------------------------------------------------------------

TASK_LABEL_MAP = {
    "🟢 Classification (Easy)": TaskTypeEnum.CLASSIFICATION,
    "🟡 Priority Classification (Medium)": TaskTypeEnum.PRIORITY_CLASSIFICATION,
    "🔴 Efficiency Triage (Hard)": TaskTypeEnum.EFFICIENCY,
}

DEPT_CHOICES = [
    ("🏦 Billing", "billing"),
    ("🔧 Technical Support", "technical"),
    ("💬 General Support", "general"),
    ("⭐ Premium Support", "premium_support"),
]

PRIORITY_CHOICES = [
    ("🟢 Low", "low"),
    ("🟡 Medium", "medium"),
    ("🟠 High", "high"),
    ("🔴 Critical", "critical"),
]


def _make_default_session() -> Dict[str, Any]:
    return {
        "env": None,
        "task_type": TaskTypeEnum.CLASSIFICATION,
        "current_obs": None,
        "done": False,
        "reward_history": [],
        "grader_history": [],
        "episode_log": [],
        "step_count": 0,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def build_reward_chart(reward_history: List[float]) -> go.Figure:
    """Build a line chart of cumulative rewards per step."""
    fig = go.Figure()
    if not reward_history:
        fig.add_annotation(
            text="No data yet — start an episode!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888"),
        )
    else:
        steps = list(range(1, len(reward_history) + 1))
        cumulative = []
        total = 0.0
        for r in reward_history:
            total += r
            cumulative.append(round(total, 4))

        fig.add_trace(go.Scatter(
            x=steps,
            y=cumulative,
            mode="lines+markers",
            name="Cumulative Reward",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=8, color="#818cf8"),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.12)",
        ))
        fig.add_trace(go.Bar(
            x=steps,
            y=reward_history,
            name="Step Reward",
            marker_color=[
                "#22c55e" if r > 0 else "#ef4444" for r in reward_history
            ],
            opacity=0.6,
        ))

    fig.update_layout(
        title=dict(text="📈 Reward Trajectory", font=dict(size=18, color="#1e293b")),
        xaxis_title="Step",
        yaxis_title="Reward",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
        height=300,
    )
    return fig


def build_grader_chart(grader_history: List[float]) -> go.Figure:
    """Build a bar chart of grader scores per ticket."""
    fig = go.Figure()
    if not grader_history:
        fig.add_annotation(
            text="No graded tickets yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888"),
        )
    else:
        tickets = list(range(1, len(grader_history) + 1))
        colors = []
        for s in grader_history:
            if s >= 0.85:
                colors.append("#22c55e")
            elif s >= 0.60:
                colors.append("#eab308")
            else:
                colors.append("#ef4444")

        fig.add_trace(go.Bar(
            x=tickets,
            y=grader_history,
            marker_color=colors,
            name="Grader Score",
            text=[f"{s:.2f}" for s in grader_history],
            textposition="outside",
        ))
        fig.add_hline(
            y=0.85, line_dash="dot", line_color="#6366f1",
            annotation_text="Target: 85%", annotation_position="right",
        )

    fig.update_layout(
        title=dict(text="🎯 Grader Scores per Ticket", font=dict(size=18, color="#1e293b")),
        xaxis_title="Ticket #",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.15]),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(l=40, r=20, t=60, b=40),
        height=300,
    )
    return fig


def build_reward_breakdown_chart(components: Dict[str, float]) -> go.Figure:
    """Build a horizontal bar chart of reward components."""
    labels = ["Correctness", "Efficiency", "Progress", "Penalties"]
    values = [
        components.get("correctness", 0.0),
        components.get("efficiency", 0.0),
        components.get("progress", 0.0),
        components.get("penalties", 0.0),
    ]
    colors = ["#22c55e", "#3b82f6", "#a855f7", "#ef4444"]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="🧩 Reward Breakdown", font=dict(size=16, color="#1e293b")),
        xaxis_title="Value",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(l=100, r=60, t=50, b=40),
        height=240,
        xaxis=dict(range=[-0.3, 1.3]),
    )
    return fig


# ---------------------------------------------------------------------------
# Mock AI suggestion generator
# ---------------------------------------------------------------------------

_DEPT_REASONING = {
    "billing": [
        "The ticket mentions payment issues and charges — clearly a Billing matter.",
        "Keywords: invoice, charge, refund. → Routing to Billing department.",
        "Customer reports unexpected debit — this requires Billing team intervention.",
    ],
    "technical": [
        "API errors and system failures indicate a Technical Support issue.",
        "Keywords: 500 error, integration, crash. → Technical Support team.",
        "Performance degradation and service outages need engineering attention.",
    ],
    "general": [
        "This is an account management inquiry — General Support can handle this.",
        "Feature requests and policy questions are handled by General Support.",
        "Password reset and access issues go to the General Support queue.",
    ],
    "premium_support": [
        "Enterprise customer with critical SLA requirements — Premium Support escalation.",
        "Production outage affecting multiple users — immediate Premium Support needed.",
        "Dedicated instance and compliance requests are Premium Support scope.",
    ],
}

_PRIORITY_REASONING = {
    "low": "Low urgency — no immediate business impact.",
    "medium": "Moderate impact — should be addressed within 1-2 business days.",
    "high": "Significant impact — requires attention within hours.",
    "critical": "CRITICAL — production-level issue, immediate response required!",
}


def _mock_ai_suggestion(obs: Observation, task_type: TaskTypeEnum) -> Tuple[str, str, str, str]:
    """
    Generate a plausible AI suggestion based on observation keywords.

    Returns:
        Tuple of (department, priority, reasoning, confidence_str).
    """
    subject = (obs.subject + " " + obs.description).lower()

    # Determine most likely department
    scores = {
        "billing": sum(1 for kw in ["payment", "charge", "invoice", "refund", "billing", "money", "cost", "price"] if kw in subject),
        "technical": sum(1 for kw in ["error", "api", "crash", "bug", "slow", "down", "fail", "integration", "500"] if kw in subject),
        "general": sum(1 for kw in ["account", "password", "feature", "how", "policy", "gdpr", "data", "access"] if kw in subject),
        "premium_support": sum(1 for kw in ["outage", "production", "enterprise", "sla", "dedicated", "critical", "urgent"] if kw in subject),
    }
    best_dept = max(scores, key=scores.get)
    if scores[best_dept] == 0:
        best_dept = random.choice(["billing", "technical", "general"])

    # Determine priority based on sentiment and wait time
    if obs.sentiment_score < -0.7 or obs.wait_time_minutes > 90:
        best_priority = "critical"
    elif obs.sentiment_score < -0.4 or obs.wait_time_minutes > 45:
        best_priority = "high"
    elif obs.sentiment_score < 0.0:
        best_priority = "medium"
    else:
        best_priority = "low"

    # Build reasoning
    dept_reason = random.choice(_DEPT_REASONING.get(best_dept, ["Analyzing ticket content..."]))
    priority_reason = _PRIORITY_REASONING.get(best_priority, "Moderate urgency.")

    reasoning = f"**Department Analysis:** {dept_reason}\n\n**Priority Assessment:** {priority_reason}"
    if obs.customer_tier in ("premium", "enterprise"):
        reasoning += f"\n\n**⚠️ Tier Alert:** {obs.customer_tier.upper()} customer — apply escalation protocol."

    confidence = min(0.99, 0.70 + (scores[best_dept] * 0.05) + random.uniform(0.0, 0.1))
    confidence_str = f"{confidence:.0%}"

    return best_dept, best_priority, reasoning, confidence_str


# ---------------------------------------------------------------------------
# Core action handlers
# ---------------------------------------------------------------------------


def start_episode(task_label: str, session: Dict) -> Tuple:
    """Initialize a new episode for the selected task."""
    task_type = TASK_LABEL_MAP.get(task_label, TaskTypeEnum.CLASSIFICATION)
    env = TicketTriageEnv(task_type=task_type, seed=session.get("seed", 42))
    obs = env.reset()

    session.update({
        "env": env,
        "task_type": task_type,
        "current_obs": obs,
        "done": False,
        "reward_history": [],
        "grader_history": [],
        "episode_log": [],
        "step_count": 0,
    })

    ticket_html = _render_ticket_html(obs)
    ai_dept, ai_priority, ai_reasoning, ai_conf = _mock_ai_suggestion(obs, task_type)
    status = f"✅ Episode started | Task: **{task_label}** | Ticket 1 of {env._tickets_per_episode}"

    reward_chart = build_reward_chart([])
    grader_chart = build_grader_chart([])
    breakdown_chart = build_reward_breakdown_chart({})
    history_df = pd.DataFrame(columns=["Step", "Action", "Department", "Priority", "Reward", "Score"])

    return (
        session,
        ticket_html,
        ai_reasoning,
        f"🎯 Suggested: **{department_label(ai_dept)}** | Confidence: {ai_conf}",
        ai_dept,
        ai_priority,
        status,
        reward_chart,
        grader_chart,
        breakdown_chart,
        history_df,
        gr.update(interactive=True),   # Execute button
        gr.update(interactive=True),   # Read button
        gr.update(interactive=True),   # Analyze button
    )


def execute_action(
    dept_choice: str,
    priority_choice: str,
    action_type_str: str,
    session: Dict,
) -> Tuple:
    """Execute a terminal action (classify/route/escalate)."""
    env: Optional[TicketTriageEnv] = session.get("env")
    obs: Optional[Observation] = session.get("current_obs")

    if env is None or obs is None or session.get("done"):
        return _episode_done_state(session)

    try:
        dept = DepartmentEnum(dept_choice) if dept_choice else DepartmentEnum.GENERAL
        priority = PriorityEnum(priority_choice) if priority_choice else None
        action_type = ActionTypeEnum(action_type_str)
    except ValueError:
        dept = DepartmentEnum.GENERAL
        priority = None
        action_type = ActionTypeEnum.CLASSIFY

    action = Action(
        action_type=action_type,
        department=dept,
        priority=priority,
        confidence=0.9,
    )

    next_obs, reward, done, info = env.step(action)
    session["step_count"] += 1
    session["reward_history"].append(reward.value)
    if reward.grader_score > 0:
        session["grader_history"].append(reward.grader_score)

    # Log entry
    session["episode_log"].append({
        "Step": session["step_count"],
        "Action": action_type.value.title(),
        "Department": department_label(dept),
        "Priority": priority_label(priority) if priority else "—",
        "Reward": f"{reward.value:+.3f}",
        "Score": f"{reward.grader_score:.3f}",
    })

    session["done"] = done
    session["current_obs"] = next_obs

    # Build outputs
    reward_chart = build_reward_chart(session["reward_history"])
    grader_chart = build_grader_chart(session["grader_history"])
    breakdown_chart = build_reward_breakdown_chart(reward.components.to_dict())
    history_df = pd.DataFrame(session["episode_log"])

    if done or next_obs is None:
        ticket_html = _render_episode_done_html(session)
        ai_reasoning = "🎉 Episode complete! Review your performance above."
        ai_suggestion = f"Final Score: **{info.get('grader_score', 0):.3f}** | Total Reward: **{info.get('cumulative_reward', 0):.3f}**"
        status = f"🏁 Episode complete | Steps: {session['step_count']} | Cumulative Reward: {info.get('cumulative_reward', 0):.3f}"
        session["done"] = True
        return (
            session, ticket_html, ai_reasoning, ai_suggestion,
            dept_choice, priority_choice, status,
            reward_chart, grader_chart, breakdown_chart, history_df,
            gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
        )

    ticket_html = _render_ticket_html(next_obs)
    ai_dept, ai_priority, ai_reasoning, ai_conf = _mock_ai_suggestion(next_obs, session["task_type"])
    tickets_done = info.get("tickets_processed", 0)
    tickets_total = env._tickets_per_episode
    status = (
        f"✅ Action executed | Step {session['step_count']} | "
        f"Ticket {tickets_done + 1}/{tickets_total} | "
        f"Last reward: {reward.value:+.3f}"
    )

    return (
        session,
        ticket_html,
        ai_reasoning,
        f"🎯 Suggested: **{department_label(ai_dept)}** | Confidence: {ai_conf}",
        ai_dept,
        ai_priority,
        status,
        reward_chart,
        grader_chart,
        breakdown_chart,
        history_df,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def read_ticket(session: Dict) -> Tuple:
    """Execute a READ action."""
    env: Optional[TicketTriageEnv] = session.get("env")
    obs: Optional[Observation] = session.get("current_obs")

    if env is None or obs is None or session.get("done"):
        return _noop_state(session)

    action = Action(action_type=ActionTypeEnum.READ)
    next_obs, reward, done, info = env.step(action)
    session["step_count"] += 1
    session["reward_history"].append(reward.value)

    session["episode_log"].append({
        "Step": session["step_count"],
        "Action": "Read",
        "Department": "—",
        "Priority": "—",
        "Reward": f"{reward.value:+.3f}",
        "Score": "—",
    })

    reward_chart = build_reward_chart(session["reward_history"])
    grader_chart = build_grader_chart(session["grader_history"])
    breakdown_chart = build_reward_breakdown_chart(reward.components.to_dict())
    history_df = pd.DataFrame(session["episode_log"])

    status = f"👁️ Read action | Step {session['step_count']} | Reward: {reward.value:+.3f}"

    return (
        session,
        _render_ticket_html(obs),
        "📖 Ticket details reviewed. AI analysis refreshed.",
        "Reading ticket for deeper context...",
        session.get("dept_suggestion", "general"),
        session.get("priority_suggestion", "medium"),
        status,
        reward_chart,
        grader_chart,
        breakdown_chart,
        history_df,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def analyze_ticket(session: Dict) -> Tuple:
    """Execute an ANALYZE action."""
    env: Optional[TicketTriageEnv] = session.get("env")
    obs: Optional[Observation] = session.get("current_obs")

    if env is None or obs is None or session.get("done"):
        return _noop_state(session)

    action = Action(action_type=ActionTypeEnum.ANALYZE)
    next_obs, reward, done, info = env.step(action)
    session["step_count"] += 1
    session["reward_history"].append(reward.value)

    session["episode_log"].append({
        "Step": session["step_count"],
        "Action": "Analyze",
        "Department": "—",
        "Priority": "—",
        "Reward": f"{reward.value:+.3f}",
        "Score": "—",
    })

    reward_chart = build_reward_chart(session["reward_history"])
    grader_chart = build_grader_chart(session["grader_history"])
    breakdown_chart = build_reward_breakdown_chart(reward.components.to_dict())
    history_df = pd.DataFrame(session["episode_log"])

    # Enhanced AI suggestion after analysis
    ai_dept, ai_priority, ai_reasoning, ai_conf = _mock_ai_suggestion(obs, session["task_type"])
    enhanced_reasoning = (
        f"🔬 **Deep Analysis Complete**\n\n{ai_reasoning}\n\n"
        f"*Confidence boosted by analysis step: {ai_conf}*"
    )
    status = f"🔬 Analysis complete | Step {session['step_count']} | Reward: {reward.value:+.3f}"

    return (
        session,
        _render_ticket_html(obs),
        enhanced_reasoning,
        f"🎯 Suggested: **{department_label(ai_dept)}** | Confidence: {ai_conf} ⬆️",
        ai_dept,
        ai_priority,
        status,
        reward_chart,
        grader_chart,
        breakdown_chart,
        history_df,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------


def _render_ticket_html(obs: Observation) -> str:
    tier_colors = {
        "free": "#64748b",
        "premium": "#7c3aed",
        "enterprise": "#b45309",
    }
    tier_badges = {
        "free": "🆓 Free",
        "premium": "⭐ Premium",
        "enterprise": "🏢 Enterprise",
    }
    tier_val = obs.customer_tier if isinstance(obs.customer_tier, str) else obs.customer_tier.value
    tier_color = tier_colors.get(tier_val, "#64748b")
    tier_badge = tier_badges.get(tier_val, tier_val.title())

    sentiment = obs.sentiment_score
    if sentiment < -0.5:
        sentiment_color = "#ef4444"
        sentiment_label = f"😤 Very Negative ({sentiment:.2f})"
    elif sentiment < 0.0:
        sentiment_color = "#f97316"
        sentiment_label = f"😞 Negative ({sentiment:.2f})"
    elif sentiment < 0.3:
        sentiment_color = "#eab308"
        sentiment_label = f"😐 Neutral ({sentiment:.2f})"
    else:
        sentiment_color = "#22c55e"
        sentiment_label = f"😊 Positive ({sentiment:.2f})"

    wait = obs.wait_time_minutes
    wait_color = "#22c55e" if wait < 30 else ("#eab308" if wait < 90 else "#ef4444")
    wait_label = f"{wait:.1f} min"
    if wait > 60:
        wait_label += " ⚠️ LONG WAIT"

    return f"""
<div style="
  font-family: 'Inter', 'Segoe UI', sans-serif;
  border: 2px solid #e2e8f0;
  border-radius: 16px;
  padding: 20px 24px;
  background: linear-gradient(135deg, #f8fafc 0%, #f0f4ff 100%);
  box-shadow: 0 4px 24px rgba(99,102,241,0.08);
  margin: 4px 0;
">
  <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;">
    <div style="display: flex; align-items: center; gap: 10px;">
      <span style="
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
      ">🎫 {obs.ticket_id}</span>
      <span style="
        background: {tier_color};
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
      ">{tier_badge}</span>
    </div>
    <span style="
      color: {wait_color};
      font-size: 13px;
      font-weight: 600;
      background: {wait_color}18;
      padding: 4px 12px;
      border-radius: 12px;
    ">⏱ {wait_label}</span>
  </div>

  <h3 style="
    margin: 0 0 10px 0;
    font-size: 18px;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.4;
  ">{obs.subject}</h3>

  <p style="
    margin: 0 0 16px 0;
    font-size: 14px;
    color: #475569;
    line-height: 1.7;
    background: white;
    padding: 12px 16px;
    border-radius: 10px;
    border-left: 4px solid #6366f1;
  ">{obs.description}</p>

  <div style="display: flex; gap: 16px; flex-wrap: wrap;">
    <div style="
      background: {sentiment_color}18;
      border: 1px solid {sentiment_color}44;
      padding: 6px 14px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      color: {sentiment_color};
    ">💬 Sentiment: {sentiment_label}</div>
    <div style="
      background: #6366f118;
      border: 1px solid #6366f144;
      padding: 6px 14px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      color: #6366f1;
    ">📋 Remaining: {obs.tickets_remaining} tickets</div>
  </div>
</div>
"""


def _render_episode_done_html(session: Dict) -> str:
    grader_scores = session.get("grader_history", [])
    avg_score = sum(grader_scores) / len(grader_scores) if grader_scores else 0.0
    total_reward = sum(session.get("reward_history", []))
    steps = session.get("step_count", 0)

    if avg_score >= 0.85:
        grade = "🥇 Excellent"
        color = "#22c55e"
    elif avg_score >= 0.70:
        grade = "🥈 Good"
        color = "#3b82f6"
    elif avg_score >= 0.55:
        grade = "🥉 Fair"
        color = "#eab308"
    else:
        grade = "📚 Needs Improvement"
        color = "#ef4444"

    return f"""
<div style="
  font-family: 'Inter', 'Segoe UI', sans-serif;
  text-align: center;
  padding: 32px 24px;
  background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
  border-radius: 16px;
  border: 2px solid #86efac;
  box-shadow: 0 4px 24px rgba(34,197,94,0.12);
">
  <div style="font-size: 48px; margin-bottom: 12px;">🏁</div>
  <h2 style="color: #15803d; font-size: 24px; margin: 0 0 8px 0;">Episode Complete!</h2>
  <p style="color: {color}; font-size: 32px; font-weight: 800; margin: 8px 0;">{grade}</p>
  <div style="display: flex; justify-content: center; gap: 24px; margin-top: 16px; flex-wrap: wrap;">
    <div style="background: white; padding: 12px 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
      <div style="font-size: 24px; font-weight: 800; color: {color};">{avg_score:.1%}</div>
      <div style="font-size: 12px; color: #64748b; font-weight: 600;">Avg Score</div>
    </div>
    <div style="background: white; padding: 12px 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
      <div style="font-size: 24px; font-weight: 800; color: #6366f1;">{total_reward:+.3f}</div>
      <div style="font-size: 12px; color: #64748b; font-weight: 600;">Total Reward</div>
    </div>
    <div style="background: white; padding: 12px 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
      <div style="font-size: 24px; font-weight: 800; color: #f97316;">{steps}</div>
      <div style="font-size: 12px; color: #64748b; font-weight: 600;">Steps Used</div>
    </div>
  </div>
  <p style="color: #64748b; font-size: 13px; margin-top: 20px;">
    Click <strong>🚀 Start New Episode</strong> to play again!
  </p>
</div>
"""


def _noop_state(session: Dict) -> Tuple:
    """Return current state without changes (for disabled actions)."""
    obs = session.get("current_obs")
    ticket_html = _render_ticket_html(obs) if obs else "<p>No active ticket.</p>"
    reward_chart = build_reward_chart(session.get("reward_history", []))
    grader_chart = build_grader_chart(session.get("grader_history", []))
    breakdown_chart = build_reward_breakdown_chart({})
    history_df = pd.DataFrame(session.get("episode_log", []) or [], columns=["Step", "Action", "Department", "Priority", "Reward", "Score"])
    return (
        session, ticket_html, "", "", "general", "medium",
        "⚠️ No active episode. Click 'Start New Episode' first.",
        reward_chart, grader_chart, breakdown_chart, history_df,
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
    )


def _episode_done_state(session: Dict) -> Tuple:
    """Return done state."""
    ticket_html = _render_episode_done_html(session)
    reward_chart = build_reward_chart(session.get("reward_history", []))
    grader_chart = build_grader_chart(session.get("grader_history", []))
    breakdown_chart = build_reward_breakdown_chart({})
    history_df = pd.DataFrame(session.get("episode_log", []) or [], columns=["Step", "Action", "Department", "Priority", "Reward", "Score"])
    return (
        session, ticket_html, "Episode is complete.", "Episode done.",
        "general", "medium", "🏁 Episode complete. Start a new episode!",
        reward_chart, grader_chart, breakdown_chart, history_df,
        gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False),
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }

.gradio-container {
  max-width: 1400px !important;
  margin: 0 auto !important;
}

.hero-header {
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 30%, #4f46e5 60%, #7c3aed 100%);
  color: white;
  padding: 32px 40px;
  border-radius: 20px;
  margin-bottom: 20px;
  box-shadow: 0 8px 32px rgba(79,70,229,0.4);
  text-align: center;
}

.hero-header h1 {
  font-size: 2.4rem;
  font-weight: 900;
  margin: 0 0 8px 0;
  letter-spacing: -0.5px;
}

.hero-header p {
  font-size: 1.1rem;
  opacity: 0.88;
  margin: 0;
  font-weight: 400;
}

.section-card {
  background: white;
  border: 1.5px solid #e2e8f0;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}

.status-bar {
  background: linear-gradient(90deg, #eff6ff, #f0fdf4);
  border: 1px solid #bfdbfe;
  border-radius: 10px;
  padding: 10px 16px;
  font-weight: 600;
  color: #1e40af;
}

button.primary-btn {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  border: none !important;
  color: white !important;
  font-weight: 700 !important;
  border-radius: 12px !important;
  transition: all 0.2s !important;
}

button.primary-btn:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
}

.action-btn-green { background: linear-gradient(135deg, #22c55e, #16a34a) !important; border: none !important; color: white !important; font-weight: 700 !important; border-radius: 12px !important; }
.action-btn-blue { background: linear-gradient(135deg, #3b82f6, #2563eb) !important; border: none !important; color: white !important; font-weight: 700 !important; border-radius: 12px !important; }
.action-btn-purple { background: linear-gradient(135deg, #a855f7, #7c3aed) !important; border: none !important; color: white !important; font-weight: 700 !important; border-radius: 12px !important; }
"""

# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="🎫 OpenEnv Ticket Triage | Meta Hackathon 2026",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # Session state
        session_state = gr.State(_make_default_session())

        # ---------------------------------------------------------------
        # Header
        # ---------------------------------------------------------------
        gr.HTML("""
        <div class="hero-header">
          <h1>🎫 OpenEnv Ticket Triage</h1>
          <p>Meta PyTorch OpenEnv Hackathon 2026 · Interactive Customer Support Triage Environment</p>
          <div style="margin-top: 14px; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.2); padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">🟢 Easy: Classification</span>
            <span style="background: rgba(255,255,255,0.2); padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">🟡 Medium: Priority</span>
            <span style="background: rgba(255,255,255,0.2); padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">🔴 Hard: Efficiency</span>
            <span style="background: rgba(255,255,255,0.2); padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">📊 Real-time Charts</span>
          </div>
        </div>
        """)

        # ---------------------------------------------------------------
        # Controls row
        # ---------------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=2):
                task_radio = gr.Radio(
                    choices=list(TASK_LABEL_MAP.keys()),
                    value="🟢 Classification (Easy)",
                    label="🎯 Select Task",
                    info="Choose difficulty level for this episode",
                )
            with gr.Column(scale=1):
                start_btn = gr.Button(
                    "🚀 Start New Episode",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn"],
                )

        # Status bar
        status_md = gr.Markdown(
            "ℹ️ Select a task above and click **🚀 Start New Episode** to begin.",
            elem_classes=["status-bar"],
        )

        gr.Markdown("---")

        # ---------------------------------------------------------------
        # Main content area
        # ---------------------------------------------------------------
        with gr.Row(equal_height=False):
            # Left panel: Ticket + AI
            with gr.Column(scale=3):
                gr.Markdown("### 📬 Current Ticket")
                ticket_display = gr.HTML(
                    """<div style="
                      text-align: center; padding: 40px;
                      background: #f8fafc; border-radius: 16px;
                      border: 2px dashed #cbd5e1; color: #94a3b8;
                      font-size: 16px; font-weight: 500;
                    ">🎫 No active ticket<br><span style="font-size: 13px;">Start an episode to see tickets</span></div>"""
                )

                with gr.Accordion("🤖 AI Reasoning Panel", open=True):
                    ai_suggestion_label = gr.Markdown(
                        "*AI analysis will appear here after starting an episode*",
                        label="AI Suggestion",
                    )
                    ai_reasoning_md = gr.Markdown(
                        "",
                        label="Chain-of-Thought Reasoning",
                    )

            # Right panel: Actions
            with gr.Column(scale=2):
                gr.Markdown("### ⚡ Take Action")

                with gr.Group():
                    gr.Markdown("**Step 1:** Optionally gather info")
                    with gr.Row():
                        read_btn = gr.Button(
                            "👁️ Read Ticket",
                            variant="secondary",
                            interactive=False,
                            elem_classes=["action-btn-blue"],
                        )
                        analyze_btn = gr.Button(
                            "🔬 Analyze",
                            variant="secondary",
                            interactive=False,
                            elem_classes=["action-btn-purple"],
                        )

                gr.Markdown("---")

                with gr.Group():
                    gr.Markdown("**Step 2:** Execute classification")

                    dept_radio = gr.Radio(
                        choices=[c[0] for c in DEPT_CHOICES],
                        value="💬 General Support",
                        label="🏢 Department",
                    )

                    priority_radio = gr.Radio(
                        choices=[c[0] for c in PRIORITY_CHOICES],
                        value="🟡 Medium",
                        label="⚡ Priority (Task 2 & 3)",
                    )

                    action_type_radio = gr.Radio(
                        choices=["classify", "route", "escalate"],
                        value="classify",
                        label="Action Type",
                        info="Use 'escalate' for critical/overdue tickets",
                    )

                    execute_btn = gr.Button(
                        "✅ Execute Action",
                        variant="primary",
                        size="lg",
                        interactive=False,
                        elem_classes=["primary-btn"],
                    )

        gr.Markdown("---")

        # ---------------------------------------------------------------
        # Visualization row
        # ---------------------------------------------------------------
        gr.Markdown("### 📊 Real-Time Performance Metrics")
        with gr.Row():
            with gr.Column(scale=1):
                reward_chart = gr.Plot(
                    label="Reward Trajectory",
                    value=build_reward_chart([]),
                )
            with gr.Column(scale=1):
                grader_chart = gr.Plot(
                    label="Grader Scores",
                    value=build_grader_chart([]),
                )

        with gr.Row():
            with gr.Column(scale=1):
                breakdown_chart = gr.Plot(
                    label="Reward Breakdown",
                    value=build_reward_breakdown_chart({}),
                )
            with gr.Column(scale=2):
                gr.Markdown("### 📋 Episode History")
                history_table = gr.DataFrame(
                    value=pd.DataFrame(columns=["Step", "Action", "Department", "Priority", "Reward", "Score"]),
                    label="Action Log",
                    interactive=False,
                    wrap=True,
                )

        gr.Markdown("---")

        # ---------------------------------------------------------------
        # Info section
        # ---------------------------------------------------------------
        with gr.Accordion("ℹ️ How to Use This Dashboard", open=False):
            gr.Markdown("""
## 🎮 Quick Start Guide

1. **Select a task** using the radio buttons at the top
2. **Click 🚀 Start New Episode** to load the first ticket
3. **Read the ticket** carefully in the ticket panel
4. Optionally **👁️ Read** or **🔬 Analyze** the ticket for bonus rewards
5. **Select Department** (and Priority for Tasks 2/3)
6. **Click ✅ Execute Action** to classify the ticket
7. Continue until all tickets are processed!

## 🎯 Task Descriptions

| Task | Difficulty | Tickets | Goal |
|------|------------|---------|------|
| Classification | 🟢 Easy | 5 | Route to correct department |
| Priority Classification | 🟡 Medium | 5 | Classify department + priority |
| Efficiency Triage | 🔴 Hard | 10 | Route quickly while maintaining accuracy |

## 🏆 Scoring

- **🟢 Green (≥85%)** - Excellent performance
- **🟡 Yellow (70-84%)** - Good performance
- **🟠 Orange (55-69%)** - Fair performance
- **🔴 Red (<55%)** - Needs improvement

## 💡 Pro Tips

- **Enterprise/Premium customers** get higher reward multipliers
- **Tickets waiting >60 minutes** should be escalated
- **Excessive reading** of the same ticket incurs penalties
- **Fast, accurate routing** maximizes efficiency bonuses
            """)

        # ---------------------------------------------------------------
        # Event wiring
        # ---------------------------------------------------------------

        # Map display choices to enum values
        dept_display_to_value = {c[0]: c[1] for c in DEPT_CHOICES}
        priority_display_to_value = {c[0]: c[1] for c in PRIORITY_CHOICES}

        def _resolve_dept(display: str) -> str:
            return dept_display_to_value.get(display, "general")

        def _resolve_priority(display: str) -> str:
            return priority_display_to_value.get(display, "medium")

        # All outputs list (shared by all handlers)
        all_outputs = [
            session_state, ticket_display, ai_reasoning_md, ai_suggestion_label,
            dept_radio, priority_radio, status_md,
            reward_chart, grader_chart, breakdown_chart, history_table,
            execute_btn, read_btn, analyze_btn,
        ]

        start_btn.click(
            fn=start_episode,
            inputs=[task_radio, session_state],
            outputs=all_outputs,
        )

        def _execute_wrapper(dept_display, priority_display, action_type, session):
            dept_val = _resolve_dept(dept_display)
            priority_val = _resolve_priority(priority_display)
            return execute_action(dept_val, priority_val, action_type, session)

        execute_btn.click(
            fn=_execute_wrapper,
            inputs=[dept_radio, priority_radio, action_type_radio, session_state],
            outputs=all_outputs,
        )

        read_btn.click(
            fn=read_ticket,
            inputs=[session_state],
            outputs=all_outputs,
        )

        analyze_btn.click(
            fn=analyze_ticket,
            inputs=[session_state],
            outputs=all_outputs,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
    )
