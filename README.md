# OpenEnv Ticket Triage

> **Customer Support Ticket Routing Environment for the Meta PyTorch OpenEnv Hackathon 2026**

[![Validate](https://github.com/ayushsinghbais29-svg/openenv-ticket-triage/actions/workflows/validate.yml/badge.svg)](https://github.com/ayushsinghbais29-svg/openenv-ticket-triage/actions/workflows/validate.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-compliant-green.svg)](https://openenv.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**OpenEnv Ticket Triage** is a production-grade reinforcement learning environment that simulates the customer support ticket routing workflow used at modern SaaS companies like Stripe, Zendesk, and Slack. Every hour, thousands of tickets pour in — billing disputes, API outages, feature questions, enterprise escalations. Getting them to the right team fast is mission-critical.

This environment tasks an agent with reading, analyzing, and routing customer support tickets to one of four departments:

| Department | Handles |
|---|---|
| **Billing** | Charges, invoices, refunds, payment issues |
| **Technical** | API errors, integration bugs, performance issues |
| **General** | How-to questions, feature requests, account settings |
| **Premium Support** | Enterprise SLAs, dedicated support, compliance |

The environment features **three graduated difficulty levels** (easy → hard), a **rich dense reward function** that rewards good reasoning and penalizes lazy behavior, and a **reproducible GPT-4 baseline** that achieves 65–85% scores across tasks.

---

## Motivation

Customer support triage is:

1. **Universally applicable** — Every software company with customers does this. It is not a toy problem.
2. **Measurably improvable** — Wrong routing means unhappy customers, delayed resolutions, and higher support costs. A 10% improvement in routing accuracy has real business value.
3. **Challenging for LLMs** — It requires reading comprehension, entity recognition, domain knowledge, and multi-step reasoning, all under time/step constraints.
4. **Rich in signal** — Unlike binary classification tasks, triage involves confidence calibration, priority ordering, and efficiency tradeoffs that make for a compelling RL training signal.

This environment fills a real gap: there are no existing OpenEnv environments that model enterprise-grade NLP workflows with real-time efficiency constraints.

---

## Architecture

```
openenv-ticket-triage/
├── src/
│   ├── __init__.py           # Package exports
│   ├── environment.py        # TicketTriageEnv: reset/step/state
│   ├── models.py             # Pydantic Observation, Action, Reward
│   ├── graders.py            # 3 task graders with deterministic scoring
│   ├── reward_functions.py   # Dense multi-component reward calculator
│   ├── tasks.py              # Ticket generation from realistic templates
│   └── utils.py              # Validation, config, helpers
│
├── baseline/
│   ├── agent.py              # GPT-4o-mini agent with few-shot prompting
│   ├── evaluate.py           # Full evaluation harness
│   ├── prompts.py            # Task-specific chain-of-thought prompts
│   └── results.json          # Expected baseline scores
│
├── tests/
│   ├── test_environment.py   # reset/step/state contract tests
│   ├── test_graders.py       # Grader correctness, determinism, variance
│   ├── test_models.py        # Pydantic model validation
│   └── test_rewards.py       # Reward component calculation
│
├── deployment/
│   ├── Dockerfile            # Multi-stage optimized build
│   ├── docker-compose.yml    # Local development
│   ├── app.py                # FastAPI Hugging Face Spaces server
│   └── requirements.txt      # Deployment dependencies
│
├── Dockerfile                # Root-level (for HF Spaces)
├── openenv.yaml              # Complete OpenEnv specification
├── requirements.txt          # Development dependencies
└── README.md                 # This file
```

---

## Observation Space

The observation returned by `reset()` and `step()` is a dictionary:

```python
{
    # Current ticket
    "ticket_id": "TKT-0001",              # Unique identifier
    "subject": "API returning 500 errors", # One-line summary
    "description": "Our integration ...",  # Full ticket body
    "sentiment": -0.8,                     # Float [-1.0, 1.0]
    "customer_tier": "enterprise",         # "free" | "premium" | "enterprise"
    "wait_time_seconds": 245,              # How long the ticket has waited

    # Episode metadata
    "task_type": "classification",         # Current task type
    "step": 3,                             # Current step number
    "max_steps": 15,                       # Episode step limit
    "read_count": 1,                       # Times this ticket was read
    "tickets_remaining": 3,                # Tickets still to process
    "tickets_processed": 2,                # Tickets already routed
    "current_score": 0.15,                 # Cumulative reward so far
    "available_actions": ["read", "analyze", "classify", "route"],
    "message": ""                          # Optional status message
}
```

### Sentiment Interpretation

| Range | Meaning |
|---|---|
| -1.0 to -0.5 | Very negative / urgent / frustrated customer |
| -0.5 to 0.0 | Negative / concerned |
| 0.0 | Neutral |
| 0.0 to 0.5 | Positive / satisfied |
| 0.5 to 1.0 | Very positive / happy |

### Customer Tier Distribution

The environment generates tickets with realistic tier distributions:

| Tier | Frequency | Department Bias |
|---|---|---|
| `free` | 50% | General (45%), Technical (30%), Billing (25%) |
| `premium` | 35% | Technical (35%), Billing (30%), General (25%), Premium (10%) |
| `enterprise` | 15% | Premium Support (45%), Technical (25%), Billing (20%) |

---

## Action Space

Actions are passed as dictionaries to `step()`:

```python
{
    "action_type": "route",       # Required. See below.
    "department": "Technical",    # For classify/route actions
    "priority": "High",           # For set_priority actions
    "confidence": 0.9,            # Float [0.0, 1.0], default 1.0
    "reasoning": "API error..."   # Optional chain-of-thought
}
```

### Action Types

| Action | Description | Reward |
|---|---|---|
| `read` | Get the full ticket content | +0.1 (dense) |
| `analyze` | Reason about ticket context | +0.2 (dense) |
| `classify` | Assign a department (does not advance) | Correctness bonus |
| `set_priority` | Set priority level | Correctness bonus |
| `route` | Route to department and **advance to next ticket** | Correctness bonus |

> **Important**: Only `route` advances the episode. You must route all tickets to complete an episode.

### Departments

- `Billing` — Payment issues, invoices, charges
- `Technical` — API errors, bugs, integrations
- `General` — How-to questions, account help
- `Premium Support` — Enterprise SLAs, dedicated support

### Priority Levels

- `Low` — Non-urgent, informational
- `Medium` — Workflow impact, non-critical
- `High` — Service degradation, billing errors
- `Critical` — Complete outage, SLA violation, data loss

---

## Reward Function

The reward function provides dense signal throughout the episode, not just at the end.

### Components

```python
reward = {
    "value": 0.72,               # Total scalar reward [-2.0, 2.0]
    "components": {
        "correctness": 0.72,     # From correct routing/classification
        "efficiency": -0.03,     # Penalty for excessive steps
        "progress": 0.10,        # From read/analyze actions
        "penalties": 0.0         # Repetition, timeout penalties
    },
    "message": "+0.1 for read; +0.72 correctness"
}
```

### Reward Shaping Details

| Signal | Amount | Condition |
|---|---|---|
| Read action | +0.10 | Every `read` action |
| Analyze action | +0.20 | Every `analyze` action |
| Correct department | +0.40–0.80 | `route`/`classify` to correct dept, scaled by confidence |
| Correct priority | +0.40 | Exact match |
| Adjacent priority | +0.20 | Off by one level (e.g., High vs Critical) |
| Efficiency penalty | -0.05×ratio | After 60% of max steps used |
| Repetition penalty | -0.20 | Reading same ticket more than 2 times |
| Timeout penalty | -1.00 | Episode truncated at max steps |

### Example Trajectory (Easy Task)

```
Step 1: read   → reward=+0.10  (dense progress)
Step 2: analyze → reward=+0.20 (dense progress)
Step 3: route(department="Billing", confidence=0.9) → reward=+0.72 (correct)

Step 4: read → reward=+0.10
Step 5: route(department="Technical", confidence=0.85) → reward=+0.68

Step 6: route(department="General", confidence=0.9)   → reward=+0.72
Step 7: route(department="Billing", confidence=0.6)   → reward=+0.48
Step 8: route(department="Premium Support", confidence=0.95) → reward=+0.76

Episode Score: 0.87
```

---

## Tasks

### Task 1: Department Classification (Easy)

**Objective**: Route each of 5 tickets to the correct department.

```yaml
task_type: classification
n_tickets: 5
max_steps: 15
grader: ClassificationGrader
metric: accuracy
target_baseline: 0.85
available_actions: [read, analyze, classify, route]
```

**Grader Logic**: Binary correctness with confidence modulation.
Correct routing scores `0.5 + 0.5 × confidence`. Wrong routing scores 0.

**Strategy**: Read the ticket subject, identify billing/technical/general/premium signals, route with high confidence.

---

### Task 2: Priority + Classification (Medium)

**Objective**: Classify department AND determine priority level for each of 5 tickets.

```yaml
task_type: priority_classification
n_tickets: 5
max_steps: 20
grader: PriorityClassificationGrader
metric: f1_score
target_baseline: 0.75
available_actions: [read, analyze, classify, set_priority, route]
```

**Grader Logic**: F1-like scoring combining department accuracy (70% weight) and priority accuracy (30% weight).
Adjacent priority mismatches receive partial credit. Enterprise tickets are weighted 1.5×, premium 1.2×.

**Priority signals**:
- Enterprise + outage = Critical
- Payment errors = High
- How-to questions = Low
- Performance degradation = Medium

---

### Task 3: Efficiency Triage (Hard)

**Objective**: Route a stream of 10 tickets while balancing quality against speed.

```yaml
task_type: efficiency_triage
n_tickets: 10
max_steps: 30
grader: EfficiencyGrader
metric: composite_score
target_baseline: 0.65
available_actions: [read, analyze, classify, set_priority, route]
```

**Grader Logic**: Composite score = 0.50 × quality + 0.35 × efficiency + 0.15 × escalation.
- **Quality**: Fraction of tickets correctly routed
- **Efficiency**: Optimal is 2 steps/ticket; penalized beyond that
- **Escalation bonus**: Extra credit for correctly handling Critical/enterprise tickets

**Strategy**: Quick, decisive routing. Skip `analyze` on clear-signal tickets.

---

## Setup

### Prerequisites

- Python 3.11+

### Local Installation

```bash
git clone https://github.com/ayushsinghbais29-svg/openenv-ticket-triage
cd openenv-ticket-triage
pip install -r requirements.txt
```

### Quick Start

```python
from src.environment import TicketTriageEnv

env = TicketTriageEnv(task_type="classification", seed=42)
obs = env.reset()
print(f"Ticket: {obs['subject']}")

obs, reward, done, truncated, info = env.step({
    "action_type": "route",
    "department": "Technical",
    "confidence": 0.9
})

if done:
    print(f"Episode score: {info['episode_score']:.3f}")
```

### Running Tests

```bash
python -m pytest tests/ -v
```

All 81 tests pass in under 1 second.

---

## Baseline Agent

The baseline uses GPT-4o-mini with few-shot chain-of-thought prompting.
It reads `OPENAI_API_KEY` from environment variables and falls back to
a heuristic rule-based system when no key is available.

### Running the Baseline

```bash
export OPENAI_API_KEY=your_key_here
python baseline/evaluate.py --n-episodes 5 --seed 42
```

### Expected Scores

| Task | Metric | Target | GPT-4o-mini |
|---|---|---|---|
| Classification (Easy) | Accuracy | 85% | ~85% |
| Priority+Classification (Medium) | F1 Score | 75% | ~75% |
| Efficiency Triage (Hard) | Composite | 65% | ~65% |

---

## Docker

```bash
docker build -t openenv-ticket-triage .
docker run -p 7860:7860 openenv-ticket-triage
curl http://localhost:7860/health
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/` | Environment info |
| POST | `/reset/{task_type}` | Reset environment |
| POST | `/step/{task_type}` | Execute action |
| GET | `/state/{task_type}` | Get full state |
| GET | `/docs` | Swagger UI |

---

## Hugging Face Spaces

Deploy to HF Spaces with the `openenv` tag:

1. Create a Space with **Docker** SDK
2. Tag with `openenv`
3. Push this repository

The server starts on port 7860 automatically.

---

## Troubleshooting

**"Call reset() before step()"** — Call `env.reset()` before `env.step()`.

**Invalid department name** — Department names are case-sensitive:
- ✅ `"Billing"`, `"Technical"`, `"General"`, `"Premium Support"`
- ❌ `"billing"`, `"tech"`, `"premium"`

**No OpenAI API key** — The baseline uses heuristic fallback automatically when `OPENAI_API_KEY` is not set.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
