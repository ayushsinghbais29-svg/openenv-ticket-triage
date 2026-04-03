# 🎫 OpenEnv Ticket Triage

**Meta PyTorch OpenEnv Hackathon 2026 Submission**

[![CI Validate](https://github.com/ayushsinghbais29-svg/openenv-ticket-triage/actions/workflows/validate.yml/badge.svg)](https://github.com/ayushsinghbais29-svg/openenv-ticket-triage/actions/workflows/validate.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade OpenEnv environment for AI-powered customer support ticket triage. Three graduated difficulty tasks test an agent's ability to classify support tickets by department and priority while optimizing for speed and accuracy.

---

## 🌟 Overview

Customer support ticket triage is a **real-world, high-value problem** used at scale by Stripe, Zendesk, Intercom, GitHub, and every major SaaS company. This environment simulates the full triage pipeline:

1. An agent receives a ticket with subject, description, customer tier, and sentiment
2. The agent can read, analyze, classify, route, or escalate tickets
3. A deterministic grader scores each action on accuracy and efficiency
4. Dense reward signals guide learning throughout the trajectory

---

## 🎯 Three Graduated Tasks

| Task | Difficulty | Tickets/Episode | Max Steps | Target Score |
|------|-----------|----------------|-----------|--------------|
| `classification` | 🟢 Easy | 5 | 15 | 85% |
| `priority_classification` | 🟡 Medium | 5 | 20 | 75% |
| `efficiency` | 🔴 Hard | 10 | 40 | 65% |

### Task 1: Department Classification (Easy 🟢)
Route each ticket to the correct department: `billing`, `technical`, `general`, or `premium_support`.

### Task 2: Priority + Department Classification (Medium 🟡)
Classify both **department** AND **priority** (`low`, `medium`, `high`, `critical`). Enterprise customers have higher weight in the F1-style scorer.

### Task 3: Efficiency Triage (Hard 🔴)
Route a 10-ticket stream optimizing **quality × speed**. Tickets accumulate wait times; slow routing incurs penalties while fast routing earns bonuses.

---

## 📦 Project Structure

```
openenv-ticket-triage/
├── src/                          # Core environment
│   ├── __init__.py
│   ├── models.py                 # Pydantic models (Observation, Action, Reward, ...)
│   ├── environment.py            # TicketTriageEnv with step/reset/state API
│   ├── graders.py                # ClassificationGrader, PriorityClassificationGrader, EfficiencyGrader
│   ├── reward_functions.py       # RewardCalculator, RewardModulator
│   ├── tasks.py                  # TicketGenerator with realistic templates
│   └── utils.py                  # Helper functions
│
├── baseline/                     # GPT-4 baseline agent
│   ├── __init__.py
│   ├── agent.py                  # GPT4Agent with chain-of-thought reasoning
│   ├── evaluate.py               # Full evaluation harness
│   ├── prompts.py                # Task-specific prompts
│   └── results.json              # Expected baseline scores
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_models.py            # Pydantic model validation tests
│   ├── test_environment.py       # Environment reset/step/state tests
│   ├── test_graders.py           # Grader determinism and score range tests
│   └── test_rewards.py           # Reward calculation tests
│
├── deployment/                   # Deployment configuration
│   ├── __init__.py
│   ├── app_gradio.py             # 450+ line Gradio dashboard UI
│   ├── requirements_ui.txt       # UI dependencies
│   └── requirements.txt          # Legacy FastAPI dependencies
│
├── .github/workflows/
│   └── validate.yml              # CI/CD pipeline
│
├── Dockerfile                    # Multi-stage Gradio build
├── docker-compose.yml            # Local development
├── openenv.yaml                  # OpenEnv specification
├── requirements.txt              # Root dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/ayushsinghbais29-svg/openenv-ticket-triage.git
cd openenv-ticket-triage

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run unit tests
python -m pytest tests/ -v

# Launch the Gradio dashboard (requires UI deps)
pip install -r deployment/requirements_ui.txt
python deployment/app_gradio.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose (recommended)
docker compose up --build

# Or build directly
docker build -t openenv-ticket-triage:latest .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... openenv-ticket-triage:latest
```

Open [http://localhost:7860](http://localhost:7860) to view the dashboard.

### Hugging Face Spaces

```bash
# Create HF Space (Docker SDK)
huggingface-cli repo create openenv-ticket-triage --type=space --space-sdk=docker

# Push code
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/openenv-ticket-triage
git push hf main

# Add secret in HF Space settings: OPENAI_API_KEY
```

---

## 🧠 Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | `str` | Unique ticket identifier |
| `subject` | `str` | Ticket subject/title |
| `description` | `str` | Full ticket body |
| `customer_tier` | `CustomerTierEnum` | `free`, `premium`, `enterprise` |
| `sentiment_score` | `float [-1.0, 1.0]` | -1=very negative, +1=very positive |
| `wait_time_minutes` | `float ≥ 0` | Time ticket has been waiting |
| `task_type` | `TaskTypeEnum` | Current active task |
| `step_number` | `int` | Steps used in current episode |
| `tickets_remaining` | `int` | Tickets left in episode |

---

## ⚡ Action Space

| Action Type | Description | Terminal? |
|-------------|-------------|-----------|
| `read` | Read the ticket for +0.05 reward | No |
| `analyze` | Analyze ticket for +0.10 reward | No |
| `classify` | Classify department (+ priority for task 2) | **Yes** |
| `route` | Route to department | **Yes** |
| `escalate` | Escalate critical ticket | **Yes** |
| `close` | Close without routing | Yes (penalty) |

---

## 🏆 Reward Function

Dense multi-component rewards guide agent learning:

```
reward = correctness + efficiency + progress + penalties
```

| Component | Range | Description |
|-----------|-------|-------------|
| `correctness` | 0.0–1.0 | Grader score × tier multiplier |
| `efficiency` | -0.10–+0.15 | Speed bonus/penalty |
| `progress` | 0.0–0.10 | Intermediate action rewards |
| `penalties` | -1.0–0.0 | Repetition and timeout penalties |

**Tier multipliers:** Free=1.0×, Premium=1.1×, Enterprise=1.25×

---

## 📊 Baseline Results

Reproducible with `seed=42`, 5 episodes per task:

| Task | Mean Score | Mean Reward |
|------|-----------|-------------|
| Classification | **85.1%** | 1.234 |
| Priority Classification | **74.8%** | 0.987 |
| Efficiency Triage | **64.8%** | 0.756 |

```bash
# Reproduce baseline (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python baseline/evaluate.py
```

---

## 🎨 Gradio Dashboard Features

The `deployment/app_gradio.py` provides a beautiful interactive dashboard:

- **Gradient header** with professional hackathon branding
- **Task selection** with difficulty labels (Easy/Medium/Hard)
- **Real-time ticket display** with color-coded sentiment and tier badges
- **AI suggestion panel** with chain-of-thought reasoning
- **Action controls**: Read, Analyze, Classify with department/priority selectors
- **Live Plotly charts**: Reward trajectory + grader score bar chart
- **Reward breakdown** horizontal bar chart
- **Episode history table** with full action log
- **Professional CSS** with Inter font and gradient design system
- **Responsive layout** with 1400px max-width

---

## 🔧 Environment API

```python
from src.environment import TicketTriageEnv
from src.models import Action, ActionTypeEnum, DepartmentEnum, PriorityEnum

# Create environment
env = TicketTriageEnv(task_type="priority_classification", seed=42)

# Reset to start episode
obs = env.reset()
print(f"Ticket: {obs.subject}")

# Take actions
done = False
while not done:
    action = Action(
        action_type=ActionTypeEnum.CLASSIFY,
        department=DepartmentEnum.BILLING,
        priority=PriorityEnum.HIGH,
        confidence=0.9,
    )
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward.value:.3f} | Score: {reward.grader_score:.3f}")

# Get full state
state = env.state()
print(f"Cumulative reward: {state.cumulative_reward:.3f}")
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_graders.py -v
```

---

## 📋 OpenEnv Specification

See [`openenv.yaml`](openenv.yaml) for the complete specification including:
- Observation and action space definitions
- Task definitions with grader mappings
- Reward function specification
- Deployment configuration

---

## 🙏 Acknowledgments

Built for the **Meta PyTorch OpenEnv Hackathon 2026**. Inspired by real-world customer support triage systems at scale SaaS companies.

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
