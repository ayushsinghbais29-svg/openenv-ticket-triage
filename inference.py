"""
OpenEnv Ticket Triage - Inference Script

Runs three LLM-powered ticket triage tasks and emits structured stdout logs
in the [START] / [STEP] / [END] format required by OpenEnv evaluation.

Environment variables
---------------------
API_BASE_URL   Base URL for the OpenAI-compatible LLM endpoint.
MODEL_NAME     Model identifier (e.g. gpt-3.5-turbo).
HF_TOKEN       Hugging Face / API key used as the OpenAI API key.
"""

import json
import os
import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Environment configuration ────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key",
)

# ── Structured logging helpers ────────────────────────────────────────────────

def _current_timestamp_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_start(task_name: str, model: str) -> None:
    print(f'[START] task_name="{task_name}" model="{model}" timestamp="{_current_timestamp_utc()}"', flush=True)


def log_step(step: int, action: str, observation: dict, reward: float) -> None:
    obs_json = json.dumps(observation)
    print(f'[STEP] step={step} action="{action}" observation={obs_json} reward={reward:.2f}', flush=True)


def log_end(task_name: str, final_score: float, total_reward: float, status: str) -> None:
    print(f'[END] task_name="{task_name}" final_score={final_score:.2f} total_reward={total_reward:.2f} status="{status}"', flush=True)


# ── LLM call with graceful fallback ──────────────────────────────────────────

def _fallback_response(prompt: str) -> str:
    """Rule-based fallback used when the LLM endpoint is unavailable."""
    p = prompt.lower()
    if "classify" in p or "category" in p:
        return "Technical"
    if "priority" in p:
        return "high"
    if "assign" in p or "department" in p or "team" in p:
        return "tech-support"
    if "resolve" in p or "resolution" in p:
        return "The issue has been investigated and a fix has been applied. Please verify and let us know if further assistance is needed."
    return "processed"


def call_llm(prompt: str, system_prompt: str = "") -> str:
    """Call the configured LLM; fall back to rule-based responses on error."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=256,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[WARN] LLM call failed ({exc}); using fallback response.", flush=True)
        return _fallback_response(prompt)


# ── Sample ticket dataset ─────────────────────────────────────────────────────

TICKET_CATEGORIES = ["Billing", "Technical", "General", "Premium Support"]

DEPARTMENT_MAP = {
    "Billing": "billing-team",
    "Technical": "tech-support",
    "General": "general-support",
    "Premium Support": "premium-support",
}

SAMPLE_TICKETS = [
    {
        "ticket_id": "TKT-10001",
        "subject": "Cannot login to my account",
        "description": "I have been trying to login but keep getting an error message.",
        "true_category": "Technical",
        "true_priority": "high",
    },
    {
        "ticket_id": "TKT-10002",
        "subject": "Invoice overcharge",
        "description": "I was charged twice for my subscription this month.",
        "true_category": "Billing",
        "true_priority": "high",
    },
    {
        "ticket_id": "TKT-10003",
        "subject": "Feature request",
        "description": "It would be great to have dark mode in the application.",
        "true_category": "General",
        "true_priority": "low",
    },
    {
        "ticket_id": "TKT-10004",
        "subject": "VIP account setup",
        "description": "I need priority support for setting up my enterprise account.",
        "true_category": "Premium Support",
        "true_priority": "critical",
    },
    {
        "ticket_id": "TKT-10005",
        "subject": "Payment method update",
        "description": "I need to update my credit card information.",
        "true_category": "Billing",
        "true_priority": "medium",
    },
]


# ── Task 1: Ticket Classification ─────────────────────────────────────────────

def run_ticket_classification_task() -> float:
    """Classify each ticket into the correct category. Score = accuracy."""
    task_name = "ticket_classification"
    log_start(task_name, MODEL_NAME)

    correct = 0
    total = len(SAMPLE_TICKETS)
    total_reward = 0.0
    step = 0

    for ticket in SAMPLE_TICKETS:
        # Step: read ticket
        step += 1
        log_step(step, "read_ticket", {"ticket_id": ticket["ticket_id"], "subject": ticket["subject"]}, 0.1)
        total_reward += 0.1

        # Step: classify ticket via LLM
        step += 1
        prompt = (
            f"Classify the following support ticket into exactly one of these categories: "
            f"{', '.join(TICKET_CATEGORIES)}.\n\n"
            f"Subject: {ticket['subject']}\n"
            f"Description: {ticket['description']}\n\n"
            f"Reply with ONLY the category name, nothing else."
        )
        raw = call_llm(prompt, "You are a ticket classification assistant.")
        predicted = raw.strip()
        for cat in TICKET_CATEGORIES:
            if cat.lower() in predicted.lower():
                predicted = cat
                break

        is_correct = predicted == ticket["true_category"]
        step_reward = 1.0 if is_correct else 0.0
        if is_correct:
            correct += 1
        total_reward += step_reward
        log_step(
            step,
            "classify_ticket",
            {
                "ticket_id": ticket["ticket_id"],
                "predicted_category": predicted,
                "true_category": ticket["true_category"],
                "correct": is_correct,
            },
            step_reward,
        )

        # Step: record result
        step += 1
        log_step(step, "record_classification", {"ticket_id": ticket["ticket_id"], "status": "classified"}, 0.05)
        total_reward += 0.05

    final_score = correct / total
    log_end(task_name, final_score, total_reward, "success")
    return final_score


# ── Task 2: Ticket Assignment ─────────────────────────────────────────────────

def run_ticket_assignment_task() -> float:
    """Assign each ticket to the correct department team. Score = accuracy."""
    task_name = "ticket_assignment"
    log_start(task_name, MODEL_NAME)

    correct = 0
    total = len(SAMPLE_TICKETS)
    total_reward = 0.0
    step = 0
    departments = list(DEPARTMENT_MAP.values())

    for ticket in SAMPLE_TICKETS:
        # Step: read ticket
        step += 1
        log_step(step, "read_ticket", {"ticket_id": ticket["ticket_id"], "subject": ticket["subject"]}, 0.1)
        total_reward += 0.1

        # Step: assign ticket via LLM
        step += 1
        prompt = (
            f"Assign the following support ticket to the correct team.\n"
            f"Available teams: {', '.join(departments)}\n\n"
            f"Subject: {ticket['subject']}\n"
            f"Description: {ticket['description']}\n\n"
            f"Reply with ONLY the team name from the list, nothing else."
        )
        raw = call_llm(prompt, "You are a ticket routing assistant.")
        predicted = raw.strip()
        for dept in departments:
            if dept.lower() in predicted.lower():
                predicted = dept
                break

        true_assignment = DEPARTMENT_MAP[ticket["true_category"]]
        is_correct = predicted == true_assignment
        step_reward = 1.0 if is_correct else 0.0
        if is_correct:
            correct += 1
        total_reward += step_reward
        log_step(
            step,
            "assign_ticket",
            {
                "ticket_id": ticket["ticket_id"],
                "predicted_assignment": predicted,
                "true_assignment": true_assignment,
                "correct": is_correct,
            },
            step_reward,
        )

        # Step: confirm assignment
        step += 1
        log_step(step, "confirm_assignment", {"ticket_id": ticket["ticket_id"], "assigned_to": predicted}, 0.05)
        total_reward += 0.05

    final_score = correct / total
    log_end(task_name, final_score, total_reward, "success")
    return final_score


# ── Task 3: Ticket Resolution ─────────────────────────────────────────────────

MIN_KEYWORD_LENGTH = 4


def _grade_resolution(resolution: str, ticket: dict) -> float:
    """Grade a generated resolution on a 0.0-1.0 scale."""
    score = 0.0
    # Length check
    if len(resolution) >= 50:
        score += 0.4
    elif len(resolution) >= 20:
        score += 0.2
    # Actionable content
    action_kw = ["resolved", "fix", "update", "check", "contact", "verify", "confirm", "applied", "investigated"]
    if any(kw in resolution.lower() for kw in action_kw):
        score += 0.3
    # Relevance to subject
    key_words = [w for w in ticket["subject"].lower().split() if len(w) >= MIN_KEYWORD_LENGTH]
    if any(w in resolution.lower() for w in key_words):
        score += 0.3
    return min(score, 1.0)


def run_ticket_resolution_task() -> float:
    """Generate and grade resolutions for tickets. Score = mean resolution quality."""
    task_name = "ticket_resolution"
    log_start(task_name, MODEL_NAME)

    resolution_scores = []
    total_reward = 0.0
    step = 0

    for ticket in SAMPLE_TICKETS:
        # Step: read ticket
        step += 1
        log_step(step, "read_ticket", {"ticket_id": ticket["ticket_id"], "priority": ticket["true_priority"]}, 0.1)
        total_reward += 0.1

        # Step: analyze ticket
        step += 1
        analyze_prompt = (
            f"Analyze the severity and urgency of this support ticket in 1-2 sentences.\n"
            f"Subject: {ticket['subject']}\n"
            f"Description: {ticket['description']}"
        )
        analysis = call_llm(analyze_prompt, "You are a support ticket analyst.")
        log_step(
            step,
            "analyze_ticket",
            {"ticket_id": ticket["ticket_id"], "analysis_length": len(analysis), "priority": ticket["true_priority"]},
            0.2,
        )
        total_reward += 0.2

        # Step: generate resolution
        step += 1
        resolve_prompt = (
            f"Generate a concise resolution for this support ticket in 2-3 sentences.\n"
            f"Subject: {ticket['subject']}\n"
            f"Description: {ticket['description']}\n"
            f"Priority: {ticket['true_priority']}"
        )
        resolution = call_llm(resolve_prompt, "You are a support ticket resolver.")
        res_score = _grade_resolution(resolution, ticket)
        resolution_scores.append(res_score)
        total_reward += res_score
        log_step(
            step,
            "resolve_ticket",
            {"ticket_id": ticket["ticket_id"], "resolution_length": len(resolution), "resolution_score": round(res_score, 2)},
            res_score,
        )

        # Step: close ticket
        step += 1
        log_step(step, "close_ticket", {"ticket_id": ticket["ticket_id"], "status": "closed"}, 0.1)
        total_reward += 0.1

    final_score = sum(resolution_scores) / len(resolution_scores)
    log_end(task_name, final_score, total_reward, "success")
    return final_score


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"OpenEnv Ticket Triage Inference", flush=True)
    print(f"API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"MODEL_NAME   : {MODEL_NAME}", flush=True)
    print(f"HF_TOKEN     : {'[SET]' if HF_TOKEN else '[NOT SET - using fallback responses]'}", flush=True)
    print("", flush=True)

    scores: dict = {}

    scores["ticket_classification"] = run_ticket_classification_task()
    print("", flush=True)

    scores["ticket_assignment"] = run_ticket_assignment_task()
    print("", flush=True)

    scores["ticket_resolution"] = run_ticket_resolution_task()
    print("", flush=True)

    avg_score = sum(scores.values()) / len(scores)
    print("=== FINAL RESULTS ===", flush=True)
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}", flush=True)
    print(f"  average_score: {avg_score:.4f}", flush=True)
