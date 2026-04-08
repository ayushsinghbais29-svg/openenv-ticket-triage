import copy
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

REWARD_PRECISION = 4

app = FastAPI(title="OpenEnv Ticket Triage API")

TICKETS = [
    {
        "id": "TKT-001",
        "title": "Payment not processed",
        "description": "Customer unable to complete payment for order #12345",
        "ticket_status": "open",
        "ticket_priority": "high",
        "assignee": None,
        "department": "Billing",
    },
    {
        "id": "TKT-002",
        "title": "Application crashes on login",
        "description": "App throws 500 error when user tries to log in",
        "ticket_status": "open",
        "ticket_priority": "critical",
        "assignee": None,
        "department": "Technical",
    },
    {
        "id": "TKT-003",
        "title": "Update account email",
        "description": "User wants to change their registered email address",
        "ticket_status": "open",
        "ticket_priority": "low",
        "assignee": None,
        "department": "General",
    },
    {
        "id": "TKT-004",
        "title": "Refund request for duplicate charge",
        "description": "Customer was charged twice for the same subscription",
        "ticket_status": "open",
        "ticket_priority": "medium",
        "assignee": None,
        "department": "Billing",
    },
    {
        "id": "TKT-005",
        "title": "API integration not working",
        "description": "Third-party API returns 403 Forbidden after credentials update",
        "ticket_status": "open",
        "ticket_priority": "high",
        "assignee": None,
        "department": "Technical",
    },
]

AGENTS = ["alice@support.com", "bob@support.com", "carol@support.com"]

_state: dict = {}


def _initial_state() -> dict:
    ticket = copy.deepcopy(random.choice(TICKETS))
    return {
        "ticket": ticket,
        "step_count": 0,
        "done": False,
        "total_reward": 0.0,
    }


class ActionRequest(BaseModel):
    action: str
    assignee: str | None = None
    comment: str | None = None


@app.post("/openenv/reset")
def reset():
    """Reset the environment and return the initial observation."""
    global _state
    _state = _initial_state()
    ticket = _state["ticket"]
    return {
        "observation": {
            "ticket_id": ticket["id"],
            "title": ticket["title"],
            "description": ticket["description"],
            "ticket_status": ticket["ticket_status"],
            "ticket_priority": ticket["ticket_priority"],
            "assignee": ticket["assignee"],
        },
        "reward": 0.0,
        "done": False,
        "info": {"step_count": 0},
    }


@app.get("/openenv/step")
def step(action: str, assignee: str | None = None, comment: str | None = None):
    """Step through the environment by taking an action."""
    global _state

    if not _state:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /openenv/reset first.")

    ticket = _state["ticket"]
    reward = 0.0
    done = False
    info: dict = {}

    if action == "assign_ticket":
        chosen_assignee = assignee or random.choice(AGENTS)
        ticket["assignee"] = chosen_assignee
        ticket["ticket_status"] = "assigned"
        reward = 0.5
        info["message"] = f"Ticket assigned to {chosen_assignee}"

    elif action == "close_ticket":
        ticket["ticket_status"] = "closed"
        reward = 1.0 if ticket["assignee"] else 0.3
        done = True
        info["message"] = "Ticket closed"

    elif action == "add_comment":
        reward = 0.2
        info["message"] = f"Comment added: {comment or '(no comment)'}"

    else:
        reward = -0.1
        info["message"] = f"Unknown action: {action}"

    _state["step_count"] += 1
    _state["done"] = done
    _state["total_reward"] = round(_state["total_reward"] + reward, REWARD_PRECISION)

    return {
        "observation": {
            "ticket_id": ticket["id"],
            "title": ticket["title"],
            "description": ticket["description"],
            "ticket_status": ticket["ticket_status"],
            "ticket_priority": ticket["ticket_priority"],
            "assignee": ticket["assignee"],
        },
        "reward": reward,
        "done": done,
        "info": {
            "step_count": _state["step_count"],
            "total_reward": _state["total_reward"],
            **info,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
