"""FastAPI application for Hugging Face Spaces deployment."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.environment import TicketTriageEnv

app = FastAPI(
    title="OpenEnv Ticket Triage",
    description="Customer Support Ticket Triage - Meta OpenEnv Hackathon 2026",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ENVIRONMENTS: Dict[str, TicketTriageEnv] = {
    "classification": TicketTriageEnv(task_type="classification", seed=42),
    "priority_classification": TicketTriageEnv(task_type="priority_classification", seed=42),
    "efficiency_triage": TicketTriageEnv(task_type="efficiency_triage", seed=42),
}


class ActionRequest(BaseModel):
    action_type: str
    department: Optional[str] = None
    priority: Optional[str] = None
    confidence: float = 1.0
    reasoning: Optional[str] = None


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "environment": "openenv-ticket-triage",
        "version": "1.0.0",
        "tasks": list(ENVIRONMENTS.keys()),
    }


@app.post("/reset/{task_type}")
def reset_environment(task_type: str) -> Dict[str, Any]:
    """Reset the environment for the given task type."""
    if task_type not in ENVIRONMENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Task type '{task_type}' not found. Available: {list(ENVIRONMENTS.keys())}",
        )
    env = ENVIRONMENTS[task_type]
    obs = env.reset()
    return {"observation": obs, "task_type": task_type}


@app.post("/step/{task_type}")
def step_environment(task_type: str, action: ActionRequest) -> Dict[str, Any]:
    """Execute a step in the environment."""
    if task_type not in ENVIRONMENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Task type '{task_type}' not found. Available: {list(ENVIRONMENTS.keys())}",
        )

    env = ENVIRONMENTS[task_type]
    action_dict = action.model_dump(exclude_none=True)

    try:
        obs, reward, done, truncated, info = env.step(action_dict)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state/{task_type}")
def get_state(task_type: str) -> Dict[str, Any]:
    """Get the current state of the environment."""
    if task_type not in ENVIRONMENTS:
        raise HTTPException(
            status_code=404,
            detail=f"Task type '{task_type}' not found. Available: {list(ENVIRONMENTS.keys())}",
        )
    return ENVIRONMENTS[task_type].state()


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint with environment info."""
    return {
        "name": "OpenEnv Ticket Triage",
        "description": "Customer Support Ticket Triage Environment for Meta OpenEnv Hackathon 2026",
        "tasks": {
            "classification": "Easy - Route tickets to the correct department",
            "priority_classification": "Medium - Classify department AND set priority",
            "efficiency_triage": "Hard - Route 10 tickets efficiently",
        },
        "endpoints": {
            "health": "/health",
            "reset": "/reset/{task_type}",
            "step": "/step/{task_type}",
            "state": "/state/{task_type}",
            "docs": "/docs",
        },
        "openenv_spec": "/openenv.yaml",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
