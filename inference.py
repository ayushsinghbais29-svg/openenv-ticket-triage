from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import json
from typing import Optional, Dict, Any

app = FastAPI(title="OpenEnv Ticket Triage API")

# Global state
current_state = None
tickets = []
current_ticket_index = 0

class Ticket:
    def __init__(self):
        self.ticket_id = f"TKT-{random.randint(10000, 99999)}"
        self.status = random.choice(["open", "assigned", "in_progress"])
        self.priority = random.choice(["low", "medium", "high", "critical"])
        self.department = random.choice(["Billing", "Technical", "General", "Premium Support"])
        self.assignee = None
        self.sentiment = random.choice(["positive", "neutral", "negative"])

class ResetRequest(BaseModel):
    task_type: str = "classification"

@app.post("/openenv/reset")
async def reset(request: ResetRequest = None):
    global current_state, tickets, current_ticket_index
    
    task_type = "classification"
    if request:
        task_type = request.task_type
    
    # Generate tickets
    num_tickets = 5 if task_type in ["classification", "priority_classification"] else 10
    tickets = [Ticket() for _ in range(num_tickets)]
    current_ticket_index = 0
    
    current_state = {
        "ticket_id": tickets[0].ticket_id,
        "ticket_status": tickets[0].status,
        "ticket_priority": tickets[0].priority,
        "department": tickets[0].department,
        "assignee": tickets[0].assignee,
        "sentiment": tickets[0].sentiment,
        "task_type": task_type,
        "step_count": 0,
        "ticket_index": 0
    }
    
    return {
        "observation": current_state,
        "reward": 0.0,
        "done": False,
        "info": {
            "step_count": 0,
            "total_reward": 0.0
        }
    }

@app.get("/openenv/step")
async def step(
    action_type: str,
    department: str = None,
    priority: str = None,
    assignee: str = None,
    confidence: float = 0.5
):
    global current_state, current_ticket_index, tickets
    
    if current_state is None:
        raise HTTPException(status_code=400, detail="Must call /openenv/reset first")
    
    current_state["step_count"] += 1
    reward = 0.0
    
    # Reward logic
    if action_type == "read":
        reward = 0.1
    elif action_type == "analyze":
        reward = 0.2
    elif action_type == "assign_ticket":
        current_state["assignee"] = assignee
        reward = 0.5 * confidence
        current_state["ticket_status"] = "assigned"
    elif action_type == "close_ticket":
        reward = 0.3
        current_state["ticket_status"] = "closed"
    elif action_type == "add_comment":
        reward = 0.15
    
    # Move to next ticket if action completed
    if action_type in ["close_ticket", "assign_ticket"]:
        current_ticket_index += 1
        if current_ticket_index < len(tickets):
            ticket = tickets[current_ticket_index]
            current_state["ticket_id"] = ticket.ticket_id
            current_state["ticket_status"] = ticket.status
            current_state["ticket_priority"] = ticket.priority
            current_state["department"] = ticket.department
            current_state["assignee"] = None
            current_state["sentiment"] = ticket.sentiment
            current_state["ticket_index"] = current_ticket_index
    
    # Check if done
    done = current_ticket_index >= len(tickets) or current_state["step_count"] >= 30
    
    return {
        "observation": current_state,
        "reward": reward,
        "done": done,
        "info": {
            "step_count": current_state["step_count"],
            "total_reward": reward,
            "current_ticket": current_ticket_index + 1,
            "total_tickets": len(tickets)
        }
    }

@app.get("/openenv/state")
async def get_state():
    if current_state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /openenv/reset first")
    return current_state

@app.get("/health")
async def health():
    return {"status": "ok", "service": "openenv-ticket-triage"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
