from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import json

app = FastAPI()

class ResetRequest(BaseModel):
    task_type: str = "classification"

class StepRequest(BaseModel):
    action_type: str
    department: str = None
    priority: str = None

# Store current state
current_state = None

@app.post("/openenv/reset")
async def reset(request: ResetRequest = None):
    global current_state
    task_type = request.task_type if request else "classification"
    
    current_state = {
        "ticket_id": f"TKT-{random.randint(1000, 9999)}",
        "ticket_status": "open",
        "ticket_priority": random.choice(["low", "medium", "high", "critical"]),
        "assignee": None,
        "task_type": task_type,
        "step_count": 0
    }
    
    return {
        "observation": current_state,
        "reward": 0.0,
        "done": False,
        "info": {"step_count": 0}
    }

@app.get("/openenv/step")
async def step(action_type: str, department: str = None, priority: str = None, confidence: float = 0.5):
    global current_state
    
    if current_state is None:
        raise HTTPException(status_code=400, detail="Must call /openenv/reset first")
    
    current_state["step_count"] += 1
    
    # Calculate reward
    reward = 0.1 if action_type == "read" else 0.2 if action_type == "analyze" else 0.5
    
    if action_type == "assign_ticket" and department:
        current_state["assignee"] = department
        reward = 0.5 * confidence
    
    if current_state["step_count"] >= 20:
        current_state["done"] = True
    
    return {
        "observation": current_state,
        "reward": reward,
        "done": current_state.get("done", False),
        "info": {"step_count": current_state["step_count"], "total_reward": reward}
    }

@app.get("/openenv/state")
async def get_state():
    if current_state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    return current_state

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
