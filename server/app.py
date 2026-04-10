from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
from typing import Dict, Any

class ResetRequest(BaseModel):
    task: str = "task_1_classification"

class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool

class StepRequest(BaseModel):
    action: str
    params: Dict[str, Any] = {}

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool

class StateResponse(BaseModel):
    task: str
    step: int
    total_reward: float

API_BASE_URL = os.getenv('API_BASE_URL', 'https://api.openai.com/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
HF_TOKEN = os.getenv('HF_TOKEN')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenEnv Ticket Triage", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKETS = [
    {'id': 'TKT-001', 'category': 'billing', 'priority': 'high', 'sentiment': 'negative'},
    {'id': 'TKT-002', 'category': 'technical', 'priority': 'critical', 'sentiment': 'negative'},
    {'id': 'TKT-003', 'category': 'general', 'priority': 'low', 'sentiment': 'positive'},
]

current_env = {
    'task': None,
    'tickets': [],
    'current_idx': 0,
    'step': 0,
    'total_reward': 0.0,
    'initialized': False
}

def grader_classification(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.lower() == ground_truth.lower() else 0.0

def grader_assignment(assigned: str, correct: str) -> float:
    return 1.0 if assigned.lower() == correct.lower() else 0.5

def grader_resolution(steps: int, optimal: int = 3) -> float:
    if steps <= optimal:
        return 1.0
    return max(0.0, 1.0 - (steps - optimal) * 0.1)

@app.post("/openenv/reset")
async def reset():
    global current_env
    
    current_env = {
        'task': 'task_1_classification',
        'tickets': TICKETS.copy(),
        'current_idx': 0,
        'step': 0,
        'total_reward': 0.0,
        'initialized': True
    }
    
    ticket = current_env['tickets'][0]
    logger.info(f"[RESET] Environment initialized")
    
    return {
        'observation': {
            'ticket_id': ticket['id'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'sentiment': ticket['sentiment'],
            'status': 'open'
        },
        'reward': 0.0,
        'done': False
    }

@app.post("/openenv/step", response_model=StepResponse)
async def step(request: StepRequest):
    global current_env
    
    if not current_env['initialized']:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /openenv/reset first")
    
    current_env['step'] += 1
    reward = 0.0
    
    current_ticket = current_env['tickets'][current_env['current_idx']]
    action = request.action
    params = request.params or {}
    
    if current_env['task'] == 'task_1_classification':
        reward = grader_classification(action, current_ticket['category'])
    elif current_env['task'] == 'task_2_assignment':
        assigned = params.get('department', 'unknown')
        reward = grader_assignment(assigned, current_ticket['category'])
    elif current_env['task'] == 'task_3_resolution':
        steps = params.get('steps', 3)
        reward = grader_resolution(steps)
    
    current_env['total_reward'] += reward
    current_env['current_idx'] += 1
    done = current_env['current_idx'] >= len(current_env['tickets'])
    
    next_ticket = None
    if not done:
        next_ticket = current_env['tickets'][current_env['current_idx']]
    
    logger.info(f"[STEP] {current_env['step']} - Action: {action}, Reward: {reward:.2f}, Done: {done}")
    
    return StepResponse(
        observation={
            'ticket_id': next_ticket['id'] if next_ticket else None,
            'category': next_ticket['category'] if next_ticket else None,
            'priority': next_ticket['priority'] if next_ticket else None,
            'sentiment': next_ticket['sentiment'] if next_ticket else None,
            'status': 'processing'
        },
        reward=reward,
        done=done
    )

@app.get("/openenv/state", response_model=StateResponse)
async def get_state():
    if not current_env['initialized']:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    
    return StateResponse(
        task=current_env['task'],
        step=current_env['step'],
        total_reward=current_env['total_reward']
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "openenv-ticket-triage",
        "model": MODEL_NAME
    }

@app.on_event("startup")
async def startup():
    logger.info(f"OpenEnv Ticket Triage Server Starting")
    logger.info(f"API_BASE_URL: {API_BASE_URL}")
    logger.info(f"MODEL_NAME: {MODEL_NAME}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)