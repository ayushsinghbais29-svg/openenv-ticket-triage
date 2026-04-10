from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ticket data
TICKETS = [
    {'id': 'TKT-001', 'category': 'billing', 'priority': 'high', 'sentiment': 'negative'},
    {'id': 'TKT-002', 'category': 'technical', 'priority': 'critical', 'sentiment': 'negative'},
    {'id': 'TKT-003', 'category': 'general', 'priority': 'low', 'sentiment': 'positive'},
]

# Global environment state
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

def reset_env():
    """Reset environment - called via POST /"""
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

def step_env(action: str):
    """Take action - called via POST /act"""
    global current_env
    
    if not current_env['initialized']:
        return {
            'observation': None,
            'reward': 0.0,
            'done': True,
            'error': 'Environment not initialized. Call reset first.'
        }
    
    current_env['step'] += 1
    reward = 0.0
    
    current_ticket = current_env['tickets'][current_env['current_idx']]
    reward = grader_classification(action, current_ticket['category'])
    
    current_env['total_reward'] += reward
    current_env['current_idx'] += 1
    done = current_env['current_idx'] >= len(current_env['tickets'])
    
    next_ticket = None
    if not done:
        next_ticket = current_env['tickets'][current_env['current_idx']]
    
    logger.info(f"[STEP] {current_env['step']} - Action: {action}, Reward: {reward:.2f}, Done: {done}")
    
    return {
        'observation': {
            'ticket_id': next_ticket['id'] if next_ticket else None,
            'category': next_ticket['category'] if next_ticket else None,
            'priority': next_ticket['priority'] if next_ticket else None,
            'sentiment': next_ticket['sentiment'] if next_ticket else None,
            'status': 'processing' if not done else 'closed'
        } if next_ticket or done else None,
        'reward': reward,
        'done': done
    }

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def reset():
    """OpenEnv reset endpoint"""
    return reset_env()

@app.post("/act")
async def act(request: dict):
    """OpenEnv action endpoint"""
    action = request.get("action", "")
    return step_env(action)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "openenv-ticket-triage"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)