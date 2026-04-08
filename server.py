from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Optional
from env.environment import NegotiationEnv
from env.models import Action, Observation, Reward, State, ActionType, NegotiationOffer
from env.tasks import get_task

app = FastAPI(title="NegotiationRL-Pro OpenEnv Standard API")

# Global environment instance (as per OpenEnv headless simulation standard)
_env: Optional[NegotiationEnv] = None

@app.get("/")
async def health_check():
    return {
        "status": "ready",
        "version": "2.0.0",
        "standards": ["OpenEnv", "Gymnasium"],
        "metadata": {
            "name": "NegotiationRL-Pro",
            "author": "Hackathon Participant"
        }
    }

@app.post("/reset")
async def reset(task_id: str = "easy_negotiation"):
    """Initializes the environment for a specific task."""
    global _env
    try:
        task_config = get_task(task_id)
        _env = NegotiationEnv(task_config)
        obs = _env.reset()
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {str(e)}")

@app.post("/step")
async def step(action: Action):
    """Execution step: State -> Action -> Environment Change -> Reward."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step execution failed: {str(e)}")

@app.get("/state")
async def get_state():
    """Returns the full internal state for analysis or grading."""
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return _env.state()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
