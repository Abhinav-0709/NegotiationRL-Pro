import sys
import os
import torch
from typing import Dict, Any

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.environment import NegotiationEnv
from env.models import Action, ActionType, NegotiationOffer
from env.tasks import get_task
from baseline.torch_agent import NegotiationPolicy, preprocess_obs

def run_inference(task_id: str = "easy_negotiation"):
    """
    Standardized inference function for Scaler OpenEnv Hub.
    Executes a full episode using the PyTorch baseline model.
    """
    print(f"🎬 Starting Inference for Task: {task_id}")
    
    # 1. Initialize Env
    task_config = get_task(task_id)
    env = NegotiationEnv(task_config)
    
    # 2. Load Model
    # Dimensions matched to openenv.yaml: Input=4, Output=3
    policy = NegotiationPolicy(4, 3) 
    
    # In a production scenario, we would load weights here:
    # policy.load_state_dict(torch.load("model_weights.pt"))
    policy.eval()
    
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        state_tensor = preprocess_obs(obs)
        
        with torch.no_grad():
            action_probs = policy(state_tensor)
            action_idx = torch.argmax(action_probs).item()
        
        if action_idx == 0: # OFFER
            price = obs.role_context["target_price"] + (obs.role_context["limit_price"] - obs.role_context["target_price"]) * (obs.round_number / obs.max_rounds)
            agent_action = Action(
                type=ActionType.OFFER,
                parameters=NegotiationOffer(
                    price=round(price, 2),
                    delivery_days=7,
                    quality_score=0.8
                )
            )
        elif action_idx == 1: # ACCEPT
            agent_action = Action(type=ActionType.ACCEPT)
        else: # WALK_AWAY
            agent_action = Action(type=ActionType.WALK_AWAY)

        obs, reward, done, info = env.step(agent_action)
        total_reward += reward.value

    print(f"✅ Inference Complete. Final Status: {env.state().status}")
    print(f"💰 Total Reward: {total_reward:.2f}")
    return env.state()

if __name__ == "__main__":
    # If a task ID is passed via command line, use it
    target_task = sys.argv[1] if len(sys.argv) > 1 else "easy_negotiation"
    run_inference(target_task)
