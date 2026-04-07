import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import List

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import NegotiationEnv
from env.models import Action, ActionType, NegotiationOffer, PersonalityType
from env.tasks import TASKS
from env.graders import NegotiationGrader

class NegotiationPolicy(nn.Module):
    """
    A simple PyTorch policy network for demonstrating AI-trainability.
    In a real submission, this would be trained using TRL or PPO.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

def preprocess_obs(obs) -> torch.Tensor:
    """Converts structured Observation to a flat tensor."""
    # Simplified vector: [round_normalized, last_offer_price, target_price, limit_price]
    round_norm = obs.round_number / obs.max_rounds
    last_price = obs.current_offer.price / 1000.0 if obs.current_offer else 0.0
    target = obs.role_context.get("target_price", 100) / 1000.0
    limit = obs.role_context.get("limit_price", 150) / 1000.0
    
    return torch.tensor([round_norm, last_price, target, limit], dtype=torch.float32)

def run_torch_agent():
    print("🚀 Initializing National-Level Baseline Agent (PyTorch)")
    
    # 1. Setup Environment
    task = TASKS[0] # Easy task
    env = NegotiationEnv(task)
    grader = NegotiationGrader()
    
    # 2. Initialize Policy
    # Input: [round, last_price, target, limit] -> 4
    # Output: [OFFER, ACCEPT, WALK_AWAY] -> 3
    policy = NegotiationPolicy(4, 3)
    
    obs = env.reset()
    done = False
    total_reward = 0
    
    print(f"Negotiating against: {env.personality.value} Opponent")
    
    while not done:
        state_tensor = preprocess_obs(obs)
        
        # Inference
        with torch.no_grad():
            action_probs = policy(state_tensor)
            action_idx = torch.argmax(action_probs).item()
        
        # Mapping index to ActionType
        if action_idx == 0: # OFFER
            # Heuristic for price within the torch agent (for simplicity)
            target = obs.role_context["target_price"]
            limit = obs.role_context["limit_price"]
            # Concede slightly based on rounds
            price = target + (limit - target) * (obs.round_number / obs.max_rounds)
            
            agent_action = Action(
                type=ActionType.OFFER,
                parameters=NegotiationOffer(
                    price=round(price , 2),
                    delivery_days=7,
                    quality_score=0.8
                )
            )
        elif action_idx == 1: # ACCEPT
            agent_action = Action(type=ActionType.ACCEPT)
        else: # WALK_AWAY
            agent_action = Action(type=ActionType.WALK_AWAY)

        print(f"Round {obs.round_number}: Agent {agent_action.type.value}")
        
        # Step
        obs, reward, done, info = env.step(agent_action)
        total_reward += reward.value

    # 3. Final Grading
    final_score = grader.score(env.state())
    print("\n--- Evaluation Results ---")
    print(f"Final Outcome: {env.state().status}")
    print(f"Total Cumulative Reward: {total_reward:.2f}")
    print(f"Judge's Score: {final_score:.2f}")
    print("--------------------------")

if __name__ == "__main__":
    run_torch_agent()
