import sys
import os
import json

# Add parent directory to path to allow imports from env.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import NegotiationEnv
from env.models import Action, ActionType, NegotiationOffer
from env.tasks import get_task
from env.graders import NegotiationGrader

def run_baseline_agent(task_id: str):
    print(f"--- Running Baseline Agent for Task: {task_id} ---")
    
    # 1. Initialize Task and Environment
    task_config = get_task(task_id)
    env = NegotiationEnv(task_config)
    grader = NegotiationGrader()
    
    obs = env.reset()
    done = False
    total_reward = 0.0
    
    print(f"Goal: {task_config['name']} ({task_config['difficulty']})")
    print(f"Constraints: {obs.role_context}")
    
    # 2. Heuristic Agent Loop
    while not done:
        # Simple Logic: 
        # - If an offer exists from the opponent and it's within our limits, accept it.
        # - Otherwise, make a counter-offer with a 5% concession from our previous offer.
        
        agent_action = None
        
        if obs.current_offer:
            # Check if acceptable
            price = obs.current_offer.price
            limit = obs.role_context["limit_price"]
            target = obs.role_context["target_price"]
            
            # Buyer: want low price
            if price <= limit:
                # If it's close to target or we are running out of rounds, accept
                if price <= target * 1.05 or obs.round_number > obs.max_rounds * 0.8:
                    agent_action = Action(type=ActionType.ACCEPT)
        
        if not agent_action:
            # Make Counter Offer
            # Start at target, concede 5% of the gap to limit each round
            round_idx = obs.round_number
            target = obs.role_context["target_price"]
            limit = obs.role_context["limit_price"]
            concession = (limit - target) * (round_idx / obs.max_rounds)
            
            my_price = target + concession
            
            agent_action = Action(
                type=ActionType.OFFER,
                parameters=NegotiationOffer(
                    price=round(my_price, 2),
                    delivery_days=obs.role_context.get("target_delivery", 7),
                    quality_score=obs.role_context.get("target_quality", 0.8)
                )
            )
        
        print(f"Round {obs.round_number}: Agent {agent_action.type.value} @ {agent_action.parameters.price if agent_action.parameters else 'N/A'}")
        
        # 3. Step Environment
        obs, reward, done, info = env.step(agent_action)
        total_reward += reward.value
        
        if obs.last_opponent_action:
             opp_price = obs.current_offer.price if obs.current_offer else 'N/A'
             print(f"        Opponent {obs.last_opponent_action.value} @ {opp_price}")

    # 4. Final results
    final_state = env.state()
    score = grader.score(final_state)
    
    print("\n--- Negotiation Ended ---")
    print(f"Status: {final_state.status}")
    print(f"Total Step Reward: {total_reward:.2f}")
    print(f"Final Grader Score: {score:.2f}")
    print("-------------------------\n")
    return score

if __name__ == "__main__":
    # Run for all tasks
    scores = []
    for task_id in ["easy_negotiation", "medium_negotiation", "hard_negotiation"]:
        scores.append(run_baseline_agent(task_id))
    
    avg_score = sum(scores) / len(scores)
    print(f"Overall Average Score: {avg_score:.2f}")
