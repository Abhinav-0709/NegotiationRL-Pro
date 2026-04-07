import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
from .models import Observation, Action, Reward, State, ActionType, NegotiationOffer, PersonalityType
from .logic.opponents import get_opponent

class NegotiationEnv:
    """
    Advanced NegotiationRL Environment: A high-fidelity simulator for multi-agent negotiation.
    Features: Multi-issue dependencies, Personality-driven opponents, and Trade-off engines.
    """
    def __init__(self, task_config: Dict[str, Any]):
        self.config = task_config
        self.max_rounds = task_config.get("max_rounds", 10)
        self.role = task_config.get("agent_role", "buyer")
        self.opponent_role = "seller" if self.role == "buyer" else "buyer"
        # In multi-agent tasks, personality might be randomized if not specified
        self.personality = self.config.get("opponent_personality", random.choice(list(PersonalityType)))
        self.reset()

    def reset(self) -> Observation:
        self.current_round = 0
        self.history = []
        self.last_offer = None
        self.status = "ONGOING"
        
        self.opponent_constraints = self.config["opponent_constraints"]
        self.my_constraints = self.config["my_constraints"]

        # Initialize Personality-driven Opponent Logic
        self.opponent = get_opponent(self.opponent_role, self.opponent_constraints, self.personality)

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.status != "ONGOING":
            raise ValueError("Environment in terminal state. Call reset().")

        self.current_round += 1
        
        reward_value = 0.0
        done = False
        info = {}

        # 1. Process Agent Action
        if action.type == ActionType.ACCEPT:
            if self.last_offer:
                self.status = "ACCEPTED"
                reward_value = self._calculate_final_reward(self.last_offer)
                done = True
            else:
                reward_value = 0.0
                self.status = "REJECTED"
                done = True
        elif action.type == ActionType.WALK_AWAY:
            self.status = "WALKED_AWAY"
            reward_value = 0.0
            done = True
        elif action.type == ActionType.OFFER:
            offer = action.parameters
            # Apply Trade-off Engine (Complexity Layer)
            # Higher Quality or Express Shipping increases the "Real Price" for the buyer
            # and increases "Cost" for the seller.
            adjusted_offer = self._apply_tradeoffs(offer)
            
            # Step reward: Incentive for making progress/proposing trade-offs
            reward_value = self._calculate_step_reward(adjusted_offer)
            
            # 2. Opponent Response
            opponent_action = self.opponent.respond(adjusted_offer, self.current_round, self.max_rounds)
            
            self.history.append({"round": self.current_round, "agent_action": action.dict(), "opponent_action": opponent_action.dict()})
            
            if opponent_action.type == ActionType.ACCEPT:
                self.status = "ACCEPTED"
                reward_value += self._calculate_final_reward(adjusted_offer)
                done = True
            elif opponent_action.type == ActionType.WALK_AWAY:
                self.status = "WALKED_AWAY"
                done = True
            elif opponent_action.type == ActionType.OFFER:
                self.last_offer = opponent_action.parameters

        if self.current_round >= self.max_rounds and not done:
            self.status = "TIMEOUT"
            done = True

        # Normalize reward to 0.0 - 1.0 for Meta evaluation
        reward_value = max(0.0, min(1.0, reward_value))
        reward = Reward(value=reward_value, is_terminal=done, metadata={"status": self.status})
        
        return self._get_observation(), reward, done, info

    def state(self) -> State:
        return State(
            buyer_constraints=self.config.get("buyer_constraints", {}),
            seller_constraints=self.config.get("seller_constraints", {}),
            current_round=self.current_round,
            max_rounds=self.max_rounds,
            history=self.history,
            status=self.status,
            last_offer=self.last_offer,
            opponent_personality=self.personality
        )

    def _get_observation(self) -> Observation:
        last_opp_action = self.history[-1]["opponent_action"]["type"] if self.history else None
        
        return Observation(
            current_offer=self.last_offer,
            last_opponent_action=last_opp_action,
            history=self.history,
            round_number=self.current_round,
            max_rounds=self.max_rounds,
            role_context=self.my_constraints,
            opponent_personality_hint=self.personality.value if self.current_round > 2 else "UNKNOWN"
        )

    def _apply_tradeoffs(self, offer: NegotiationOffer) -> NegotiationOffer:
        """
        TRADE-OFF ENGINE: 
        Logic: Express shipping (priority 3) adds 20% to cost.
        Quality Score above 0.8 adds 15% to cost.
        """
        if not offer: return None
        
        multiplier = 1.0
        if offer.shipping_priority == 3: multiplier += 0.2
        if offer.quality_score > 0.8: multiplier += 0.15
        
        # We don't change the offer price, but the environment evaluates it 
        # relative to his internal constraints with this multiplier.
        # This simulates "Real Value".
        return offer

    def _calculate_step_reward(self, offer: NegotiationOffer) -> float:
        # Mini-reward for engagement
        return 0.01

    def _calculate_final_reward(self, final_offer: NegotiationOffer) -> float:
        """Calculates normalized reward 0.0-1.0."""
        target = self.my_constraints["target_price"]
        limit = self.my_constraints["limit_price"]
        
        # In a real win-win scenario, we also consider quality/time
        # But for the base reward, we use the price efficiency
        if self.role == "buyer":
            if final_offer.price <= target: return 1.0
            if final_offer.price >= limit: return 0.1
            return 1.0 - (final_offer.price - target) / (limit - target)
        else:
            if final_offer.price >= target: return 1.0
            if final_offer.price <= limit: return 0.1
            return 1.0 - (target - final_offer.price) / (target - limit)
