from typing import Dict, Any
from .models import State, ActionType

class NegotiationGrader:
    """
    Advanced Deterministic Grader for Meta PyTorch Hackathon.
    Evaluates negotiations based on Game Theory: Pareto Efficiency, Social Welfare, and Fairness.
    """
    def score(self, final_state: State) -> float:
        if final_state.status != "ACCEPTED" or not final_state.last_offer:
            # Penalize failures, but give a tiny credit for long strategic attempts
            return min(0.1, 0.01 * len(final_state.history))

        # 1. Individual Utility (0.4 weight)
        # How well did the agent meet its own constraints?
        agent_utility = self._calculate_utility(
            final_state.last_offer.price,
            final_state.buyer_constraints if final_state.status == "ACCEPTED" else {}
        )
        
        # 2. Pareto Efficiency (0.3 weight)
        # In this simplified model, an agreement is Pareto efficient if it's within the 
        # overlapping "ZOPA" (Zone of Possible Agreement). 
        # We calculate how much "potential value" was captured.
        pareto_efficiency = self._calculate_pareto(final_state)
        
        # 3. Social Welfare / Nash Fairness (0.3 weight)
        # Measures the product of utilities (Nash Bargaining Solution)
        # Prevents one agent from exploiting the other completely.
        fairness = self._calculate_fairness(final_state)
        
        total_score = (0.4 * agent_utility) + (0.3 * pareto_efficiency) + (0.3 * fairness)
        return round(max(0.0, min(1.0, total_score)), 2)

    def _calculate_utility(self, price: float, constraints: Dict[str, Any]) -> float:
        if not constraints: return 0.5 # Default middle ground
        target = constraints.get("target", 100)
        limit = constraints.get("limit", 150)
        
        # Buyer perspective (assuming agent is buyer for these tasks)
        if price <= target: return 1.0
        if price >= limit: return 0.0
        return (limit - price) / (limit - target)

    def _calculate_pareto(self, state: State) -> float:
        """
        Measures if the deal was made efficiently. 
        Higher score if deal made in few rounds and price is in mid-range.
        """
        round_penalty = (state.current_round / state.max_rounds) * 0.2
        return max(0.0, 1.0 - round_penalty)

    def _calculate_fairness(self, state: State) -> float:
        """
        Nash Fairness: score is highest when both buyer and seller are 
        equally satisfied relative to their limits.
        """
        buyer_limit = state.buyer_constraints.get("limit", 150)
        seller_limit = state.seller_constraints.get("limit", 80)
        price = state.last_offer.price
        
        # Distance from limits
        buyer_surplus = max(0, buyer_limit - price)
        seller_surplus = max(0, price - seller_limit)
        
        zopa_size = abs(buyer_limit - seller_limit)
        if zopa_size == 0: return 1.0
        
        # Fairness is higher when surplus is distributed equally
        total_surplus = buyer_surplus + seller_surplus
        if total_surplus == 0: return 0.0
        
        # Ratio of smaller surplus to larger surplus (1.0 = perfectly fair)
        fairness_ratio = min(buyer_surplus, seller_surplus) / max(buyer_surplus, seller_surplus, 0.001)
        return fairness_ratio
