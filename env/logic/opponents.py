import random
from typing import Optional
from ..models import Action, ActionType, NegotiationOffer, PersonalityType

class BaseOpponent:
    def __init__(self, role: str, constraints: dict, personality: PersonalityType):
        self.role = role
        self.constraints = constraints
        self.personality = personality

    def respond(self, agent_offer: Optional[NegotiationOffer], round_num: int, max_rounds: int) -> Action:
        raise NotImplementedError

class AggressiveOpponent(BaseOpponent):
    """The Wall: Slow concessions, high pressure."""
    def respond(self, agent_offer: Optional[NegotiationOffer], round_num: int, max_rounds: int) -> Action:
        if not agent_offer:
            return Action(type=ActionType.OFFER, parameters=self._get_initial_offer())
        
        target = self.constraints["target_price"]
        limit = self.constraints["limit_price"]
        
        # Accept only if it's better than 95% of target
        is_acceptable = (agent_offer.price >= target * 0.95) if self.role == "seller" else (agent_offer.price <= target * 1.05)
        
        if is_acceptable:
            return Action(type=ActionType.ACCEPT)
        
        # Small concession based on deadline
        progress = round_num / max_rounds
        concession_rate = 0.05 * progress # Very slow
        
        new_price = self._calculate_concession(agent_offer.price, concession_rate)
        return Action(type=ActionType.OFFER, parameters=self._create_offer(new_price))

    def _calculate_concession(self, agent_price: float, rate: float) -> float:
        target = self.constraints["target_price"]
        return target + (agent_price - target) * (1 - rate)

    def _get_initial_offer(self) -> NegotiationOffer:
        return self._create_offer(self.constraints["target_price"])

    def _create_offer(self, price: float) -> NegotiationOffer:
        return NegotiationOffer(
            price=round(price, 2),
            delivery_days=self.constraints.get("target_delivery", 7),
            quality_score=self.constraints.get("target_quality", 0.9),
            shipping_priority=1
        )

class CooperativeOpponent(BaseOpponent):
    """The Win-Win: Seeks fair middle ground quickly."""
    def respond(self, agent_offer: Optional[NegotiationOffer], round_num: int, max_rounds: int) -> Action:
        if not agent_offer:
            return Action(type=ActionType.OFFER, parameters=self._get_initial_offer())
        
        target = self.constraints["target_price"]
        limit = self.constraints["limit_price"]
        
        # Accept if it's within feasible range (even at limit)
        is_acceptable = (agent_offer.price >= limit) if self.role == "seller" else (agent_offer.price <= limit)
        
        if is_acceptable:
            # High chance to accept even if not perfect
            if random.random() > 0.2:
                return Action(type=ActionType.ACCEPT)
        
        # Fast concession towards agent
        concession_rate = 0.2 + (0.3 * (round_num / max_rounds))
        new_price = agent_offer.price * concession_rate + target * (1 - concession_rate)
        
        return Action(type=ActionType.OFFER, parameters=self._create_offer(new_price))

    def _get_initial_offer(self) -> NegotiationOffer:
        return self._create_offer(self.constraints["target_price"])

    def _create_offer(self, price: float) -> NegotiationOffer:
        return NegotiationOffer(
            price=round(price, 2),
            delivery_days=self.constraints.get("target_delivery", 7),
            quality_score=self.constraints.get("target_quality", 0.8),
            shipping_priority=1
        )

class TitForTatOpponent(BaseOpponent):
    """The Mirror: Matches the agent's last concession rate."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_agent_price = None

    def respond(self, agent_offer: Optional[NegotiationOffer], round_num: int, max_rounds: int) -> Action:
        if not agent_offer:
            return Action(type=ActionType.OFFER, parameters=self._create_offer(self.constraints["target_price"]))
        
        if self.last_agent_price is None:
            self.last_agent_price = agent_offer.price
            return Action(type=ActionType.OFFER, parameters=self._create_offer(self.constraints["target_price"]))

        # Calculate agent concession
        agent_concession = abs(agent_offer.price - self.last_agent_price)
        self.last_agent_price = agent_offer.price
        
        # Match it
        target = self.constraints["target_price"]
        if self.role == "seller":
            new_price = max(self.constraints["limit_price"], agent_offer.price + agent_concession)
        else:
            new_price = min(self.constraints["limit_price"], agent_offer.price - agent_concession)
            
        return Action(type=ActionType.OFFER, parameters=self._create_offer(new_price))

    def _create_offer(self, price: float) -> NegotiationOffer:
        return NegotiationOffer(
            price=round(price, 2),
            delivery_days=7,
            quality_score=0.8
        )

def get_opponent(role: str, constraints: dict, personality: PersonalityType) -> BaseOpponent:
    if personality == PersonalityType.AGGRESSIVE:
        return AggressiveOpponent(role, constraints, personality)
    elif personality == PersonalityType.COOPERATIVE:
        return CooperativeOpponent(role, constraints, personality)
    elif personality == PersonalityType.TIT_FOR_TAT:
        return TitForTatOpponent(role, constraints, personality)
    else:
        # Fallback to aggressive for challenge
        return AggressiveOpponent(role, constraints, PersonalityType.AGGRESSIVE)
