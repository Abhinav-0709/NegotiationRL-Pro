from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class ActionType(str, Enum):
    OFFER = "OFFER"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    WALK_AWAY = "WALK_AWAY"

class PersonalityType(str, Enum):
    AGGRESSIVE = "AGGRESSIVE" # Low concession, high pressure
    COOPERATIVE = "COOPERATIVE" # Win-win focused, fair concessions
    TIT_FOR_TAT = "TIT_FOR_TAT" # Mimics partner's behavior
    RANDOM = "RANDOM" # Unpredictable

class NegotiationOffer(BaseModel):
    price: float = Field(..., description="The price being offered")
    delivery_days: int = Field(..., description="Requested delivery time in days")
    quality_score: float = Field(..., description="Target quality score (0.0 - 1.0)")
    # New: Add resource usage to simulate constraints
    shipping_priority: int = Field(1, ge=1, le=3, description="1: Standard, 2: Express, 3: Next-Day")

class Action(BaseModel):
    type: ActionType
    parameters: Optional[NegotiationOffer] = None

class Observation(BaseModel):
    current_offer: Optional[NegotiationOffer] = None
    last_opponent_action: Optional[ActionType] = None
    history: List[Dict[str, Any]] = []
    round_number: int
    max_rounds: int
    # Context specific to the agent's role (Buyer/Seller)
    role_context: Dict[str, Any] = {}
    opponent_personality_hint: Optional[str] = "UNKNOWN"

class Reward(BaseModel):
    value: float = Field(..., description="Normalized continuous reward signal (0.0 - 1.0)")
    is_terminal: bool = False
    metadata: Dict[str, Any] = {}

class State(BaseModel):
    buyer_constraints: Dict[str, Any]
    seller_constraints: Dict[str, Any]
    current_round: int
    max_rounds: int
    history: List[Dict[str, Any]]
    status: str # "ONGOING", "ACCEPTED", "REJECTED", "WALKED_AWAY", "TIMEOUT"
    last_offer: Optional[NegotiationOffer] = None
    opponent_personality: PersonalityType
