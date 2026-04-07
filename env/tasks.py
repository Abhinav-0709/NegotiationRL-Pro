from typing import Dict, Any, List

TASKS: List[Dict[str, Any]] = [
    {
        "id": "easy_negotiation",
        "name": "Quick Deal",
        "difficulty": "Easy",
        "agent_role": "buyer",
        "max_rounds": 10,
        "my_constraints": {
            "target_price": 100,
            "limit_price": 150,
            "target_delivery": 7,
            "target_quality": 0.8
        },
        "opponent_constraints": {
            "target_price": 110,
            "limit_price": 90,
            "target_delivery": 7,
            "target_quality": 0.8
        },
        "buyer_constraints": {"target": 100, "limit": 150},
        "seller_constraints": {"target": 110, "limit": 90}
    },
    {
        "id": "medium_negotiation",
        "name": "Strategic Gap",
        "difficulty": "Medium",
        "agent_role": "buyer",
        "max_rounds": 15,
        "my_constraints": {
            "target_price": 100,
            "limit_price": 120,
            "target_delivery": 5,
            "target_quality": 0.9
        },
        "opponent_constraints": {
            "target_price": 140,
            "limit_price": 115,
            "target_delivery": 10,
            "target_quality": 0.7
        },
        "buyer_constraints": {"target": 100, "limit": 120},
        "seller_constraints": {"target": 140, "limit": 115}
    },
    {
        "id": "hard_negotiation",
        "name": "Executive Standoff",
        "difficulty": "Hard",
        "agent_role": "buyer",
        "max_rounds": 20,
        "my_constraints": {
            "target_price": 500,
            "limit_price": 550,
            "target_delivery": 3,
            "target_quality": 1.0
        },
        "opponent_constraints": {
            "target_price": 700,
            "limit_price": 540,
            "target_delivery": 14,
            "target_quality": 0.5
        },
        "buyer_constraints": {"target": 500, "limit": 550},
        "seller_constraints": {"target": 700, "limit": 540}
    }
]

def get_task(task_id: str) -> Dict[str, Any]:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    raise ValueError(f"Task {task_id} not found.")
