---
title: NegotationRL Pro
emoji: 🤝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏆 NegotiationRL-Pro: Advanced Multi-Agent Negotiation Env

**Submission for Meta PyTorch Hackathon (National Level)**

NegotiationRL-Pro is a high-fidelity Reinforcement Learning environment designed to train AI agents in complex, multi-issue strategic negotiations. Built on **Meta's OpenEnv** standard, it goes beyond simple price sliders to simulate behavioral psychology and game-theoretic optimality.

## 🌟 Why This Wins
1.  **Strict OpenEnv Compliance**: Headless, containerized, and Gymnasium-API compliant.
2.  **PyTorch-Powered Intelligence**: Features a baseline policy network built in PyTorch to demonstrate multi-step decision-making.
3.  **Behavioral Depth**: Agents face 4 distinct **Psychological Personalities** (Aggressive, Cooperative, Tit-for-Tat, Random).
4.  **Game-Theoretic Evaluation**: Graders calculate **Pareto Efficiency** and **Nash Fairness**, ensuring agents optimize for "Social Welfare," not just greed.

## ⚙️ Core Architecture
- **Environment**: Multi-issue simulator with a non-linear Trade-off Engine (Quality vs. Priority vs. Price).
- **Agents**: Supports both Buyer and Seller roles with asymmetric information.
- **Reward System**: Continuous, normalized (0.0 - 1.0) reward shaping based on negotiation progress.
- **Hugging Face Hub**: Integrated metadata for automated model tracking.

## 🚀 Quick Start (For Judges)

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the National-Level Baseline (PyTorch)
This script demonstrates the agent using a Neural Network to negotiate against various personalities:
```bash
python baseline/torch_agent.py
```

### 3. Docker Verification
```bash
docker build -t negotiation-pro .
docker run negotiation-pro
```

## 📊 Performance Metrics
Our environment evaluates agents on three pillars:
- **Individual Utility**: Satisfaction of private constraints.
- **Pareto Optimality**: Efficiency of the reached deal (leaving no value on the table).
- **Nash Fairness**: Equilibrium of the agreement relative to both parties' limits.

## 📂 Project Structure
- `env/`: Meta OpenEnv core logic.
  - `logic/`: Personality-driven opponent simulations.
  - `models.py`: Pydantic-validated data structures.
  - `graders.py`: Game-theoretic evaluation logic.
- `baseline/`: PyTorch agent implementations.
- `openenv.yaml`: Standardized metadata with HF tags.
- `Dockerfile`: Production-ready containerization.

---
*Created for the Meta PyTorch Hackathon.*
