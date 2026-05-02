# Smart Building Energy Management — RL + MLOps

## Problem Statement
Train a Q-learning agent to manage energy in a smart building 
by controlling AC and lighting to minimize energy usage 
while maintaining occupant comfort.

## SDG Connection
**SDG 7 — Affordable and Clean Energy**
Reducing unnecessary energy consumption in buildings 
directly supports SDG 7 by promoting energy efficiency.

## Project Structure
Smart_Building_RL/
├── sim/
│   ├── environment.py   # building simulator
│   ├── agent.py         # Q-learning agent
│   └── train.py         # training loop
├── configs/
│   └── qlearning_v1.yaml
├── experiments/
│   └── results.csv
├── policy_v1.pkl
├── policy_v2_explored.pkl
└── README.md

## How to Reproduce
1. Clone the repo
```bash
git clone https://github.com/AashishSrinivasa/Smart-Building-RL.git
cd Smart-Building-RL
```

2. Install dependencies
```bash
pip install numpy
```

3. Run training
```bash
python sim/train.py
```

## Monitoring Plan
If deployed in a real building we would monitor:
- Average energy consumption per hour
- Occupant comfort score
- Episodes where agent chose wrong action
- Epsilon value over time to track exploration vs exploitation

## Results
Training over 1000 episodes shows the agent learns to:
- Turn devices off during unoccupied hours
- Use low energy mode during mild occupancy
- Balance comfort and energy saving effectively