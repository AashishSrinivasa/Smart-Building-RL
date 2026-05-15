"""
train.py — Q-learning training loop for Smart Building Energy Management
Usage:
    python sim/train.py                            # uses default config
    python sim/train.py --config configs/qlearning_v1.yaml
    python sim/train.py --config configs/qlearning_v2.yaml
"""
import argparse
import csv
import json
import os
import sys
import uuid

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.environment import BuildingEnvironment
from sim.agent import QLearningAgent


DEFAULT_CONFIG = {
    "run_id": "run_1",
    "episodes": 1000,
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 0.995,
    "n_actions": 3,
    "description": "Q-learning baseline for smart building energy management",
}


def load_config(path=None):
    if path and os.path.exists(path):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded config from {path}")
    else:
        cfg = {}
    # Fill missing keys from defaults
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg


def train(cfg):
    run_id   = cfg["run_id"]
    episodes = cfg["episodes"]
    alpha    = cfg["alpha"]
    gamma    = cfg["gamma"]
    epsilon  = cfg["epsilon"]

    env = BuildingEnvironment()
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )
    # Override epsilon_min and decay from config
    agent.epsilon_min   = cfg.get("epsilon_min", 0.1)
    agent.epsilon_decay = cfg.get("epsilon_decay", 0.995)

    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    results = []
    halfway = episodes // 2

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_energy = 0
        total_comfort = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            total_reward  += reward
            total_energy  += action       # action = energy used (0/1/2)
            # comfort: reward already encodes it; track separately
            # comfort = 1 if action >= occupancy (or occupancy==0), else 0
            occupancy = state[2]
            comfort = 1.0 if (occupancy == 0 or action >= occupancy) else 0.0
            total_comfort += comfort
            state = next_state

        avg_reward  = round(total_reward  / 24, 4)
        avg_energy  = round(total_energy  / 24, 4)
        avg_comfort = round(total_comfort / 24, 4)

        results.append({
            "run_id":            run_id,
            "episode":           episode + 1,
            "avg_reward":        avg_reward,
            "avg_energy":        avg_energy,
            "avg_waiting_time":  avg_energy,   # smart-building equivalent of waiting time (energy cost per hour)
            "avg_comfort":       avg_comfort,
            "epsilon":           round(agent.epsilon, 4),
            "alpha":             alpha,
            "gamma":             gamma,
        })

        # Save policy at halfway point
        if episode + 1 == halfway:
            agent.save_policy(f"policy_v1.pkl")
            print(f"  → policy_v1.pkl saved at episode {episode+1}")

        if (episode + 1) % 100 == 0:
            print(f"  Episode {episode+1:>4}/{episodes} | "
                  f"Avg Reward: {avg_reward:+.4f} | "
                  f"Avg Energy: {avg_energy:.4f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    # Save final policy
    agent.save_policy("policy_v2_explored.pkl")
    print(f"  → policy_v2_explored.pkl saved at episode {episodes}")

    # Save CSV results
    csv_path = "experiments/results.csv"
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(results)
    print(f"  → Results appended to {csv_path}")

    # Save JSON log for this run
    log = {
        "run_id":          run_id,
        "episodes":        episodes,
        "alpha":           alpha,
        "gamma":           gamma,
        "epsilon_start":   cfg["epsilon"],
        "epsilon_min":     agent.epsilon_min,
        "epsilon_decay":   agent.epsilon_decay,
        "description":     cfg.get("description", ""),
        "avg_reward_final":       results[-1]["avg_reward"],
        "avg_energy_final":       results[-1]["avg_energy"],
        "avg_waiting_time_final": results[-1]["avg_energy"],   # smart-building equivalent of waiting time
        "avg_comfort_final":      results[-1]["avg_comfort"],
    }
    log_path = f"experiments/log_{run_id}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  → Log saved to {log_path}")

    return agent, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-learning agent for Smart Building")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (e.g. configs/qlearning_v1.yaml)")
    args = parser.parse_args()

    # Clear old results so we start fresh
    if os.path.exists("experiments/results.csv"):
        open("experiments/results.csv", "w").close()
        print("Cleared old results.csv")

    cfg1 = load_config(args.config or "configs/qlearning_v1.yaml")
    cfg1["run_id"] = "run_1"
    print(f"\n=== Training Run 1 ({cfg1['episodes']} episodes) ===")
    agent1, _ = train(cfg1)

    cfg2 = load_config("configs/qlearning_v2.yaml")
    cfg2["run_id"] = "run_2"
    print(f"\n=== Training Run 2 ({cfg2['episodes']} episodes) ===")
    agent2, _ = train(cfg2)

    print("\n✓ Training complete!")
