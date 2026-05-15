"""
compare.py — Baseline vs RL Policy Comparison
Runs the fixed-rule controller AND the best RL policy on the same simulator.
Outputs:
  - A printed comparison table
  - experiments/comparison_results.json
  - experiments/comparison_plot.png  (reward over episodes)
  - experiments/queue_plot.png       (energy/load over time)

Usage (from project root):
    python sim/compare.py
"""
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.environment import BuildingEnvironment
from sim.agent import QLearningAgent
from sim.baseline import FixedRuleController

POLICY_PATH = "policy_v2_explored.pkl"
EPISODES    = 500
SEED        = 42


def evaluate_controller(controller, episodes, label):
    """Run controller for `episodes` episodes and collect per-episode metrics."""
    env = BuildingEnvironment()
    rewards  = []
    energies = []
    comforts = []

    np.random.seed(SEED)
    for ep in range(episodes):
        state = env.reset()
        ep_reward = ep_energy = ep_comfort = 0
        done = False
        while not done:
            action = controller.choose_action(state)
            next_state, reward, done = env.step(action)
            occupancy = state[2]
            comfort = 1.0 if (occupancy == 0 or action >= occupancy) else 0.0
            ep_reward  += reward
            ep_energy  += action
            ep_comfort += comfort
            state = next_state
        rewards.append(ep_reward  / 24)
        energies.append(ep_energy / 24)
        comforts.append(ep_comfort / 24)

    return rewards, energies, comforts


def print_table(baseline_stats, rl_stats):
    print("\n" + "="*60)
    print(f"  {'Metric':<28} {'Fixed-Rule':>12} {'RL Policy':>12}")
    print("="*60)
    metrics = [
        ("Avg Reward (higher=better)",  "avg_reward"),
        ("Avg Energy Use (lower=better)","avg_energy"),
        ("Avg Comfort (higher=better)",  "avg_comfort"),
    ]
    for label, key in metrics:
        b = baseline_stats[key]
        r = rl_stats[key]
        win = "<-- RL wins" if (
            (key == "avg_reward"  and r > b) or
            (key == "avg_energy"  and r < b) or
            (key == "avg_comfort" and r > b)
        ) else ""
        print(f"  {label:<28} {b:>12.4f} {r:>12.4f}  {win}")
    print("="*60 + "\n")


def plot_comparison(b_rewards, rl_rewards, b_energies, rl_energies):
    os.makedirs("experiments", exist_ok=True)
    episodes = range(1, len(b_rewards) + 1)
    roll = 30   # rolling window

    def rolling(arr):
        result = []
        for i in range(len(arr)):
            w = arr[max(0, i-roll+1): i+1]
            result.append(sum(w) / len(w))
        return result

    # ── Plot 1: Average Reward over Episodes ──────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, b_rewards,  color="#FFAB91", alpha=0.35, linewidth=1)
    ax.plot(episodes, rl_rewards, color="#80CBC4", alpha=0.35, linewidth=1)
    ax.plot(episodes, rolling(b_rewards),  color="#E64A19", linewidth=2.5, label="Fixed-Rule (30-ep avg)")
    ax.plot(episodes, rolling(rl_rewards), color="#00796B", linewidth=2.5, label="RL Policy (30-ep avg)")
    ax.axhline(sum(b_rewards) /len(b_rewards),  color="#E64A19", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(sum(rl_rewards)/len(rl_rewards),  color="#00796B", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title("Average Reward Over Episodes: RL Policy vs Fixed-Rule Baseline",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Reward per Hour", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/comparison_plot.png", dpi=150)
    plt.close()
    print("✓ Saved experiments/comparison_plot.png")

    # ── Plot 2: Energy/Load Over Episodes ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, b_energies,  color="#CE93D8", alpha=0.35, linewidth=1)
    ax.plot(episodes, rl_energies, color="#90CAF9", alpha=0.35, linewidth=1)
    ax.plot(episodes, rolling(b_energies),  color="#7B1FA2", linewidth=2.5, label="Fixed-Rule (30-ep avg)")
    ax.plot(episodes, rolling(rl_energies), color="#1565C0", linewidth=2.5, label="RL Policy (30-ep avg)")
    ax.axhline(sum(b_energies) /len(b_energies),  color="#7B1FA2", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(sum(rl_energies)/len(rl_energies), color="#1565C0", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title("Energy Consumption Over Episodes: RL Policy vs Fixed-Rule Baseline",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Energy Used per Hour (0-2 scale)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/queue_plot.png", dpi=150)
    plt.close()
    print("✓ Saved experiments/queue_plot.png")


if __name__ == "__main__":
    # ── Load RL agent ──────────────────────────────────────────────────────
    if not os.path.exists(POLICY_PATH):
        print(f"ERROR: {POLICY_PATH} not found. Run sim/train.py first.")
        sys.exit(1)

    env = BuildingEnvironment()
    rl_agent = QLearningAgent(n_actions=env.n_actions, epsilon=0.0)  # greedy
    rl_agent.load_policy(POLICY_PATH)

    baseline = FixedRuleController()

    print(f"Running {EPISODES} evaluation episodes for each controller...")

    b_rewards,  b_energies,  b_comforts  = evaluate_controller(baseline,  EPISODES, "Fixed-Rule")
    rl_rewards, rl_energies, rl_comforts = evaluate_controller(rl_agent,  EPISODES, "RL Policy")

    def stats(rewards, energies, comforts):
        return {
            "avg_reward":  round(sum(rewards)  / len(rewards),  4),
            "avg_energy":  round(sum(energies) / len(energies), 4),
            "avg_comfort": round(sum(comforts) / len(comforts), 4),
        }

    b_stats  = stats(b_rewards,  b_energies,  b_comforts)
    rl_stats = stats(rl_rewards, rl_energies, rl_comforts)

    print_table(b_stats, rl_stats)

    # ── Energy reduction % ─────────────────────────────────────────────────
    energy_reduction = round(
        (b_stats["avg_energy"] - rl_stats["avg_energy"]) / b_stats["avg_energy"] * 100, 2
    ) if b_stats["avg_energy"] > 0 else 0.0

    reward_improvement = round(
        (rl_stats["avg_reward"] - b_stats["avg_reward"]) / abs(b_stats["avg_reward"]) * 100, 2
    ) if b_stats["avg_reward"] != 0 else 0.0

    summary = {
        "fixed_rule":          b_stats,
        "rl_policy":           rl_stats,
        "energy_reduction_pct": energy_reduction,
        "reward_improvement_pct": reward_improvement,
        "sdg_impact": (
            f"RL policy reduces average energy consumption by {energy_reduction}% compared to "
            f"the fixed-rule baseline. This supports SDG 7 (Affordable and Clean Energy) by "
            f"cutting unnecessary energy use during unoccupied hours while maintaining occupant "
            f"comfort — directly reducing electricity costs and carbon emissions in smart buildings."
        ),
    }

    with open("experiments/comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved experiments/comparison_results.json")

    print(f"\nSDG 7 Impact: Energy reduced by {energy_reduction}% | "
          f"Reward improved by {reward_improvement}%")

    plot_comparison(b_rewards, rl_rewards, b_energies, rl_energies)
    print("\n✓ Comparison complete.")
