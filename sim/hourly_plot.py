"""
hourly_plot.py — Energy Load & Comfort Over Time (per hour of day)
Equivalent to "queue length over time" for the smart building domain.

Runs the trained RL agent and the fixed-rule baseline for multiple episodes,
then averages the per-hour energy and comfort values to show how each
controller behaves over the 24-hour building cycle.

Usage (from project root):
    python sim/hourly_plot.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.environment import BuildingEnvironment
from sim.agent import QLearningAgent
from sim.baseline import FixedRuleController

POLICY_PATH = "policy_v2_explored.pkl"
EVAL_EPISODES = 200
HOURS = 24


def collect_hourly(controller, episodes):
    """Returns arrays of shape (HOURS,) with mean energy and comfort per hour."""
    env = BuildingEnvironment()
    hour_energy  = np.zeros((episodes, HOURS))
    hour_comfort = np.zeros((episodes, HOURS))

    np.random.seed(42)
    for ep in range(episodes):
        state = env.reset()
        done  = False
        step  = 0
        while not done:
            action = controller.choose_action(state)
            occupancy = state[2]
            comfort = 1.0 if (occupancy == 0 or action >= occupancy) else 0.0
            hour_energy[ep, step]  = action
            hour_comfort[ep, step] = comfort
            state, _, done = env.step(action)
            step += 1

    return hour_energy.mean(axis=0), hour_comfort.mean(axis=0)


if __name__ == "__main__":
    if not os.path.exists(POLICY_PATH):
        print(f"ERROR: {POLICY_PATH} not found. Run sim/train.py first.")
        sys.exit(1)

    env = BuildingEnvironment()
    rl_agent = QLearningAgent(n_actions=env.n_actions, epsilon=0.0)  # greedy
    rl_agent.load_policy(POLICY_PATH)
    baseline = FixedRuleController()

    print(f"Collecting hourly data over {EVAL_EPISODES} episodes...")
    b_energy,  b_comfort  = collect_hourly(baseline,  EVAL_EPISODES)
    rl_energy, rl_comfort = collect_hourly(rl_agent,  EVAL_EPISODES)

    hours = list(range(HOURS))
    os.makedirs("experiments", exist_ok=True)

    # ── Plot: Energy Load Over Time (hours 0-23) ──────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Top: energy load per hour
    ax1.fill_between(hours, b_energy,  alpha=0.25, color="#E64A19")
    ax1.fill_between(hours, rl_energy, alpha=0.25, color="#00796B")
    ax1.plot(hours, b_energy,  color="#E64A19", linewidth=2.5,
             marker="o", markersize=5, label="Fixed-Rule Baseline")
    ax1.plot(hours, rl_energy, color="#00796B", linewidth=2.5,
             marker="s", markersize=5, label="RL Policy")

    # Shade office hours
    ax1.axvspan(9, 18, alpha=0.08, color="gold", label="Office Hours (9-18)")
    ax1.set_ylabel("Avg Energy Used per Hour\n(0=off, 1=low, 2=high)", fontsize=11)
    ax1.set_title("Building Energy Load Over Time: RL Policy vs Fixed-Rule Baseline\n"
                  f"(averaged over {EVAL_EPISODES} episodes)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 2.1)

    # Bottom: comfort score per hour
    ax2.fill_between(hours, b_comfort,  alpha=0.25, color="#E64A19")
    ax2.fill_between(hours, rl_comfort, alpha=0.25, color="#00796B")
    ax2.plot(hours, b_comfort,  color="#E64A19", linewidth=2.5,
             marker="o", markersize=5, label="Fixed-Rule Baseline")
    ax2.plot(hours, rl_comfort, color="#00796B", linewidth=2.5,
             marker="s", markersize=5, label="RL Policy")
    ax2.axvspan(9, 18, alpha=0.08, color="gold", label="Office Hours (9-18)")
    ax2.set_ylabel("Avg Comfort Score per Hour\n(1=satisfied, 0=unsatisfied)", fontsize=11)
    ax2.set_xlabel("Hour of Day", fontsize=12)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xticks(hours)

    plt.tight_layout()
    plt.savefig("experiments/hourly_load_plot.png", dpi=150)
    plt.close()
    print("✓ Saved experiments/hourly_load_plot.png")

    # Print summary table
    print(f"\nHour-by-hour summary (avg across {EVAL_EPISODES} episodes):")
    print(f"{'Hour':>5} | {'Baseline Energy':>15} | {'RL Energy':>10} | {'Baseline Comfort':>16} | {'RL Comfort':>10}")
    print("-" * 65)
    for h in hours:
        print(f"{h:>5} | {b_energy[h]:>15.3f} | {rl_energy[h]:>10.3f} | {b_comfort[h]:>16.3f} | {rl_comfort[h]:>10.3f}")
