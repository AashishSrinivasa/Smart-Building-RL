"""
baseline.py — Fixed-Rule Controller for Smart Building
Simulates a naive controller that always uses a fixed action based on time-of-day,
with no learning — used as the comparison baseline against the RL agent.

Fixed Rule:
  - Night hours (0-8, 19-23): action = 0 (all off)
  - Office hours (9-18):       action = 1 (low energy)
  Always uses action regardless of actual occupancy — no adaptation.

Usage (from project root):
    python sim/baseline.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.environment import BuildingEnvironment


class FixedRuleController:
    """
    A simple fixed-timer controller.
    Does NOT learn — always applies the same rule based on hour.
    """
    def choose_action(self, state):
        hour = state[0]
        if 9 <= hour <= 18:
            return 1   # low energy during office hours
        return 0       # off outside office hours


def run_baseline(episodes=1000):
    env = BuildingEnvironment()
    controller = FixedRuleController()

    total_rewards = []
    total_energies = []
    total_comforts = []

    for episode in range(episodes):
        state = env.reset()
        ep_reward = 0
        ep_energy = 0
        ep_comfort = 0
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

        total_rewards.append(ep_reward / 24)
        total_energies.append(ep_energy / 24)
        total_comforts.append(ep_comfort / 24)

    avg_reward  = round(sum(total_rewards)  / episodes, 4)
    avg_energy  = round(sum(total_energies) / episodes, 4)
    avg_comfort = round(sum(total_comforts) / episodes, 4)

    result = {
        "controller":   "FixedRule",
        "episodes":     episodes,
        "avg_reward":   avg_reward,
        "avg_energy":   avg_energy,
        "avg_comfort":  avg_comfort,
    }

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/baseline_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Fixed-Rule Baseline Results:")
    print(f"  Avg Reward  : {avg_reward}")
    print(f"  Avg Energy  : {avg_energy}")
    print(f"  Avg Comfort : {avg_comfort}")
    print("  Saved to experiments/baseline_results.json")

    return result, total_rewards, total_energies


if __name__ == "__main__":
    run_baseline(episodes=1000)
