"""
plot.py — Training curve plots for the RL agent
Usage (from project root):
    python sim/plot.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = "experiments/results.csv"

df = pd.read_csv(RESULTS_CSV)

# Remove any duplicate header rows that may exist
df = df[df["episode"] != "episode"].copy()
df["episode"]     = pd.to_numeric(df["episode"],     errors="coerce")
df["avg_reward"]  = pd.to_numeric(df["avg_reward"],  errors="coerce")
df["avg_energy"]  = pd.to_numeric(df["avg_energy"],  errors="coerce")
df["avg_comfort"] = pd.to_numeric(df.get("avg_comfort", 0), errors="coerce")
df = df.dropna(subset=["episode", "avg_reward", "avg_energy"])

# Use run_2 (longer training) for plots
run2 = df[df["run_id"] == "run_2"].reset_index(drop=True)

os.makedirs("experiments", exist_ok=True)

# ── Plot 1: Reward over training ─────────────────────────────────────────────
plt.figure(figsize=(12, 6))
plt.plot(run2["episode"], run2["avg_reward"],
         color="lightblue", alpha=0.45, linewidth=1, label="Episode Reward")
rolling = run2["avg_reward"].rolling(window=50, min_periods=1).mean()
plt.plot(run2["episode"], rolling,
         color="darkblue", linewidth=3, label="50-Episode Moving Average")
mean_r = run2["avg_reward"].mean()
plt.axhline(mean_r, color="green", linestyle="--", linewidth=2,
            label=f"Overall Average: {mean_r:.3f}")
plt.title("Agent Learning Progress: Reward Over Training Episodes",
          fontsize=15, fontweight="bold")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Average Reward per Hour", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/reward_plot.png", dpi=150)
plt.close()
print(f"✓ Saved experiments/reward_plot.png  (mean={mean_r:.4f})")

# ── Plot 2: Energy consumption over training ─────────────────────────────────
plt.figure(figsize=(12, 6))
plt.plot(run2["episode"], run2["avg_energy"],
         color="lightsalmon", alpha=0.45, linewidth=1, label="Episode Energy")
rolling_e = run2["avg_energy"].rolling(window=50, min_periods=1).mean()
plt.plot(run2["episode"], rolling_e,
         color="darkred", linewidth=3, label="50-Episode Moving Average")
mean_e = run2["avg_energy"].mean()
plt.axhline(mean_e, color="orange", linestyle="--", linewidth=2,
            label=f"Overall Average: {mean_e:.3f}")
plt.title("Energy Consumption During Training",
          fontsize=15, fontweight="bold")
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Avg Energy Used per Hour (0-2 scale)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experiments/energy_plot.png", dpi=150)
plt.close()
print(f"✓ Saved experiments/energy_plot.png  (mean={mean_e:.4f})")
