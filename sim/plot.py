import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results - skip first row and use header
df = pd.read_csv('experiments/results.csv')

# Convert numeric columns to float (coerce errors to NaN)
df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
df['avg_reward'] = pd.to_numeric(df['avg_reward'], errors='coerce')
df['avg_waiting_time'] = pd.to_numeric(df['avg_waiting_time'], errors='coerce')

# Remove rows with NaN values (removes the duplicate header row)
df = df.dropna(subset=['episode', 'avg_reward', 'avg_waiting_time'])

# Filter run_2 (1000 episodes)
run2 = df[df['run_id'] == 'run_2'].reset_index(drop=True)

# ===== PLOT 1: AGENT LEARNING - AVERAGE REWARD =====
plt.figure(figsize=(12, 6))

# Plot actual rewards
plt.plot(run2['episode'], run2['avg_reward'], color='lightblue', alpha=0.5, linewidth=1, label='Episode Reward')

# Plot 50-episode moving average (shows trend)
rolling_avg = run2['avg_reward'].rolling(window=50, min_periods=1).mean()
plt.plot(run2['episode'], rolling_avg, color='darkblue', linewidth=3, label='50-Episode Average')

# Add mean line
mean_reward = run2['avg_reward'].mean()
plt.axhline(y=mean_reward, color='green', linestyle='--', linewidth=2, label=f'Overall Average: {mean_reward:.3f}')

plt.title('Agent Learning Progress: Reward Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Episode Number', fontsize=12)
plt.ylabel('Reward Score', fontsize=12)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/reward_plot.png', dpi=150)
plt.close()
print("✓ Reward plot saved to experiments/reward_plot.png")
print(f"  → Min Reward: {run2['avg_reward'].min():.4f}")
print(f"  → Max Reward: {run2['avg_reward'].max():.4f}")
print(f"  → Average Reward: {mean_reward:.4f}")


# ===== PLOT 2: ENERGY CONSUMPTION =====
plt.figure(figsize=(12, 6))

# Plot actual energy
plt.plot(run2['episode'], run2['avg_waiting_time'], color='lightsalmon', alpha=0.5, linewidth=1, label='Episode Energy')

# Plot 50-episode moving average
rolling_energy = run2['avg_waiting_time'].rolling(window=50, min_periods=1).mean()
plt.plot(run2['episode'], rolling_energy, color='darkred', linewidth=3, label='50-Episode Average')

# Add mean line
mean_energy = run2['avg_waiting_time'].mean()
plt.axhline(y=mean_energy, color='orange', linestyle='--', linewidth=2, label=f'Overall Average: {mean_energy:.3f}')

plt.title('Energy Consumption Over Training', fontsize=16, fontweight='bold')
plt.xlabel('Episode Number', fontsize=12)
plt.ylabel('Energy Usage (units)', fontsize=12)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/energy_plot.png', dpi=150)
plt.close()
print("✓ Energy plot saved to experiments/energy_plot.png")
print(f"  → Min Energy: {run2['avg_waiting_time'].min():.4f}")
print(f"  → Max Energy: {run2['avg_waiting_time'].max():.4f}")
print(f"  → Average Energy: {mean_energy:.4f}")