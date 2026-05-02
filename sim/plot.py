import pandas as pd
import matplotlib.pyplot as plt

# Load results with column names
df = pd.read_csv('experiments/results.csv', header=None,
                 names=['run_id','episode','avg_reward','avg_energy','epsilon','alpha','gamma'])

# Filter run_2 (1000 episodes)
run2 = df[df['run_id'] == 'run_2']

# Plot avg reward over episodes
plt.figure(figsize=(10, 5))
plt.plot(run2['episode'], run2['avg_reward'], color='blue', alpha=0.6)
plt.title('Average Reward over Episodes')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid(True)
plt.savefig('experiments/reward_plot.png')
plt.show()
print("Plot saved to experiments/reward_plot.png")


# Plot 2 — avg energy over episodes
plt.figure(figsize=(10, 5))
plt.plot(run2['episode'], run2['avg_waiting_time'], color='red', alpha=0.6)
plt.title('Average Energy Usage over Episodes')
plt.xlabel('Episode')
plt.ylabel('Average Energy Usage')
plt.grid(True)
plt.savefig('experiments/energy_plot.png')
plt.show()
print("Energy plot saved to experiments/energy_plot.png")