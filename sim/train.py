import csv
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.environment import BuildingEnvironment
from sim.agent import QLearningAgent

def train(run_id, episodes=1000, alpha=0.1, gamma=0.9, epsilon=1.0):
    env = BuildingEnvironment()
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )

    # Track results
    results = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_energy = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)

            total_reward += reward
            total_energy += action  # action = energy used
            state = next_state

        avg_reward = total_reward / 24
        avg_energy = total_energy / 24

        results.append({
            'run_id': run_id,
            'episode': episode + 1,
            'avg_reward': round(avg_reward, 4),
            'avg_energy': round(avg_energy, 4),
            'epsilon': round(agent.epsilon, 4),
            'alpha': alpha,
            'gamma': gamma
        })

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    # Save policy v1 at episode 500
    # Save policy v2 at episode 1000
    agent.save_policy('policy_v1.pkl') if episodes == 500 else None
    agent.save_policy('policy_v2_explored.pkl')

    # Save results to CSV
    save_results(results)

    return agent, results

def save_results(results):
    filepath = 'experiments/results.csv'
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {filepath}")

if __name__ == "__main__":
    print("=== Training Run 1 (500 episodes) ===")
    agent, results = train(run_id="run_1", episodes=500)

    print("\n=== Training Run 2 (1000 episodes) ===")
    agent, results = train(run_id="run_2", episodes=1000)

    print("\nTraining complete!")