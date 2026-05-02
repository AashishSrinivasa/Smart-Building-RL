import numpy as np
import pickle

class QLearningAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        # Hyperparameters
        self.n_actions = n_actions
        self.alpha = alpha        # learning rate
        self.gamma = gamma        # discount factor
        self.epsilon = epsilon    # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        # Q-table — dictionary because state is a tuple
        self.q_table = {}

    def get_q(self, state):
        # If state not seen before, initialize with zeros
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # explore
        return np.argmax(self.get_q(state))           # exploit

    def update(self, state, action, reward, next_state):
        current_q = self.get_q(state)[action]
        next_max_q = np.max(self.get_q(next_state))

        # Q-learning formula
        new_q = current_q + self.alpha * (
            reward + self.gamma * next_max_q - current_q
        )
        self.q_table[state][action] = new_q

        # Decay epsilon after each update
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_policy(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Policy saved to {filename}")

    def load_policy(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Policy loaded from {filename}")