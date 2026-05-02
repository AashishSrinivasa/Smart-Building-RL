import numpy as np

class BuildingEnvironment:
    def __init__(self):
        # State space
        self.hours = 24          # 0-23
        self.temp_levels = 3     # 0=cold, 1=mild, 2=hot
        self.occupancy_levels = 3 # 0=empty, 1=medium, 2=full
        self.energy_levels = 3   # 0=low, 1=medium, 2=high

        # Action space
        # 0 = all off, 1 = low, 2 = high
        self.n_actions = 3

        self.reset()

    def reset(self):
        self.hour = 0
        self.temp = np.random.randint(0, 3)
        self.occupancy = np.random.randint(0, 3)
        self.energy = 0
        return self._get_state()

    def _get_state(self):
        return (self.hour, self.temp, self.occupancy, self.energy)

    def step(self, action):
        # Energy consumed based on action
        energy_used = action  # 0, 1, or 2

        # Comfort score — people need AC/heat when present
        if self.occupancy == 0:
            comfort = 1.0  # nobody there, any action is fine
        else:
            comfort = 1.0 if action >= self.occupancy else 0.0

        # Reward = comfort - energy cost
        reward = comfort - (0.5 * energy_used)

        # Update state
        self.energy = min(energy_used, 2)
        self.hour = (self.hour + 1) % 24

        # Randomly change temp and occupancy each step
        self.temp = np.random.randint(0, 3)
        self.occupancy = self._get_occupancy()

        done = self.hour == 0  # episode ends after 24 hours
        return self._get_state(), reward, done

    def _get_occupancy(self):
        # Office hours = more people
        if 9 <= self.hour <= 18:
            return np.random.choice([1, 2], p=[0.3, 0.7])
        else:
            return np.random.choice([0, 1], p=[0.8, 0.2])