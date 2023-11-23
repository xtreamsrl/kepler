import numpy as np


class Body:
    def __init__(self, name: str, mass: float, initial_position: np.array, initial_velocity: np.array):
        self.name = name
        self.mass = mass
        self.states = [np.concatenate([initial_position, initial_velocity])]

    @property
    def current_state(self) -> np.array:
        return self.states[-1]

    @property
    def history(self) -> np.array:
        return np.stack(self.states)

    def update_state(self, new_state: np.array):
        self.states.append(new_state)

