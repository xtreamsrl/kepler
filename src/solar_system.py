from pathlib import Path

import numpy as np
import json
from src.body import Body

G = 6.67259e-20


class SolarSystem:
    def __init__(self, planets: list[Body] = None):
        self.planets = planets

    def load_data(self, data_path: Path):
        with open(data_path, "r") as f:
            data_dict = json.load(f)
            planets = []
            for body_name, body_data in data_dict.items():
                planets.append(
                    Body(
                        name=body_name,
                        mass=body_data["mass"],
                        initial_position=np.array(body_data["initial_position"]),
                        initial_velocity=np.array(body_data["initial_velocity"]),
                    )
                )
            self.planets = planets

    @property
    def current_state(self) -> np.array:
        return np.stack([p.current_state for p in self.planets])

    def update_state(self, state: np.array):
        for body_idx, body_state in enumerate(state):
            self.planets[body_idx].update_state(body_state)

    def eqm_derivatives(self, _y: np.array, t: float) -> np.array:
        """
        derivatives of the equations of motion describing the n-body system
        t is unused
        """
        derivatives = []
        for i in range(_y.shape[0]):
            ri = _y[i, 0:3]
            mi = self.planets[i].mass
            vi = _y[i, 3:6]

            # acceleration
            ai = G * mi * sum([
                (_y[j, 0:3] - ri) / np.linalg.norm(_y[j, 0:3] - ri) ** 3
                for j in set(range(_y.shape[0])) - {i}
            ])
            derivatives.append(np.concatenate([vi, ai]))

        derivatives = np.stack(derivatives)

        return derivatives
