from pathlib import Path

import numpy as np
import json
from src.body import Body
from src.system_interface import SystemInterface

G = 6.67e-11  # Gravitational constant (m**3/kg/s**2)


class SolarSystem(SystemInterface):
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
                        radius=body_data["radius"],
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

    def state_derivatives(self, state: np.array, t: float) -> np.array:
        """
        derivatives of the equations of motion describing the n-body system
        t is unused
        """
        masses = [planet.mass for planet in self.planets]
        derivatives = []
        for i in range(state.shape[0]):
            ri = state[i, 0:3]
            vi = state[i, 3:6]
            # acceleration
            ai = G * sum([
                masses[j] * (state[j, 0:3] - ri) / np.linalg.norm(state[j, 0:3] - ri) ** 3
                for j in set(range(state.shape[0])) - {i}
            ])
            derivatives.append(np.concatenate([vi, ai]))

        derivatives = np.stack(derivatives)

        return derivatives
