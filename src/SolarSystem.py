import numpy as np

from src import Body
from src.NumericalIntegrationMethods import NumericalIntegrationStrategy


class SolarSystem:
    def __init__(self, planets: list[Body], num_integration_strategy: NumericalIntegrationStrategy):
        self.planets = planets
        self.num_integration_strategy = num_integration_strategy
        self.G = 6.67259e-20

    @property
    def current_state(self) -> np.array:
        return np.stack([p.current_state for p in self.planets])

    def update_state(self, state: np.array):
        for body_idx, body_state in enumerate(state):
            self.planets[body_idx].update_state(body_state)

    def evolve(self, n_steps: int, dt: float):
        tn = 0
        for n in range(n_steps):
            tn += n * dt
            y = self.current_state
            increment = self.num_integration_strategy.compute_increment(y, tn, dt, self.eqm_derivatives)
            self.update_state(y + increment * dt)

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
            ai = self.G * mi * sum([
                (_y[j, 0:3] - ri) / np.linalg.norm(_y[j, 0:3] - ri)**3
            for j in set(range(_y.shape[0])) - {i}
            ])
            derivatives.append(np.concatenate([vi, ai]))

        derivatives = np.stack(derivatives)

        return derivatives




