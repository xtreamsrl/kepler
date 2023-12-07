from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


# STRATEGY PATTERN
class NumericalIntegrationStrategy(ABC):
    @abstractmethod
    def compute_increment(
            self,  y: np.array, t: float, dt: float, derivatives: Callable[[np.array, float], np.array]
    ) -> np.array:
        """A method that compute increments of the numerical strategy"""


class EulerStrategy(NumericalIntegrationStrategy):
    def compute_increment(
            self, y: np.array, t: float, dt: float, derivatives: Callable[[np.array, float], np.array]
    ) -> np.array:
        f = derivatives(y, t)
        return f


class RungeKutta4Strategy(NumericalIntegrationStrategy):
    def compute_increment(
            self, y: np.array, t: float, dt: float, derivatives: Callable[[np.array, float], np.array]
    ) -> np.array:
        f1 = derivatives(y, t)
        f2 = derivatives(y + f1 * dt / 2, t + dt / 2)
        f3 = derivatives(y + f2 * dt / 2, t + dt / 2)
        f4 = derivatives(y + f3 * dt, t + dt)
        return (f1 + 2 * f2 + 2 * f3 + f4) / 6
