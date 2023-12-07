from abc import abstractmethod
from pathlib import Path

import numpy as np


class SystemInterface:
    @property
    @abstractmethod
    def current_state(self) -> np.array:
        """Get the state of the system at the moment of the call"""

    @abstractmethod
    def update_state(self, state: np.array):
        """Update the current state of the system"""

    @abstractmethod
    def state_derivatives(self, state: np.array, t: float) -> np.array:
        """
        State derivatives of the system
        """