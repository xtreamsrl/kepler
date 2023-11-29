import numpy as np


def compute_relative_marker_size(m: float, max_m: float):
    return 30 - 3*(np.log10(max_m) - np.log10(m))