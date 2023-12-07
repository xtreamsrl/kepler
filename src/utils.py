import json
from pathlib import Path

import numpy as np

from src.body import Body


def compute_relative_marker_size(m: float, max_m: float):
    return 30 - 3 * (np.log10(max_m) - np.log10(m))


def load_planets_from_json(data_path: Path) -> list[Body]:
    with open(data_path, "r") as f:
        data_dict = json.load(f)
        planets = [
            Body(
                name=body_name,
                mass=body_data["mass"],
                radius=body_data["radius"],
                initial_position=np.array(body_data["initial_position"]),
                initial_velocity=np.array(body_data["initial_velocity"])
            ) for body_name, body_data in data_dict.items()
        ]

    return planets
