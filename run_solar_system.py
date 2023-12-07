from pathlib import Path

from src.strategies import RungeKutta4Strategy, NumericalIntegrationStrategy
from src.solar_system import SolarSystem
from src.system_interface import SystemInterface
from src.utils import load_planets_from_json
from src.visualization import plot_animation, plot_orbits


def evolve(system: SystemInterface, evolving_strategy: NumericalIntegrationStrategy):
    tn = 0
    for n in range(n_steps):
        if n % 500 == 0:
            print(f"{n}/{n_steps} steps calculated")
        tn += n * dt
        y = system.current_state
        increment = evolving_strategy.compute_increment(y, tn, dt, system.state_derivatives)
        system.update_state(y + increment * dt)


if __name__ == "__main__":
    n_steps = 10000
    dt = 60*60*24  # s

    data_path = Path("data/planets_data.json")
    planets = load_planets_from_json(data_path)

    solar_system = SolarSystem(planets)
    evolving_strategy = RungeKutta4Strategy()

    evolve(solar_system, evolving_strategy)

    plot_animation(solar_system.planets)
