from pathlib import Path

from src.strategies import RungeKutta4Strategy
from src.solar_system import SolarSystem
from src.visualization import plot_animation, orbit_plot

if __name__ == "__main__":
    n_steps = 100
    dt = 0.1

    solar_system = SolarSystem()
    solar_system.load_data(Path("data/planets_data.json"))

    tn = 0
    for n in range(n_steps):
        if n % 500 == 0:
            print(f"{n}/{n_steps} steps calculated")
        tn += n * dt
        y = solar_system.current_state
        increment = RungeKutta4Strategy().compute_increment(y, tn, dt, solar_system.eqm_derivatives)
        solar_system.update_state(y + increment * dt)

    plot_animation(solar_system.planets)
