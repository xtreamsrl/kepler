from pathlib import Path

from matplotlib import pyplot as plt

from src.strategies import RungeKutta4Strategy
from src.solar_system import SolarSystem
from src.visualization import plot_animation

if __name__ == "__main__":

    n_steps = 10000
    dt = 0.001

    solar_system = SolarSystem()
    solar_system.load_data(Path("bodies_data.json"))

    tn = 0
    for n in range(n_steps):
        tn += n * dt
        y = solar_system.current_state
        increment = RungeKutta4Strategy().compute_increment(y, tn, dt, solar_system.eqm_derivatives)
        solar_system.update_state(y + increment * dt)

    anim = plot_animation(solar_system.planets)
    anim.save('animation.gif', fps=10)

    plt.show()
