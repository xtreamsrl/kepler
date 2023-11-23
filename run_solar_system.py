import numpy as np

from src.Body import Body
from src.NumericalIntegrationMethods import RungeKutta4Strategy
from src.SolarSystem import SolarSystem

if __name__ == "__main__":
    Body1 = Body(name="body1",
                 mass=1e26,  # (kg)
                 initial_position=np.array([2E4, 0, 0]),  # (km)
                 initial_velocity=np.array([0, -10, 0])  # (km/s)
                 )

    Body2 = Body(name="body2",
                 mass=1e26,  # (kg)
                 initial_position=np.array([-2E4, 0, 0]),  # (km)
                 initial_velocity=np.array([0, 10, 0])  # (km/s)
                 )

    Body3 = Body(name="body3",
                 mass=1e31,  # (kg)
                 initial_position=np.array([0, 0, 0]),  # (km)
                 initial_velocity=np.array([0, 0, 0])  # (km/s)
                 )

    G = 6.67259e-20  # Gravitational constant (km**3/kg/s**2)
    bodies = [Body1, Body2, Body3]

    solar_system = SolarSystem(
        planets=bodies,
        num_integration_strategy=RungeKutta4Strategy()
    )
    solar_system.evolve(n_steps=100, dt=0.001)

    for p in solar_system.planets:
        print(p.name, p.states)