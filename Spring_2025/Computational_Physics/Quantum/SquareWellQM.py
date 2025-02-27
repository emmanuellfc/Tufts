from QMSolver import QMSolver
from animation_functions import *

# Solving the Square Well Potential
t_steps = 200
resolution = 200
x0 = 0.0      # Initial center position
sigma = 1.0   # Width of the wave packet
k0 = 5.0      # Initial wave number
sim = QMSolver(dt=0.1, dx=0.1, n=resolution, steps=t_steps)
sim.create_grid(-5,5)
sim.gaussian_wave_packet(x0, sigma, k0)
sim.sw_potential()
sim.create_hamiltonian()
sim_solution = sim.solve()

# Animation
time_series_data = generate_time_series_data(sim, t_steps, resolution)
animate_psi_squared_series(time_series_data, t_steps, filename='Square Well Potential.mp4')