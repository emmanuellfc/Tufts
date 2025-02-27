# Load modules
from QMSolver import QMSolver
from animation_functions import *

# Solving the SHO
t_steps = 500
resolution = 100
sigma = 1.0  # Ground state width
x0 = 0.0      # Initial center position (equilibrium)
k0 = 0.0      # Initial wave number (ground state)
sim = QMSolver(dt=0.1, dx=0.1, n=resolution, steps=t_steps)
sim.create_grid(-5,5)
sim.gaussian_wave_packet(x0, sigma, k0)
sim.sho_potential()
sim.create_hamiltonian()
sim_solution = sim.solve()

# Animation
time_series_data = generate_time_series_data(sim, t_steps, resolution)
animate_psi_squared_series(time_series_data, t_steps, filename='Simple Harmonic Potential.mp4')
