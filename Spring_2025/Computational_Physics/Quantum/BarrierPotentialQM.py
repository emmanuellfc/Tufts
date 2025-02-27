from QMSolver import QMSolver
from animation_functions import *

# Solving the Square Well Potential
t_steps = 500
resolution = 100
# Gaussian wave packet parameters
x0 = 0.0      # Initial center position
sigma = 1.0   # Width of the wave packet
k0 = 5.0      # Initial wave number
# Potential barrier parameters
v0 = 1.0       # Barrier height
x_left = 2.0   # Left boundary of the barrier
x_right = 4.0  # Right boundary of the barrier
sim = QMSolver(dt=0.1, dx=0.1, n=resolution, steps=t_steps)
sim.create_grid(-5,5)
sim.gaussian_wave_packet(x0, sigma, k0)
sim.potential_barrier(v0, x_left, x_right)
sim.create_hamiltonian()
sim_solution = sim.solve()

# Animation
time_series_data = generate_time_series_data(sim, t_steps, resolution)
animate_psi_squared_series(time_series_data, t_steps, filename='Potential Barrier.mp4')