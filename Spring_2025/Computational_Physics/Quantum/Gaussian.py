from QMSolver import QMSolver
from helper_functions import *

# Solving the Square Well Potential
t_steps = 200
resolution = 200
x0 = 0.0      # Initial center position
sigma = 1.0   # Width of the wave packet
k0 = 5.0      # Initial wave number
sim = QMSolver(dt=1, dx=1, n=resolution, steps=t_steps)
sim.create_grid(-5,5)
sim.gaussian_wave(x0=5)
sim.sw_potential()
sim.create_hamiltonian_fd()
sim_solution = sim.solve_finite_difference()

# Animation
# time_series_psi_sq = generate_time_series_psi_squared(sim, t_steps)
# f_name_psi_sq = 'Square Well Potential (FD Scheme)'
# label_sq = r"$|\psi|^2$"
# animate_psi(time_series_psi_sq, t_steps, filename=f_name_psi_sq, label=label_sq, color = "magenta")
#
# time_series_psi_real =generate_time_series_psi_real(sim, t_steps)
# f_name_psi_re = 'Square Well Potential Re Part (FD Scheme)'
# label_re = r"Re($\psi$)"
# animate_psi(time_series_psi_real, t_steps, filename=f_name_psi_re, label=label_re, color = "red")
#
# time_series_psi_imag =generate_time_series_psi_imag(sim, t_steps)
# f_name_psi_im = 'Square Well Potential Im Part (FD Scheme)'
# label_im = r"Im($\psi$)"
# animate_psi(time_series_psi_imag, t_steps, filename=f_name_psi_im, label=label_im, color = "blue")

# Check probability
for i in range(t_steps):
    if i % 10 == 0:
        print(check_norm(sim_solution[i], t_steps, dx=1))