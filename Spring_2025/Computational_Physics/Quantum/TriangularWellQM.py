from QMSolver import QMSolver
from helper_functions import *

# Solving the Square Well Potential
t_steps = 500
resolution = 100
x0 = 0.0      # Initial center position
sigma = 2.0   # Width of the wave packet
k0 = 0.0      # Initial wave number
sim = QMSolver(dt=0.1, dx=0.1, n=resolution, steps=t_steps)
sim.create_grid(-5,5)
sim.gaussian_wave_packet(x0, sigma, k0)
sim.tw_potential()
sim.create_hamiltonian_cn()
sim_solution = sim.solve_crank_nicolson()

# Animation
time_series_psi_sq = generate_time_series_psi_squared(sim, t_steps)
f_name_psi_sq = 'Triangular Potential (CN Scheme)'
label_sq = r"$|\psi|^2$"
animate_psi(time_series_psi_sq, t_steps, filename=f_name_psi_sq, label=label_sq, color = "magenta")

time_series_psi_real =generate_time_series_psi_real(sim, t_steps)
f_name_psi_re = 'Triangular Potential Re Part (CN Scheme)'
label_re = r"Re($\psi$)"
animate_psi(time_series_psi_real, t_steps, filename=f_name_psi_re, label=label_re, color = "red")

time_series_psi_imag =generate_time_series_psi_imag(sim, t_steps)
f_name_psi_im = 'Triangular Potential Im Part (CN Scheme)'
label_im = r"Im($\psi$)"
animate_psi(time_series_psi_imag, t_steps, filename=f_name_psi_im, label=label_im, color = "blue")