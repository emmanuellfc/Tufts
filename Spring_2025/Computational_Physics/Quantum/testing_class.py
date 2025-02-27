from QMSolver import QMSolver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Solving the SHO

t_steps = 100
resolution = 200
sim = QMSolver(dt=0.1, dx=0.1, n=resolution, steps=t_steps)
sim.create_grid(-5,5)
sim.initial_condition()
sim.tw_potential()
sim.create_hamiltonian()
sim_solution = sim.solve()

# Animation
def generate_time_series_data(solution, num_time_steps, num_points):
    """Generates sample time-series graph data."""
    data = []
    for t in range(num_time_steps):
        data.append(np.column_stack((solution.x, np.real(solution.psi_total[t]))))
    return np.array(data)

time_series_data = generate_time_series_data(sim, t_steps, resolution)

# Set up the figure and axes
fig, ax = plt.subplots()
line, = ax.plot(time_series_data[0, :, 0], time_series_data[0, :, 1], label = "SHO")  # Initial plot
ax.set_xlim(np.min(time_series_data[:,:,0]), np.max(time_series_data[:,:,0]))
ax.set_ylim(np.min(time_series_data[:,:,1]), np.max(time_series_data[:,:,1]))
ax.set_xlabel("x")
ax.set_ylabel(r"Re($\psi(t)$)")

# Animation function
def animate(i):
    line.set_xdata(time_series_data[i, :, 0])
    line.set_ydata(time_series_data[i, :, 1])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=t_steps, interval=1, blit=True)

# Display or save the animation
ani.save('triangle_well_potential.mp4', writer='ffmpeg', fps=30)
