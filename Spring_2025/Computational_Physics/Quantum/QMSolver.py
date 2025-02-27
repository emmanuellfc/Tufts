# Load Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set figure parameters for all plots
newParams = {'figure.figsize'  : (12, 6),  # Figure size
             'figure.dpi'      : 200,      # figure resolution
             'axes.titlesize'  : 20,       # fontsize of title
             'axes.labelsize'  : 11,       # fontsize of axes labels
             'axes.linewidth'  : 2,        # width of the figure box lines
             'lines.linewidth' : 1,        # width of the plotted lines
             'savefig.dpi'     : 200,      # resolution of a figured saved using plt.savefig(filename)
             'ytick.labelsize' : 11,       # fontsize of tick labels on y axis
             'xtick.labelsize' : 11,       # fontsize of tick labels on x axis
             'legend.fontsize' : 12,       # fontsize of labels in legend
             'legend.frameon'  : True,     # activate frame on lengend
            }
plt.rcParams.update(newParams) # Set new plotting parameters

# Main Class Definition
class QMSolver:
    """
    A class for solving the time-dependent Schrödinger equation using the Crank-Nicolson scheme.

    Parameters:
        dt (float): Time step.
        dx (float): Spatial step.
        n (int): Number of grid points.
        steps (int): Number of time steps.
    """
    def __init__(self, dt: float, dx: float, n: int, steps: int):
        """
        Args:
            dt: delta t
            dx: delta x
            n: number of grid points
            steps: number of time steps (time-evolution)
        """
        self.dt = dt
        self.dx = dx
        self.n = n
        self.steps = steps
        self.x = None
        self.psi = None
        self.psi_total = None
        self.potential = None
        self.hamiltonian = None

    def create_grid(self, x_min: float, x_max: float):
        """
        Create Space Grid
        Args:
            x_min:
            x_max:
        Returns:
            - Numpy array
        """
        self.x = np.linspace(x_min, x_max, self.n)
        return self.x

    def initial_condition(self):
        """
        Create initial condition for psi
        Returns:
            - Numpy array with initial condition
        """
        self.psi = (1/np.pi**0.25) * np.exp(-0.5*self.x**2)

    def sho_potential(self):
        """
        Create SHO potential
        Returns:
            - Numpy array with harmonic potential
        """
        h_bar, m, omega = 1, 1, 1
        self.potential = 0.5 * self.x ** 2
        return self.potential

    def create_hamiltonian(self):
        """
        Constructs the Hamiltonian matrix
        Returns:
            - Square matrix of the Hamiltonian
        """
        # Construct the Hamiltonian matrix
        c: float = - 1 / (2 * self.dx ** 2)
        self.hamiltonian = np.zeros((self.n, self.n), dtype=complex)
        for i in range(1, self.n - 1):
            self.hamiltonian[i, i] = 2 * c + self.potential[i]
            self.hamiltonian[i, i] = 2 * c 
            self.hamiltonian[i, i + 1] = -c
            self.hamiltonian[i, i - 1] = -c
        # Set Dirichlet Boundary Conditions
        self.hamiltonian[0, 0]   = 1e30
        self.hamiltonian[-1, -1] = 1e30
        return self.hamiltonian

    def solve(self):
        """
        Returns:
            Wave function at all times
        """
        identity_matrix = np.identity(self.n, dtype=complex)
        # Create matrices A and B
        forward_matrix  = identity_matrix + (1j / 2) * self.hamiltonian * self.dt
        backward_matrix = identity_matrix - (1j / 2) * self.hamiltonian * self.dt
        # Time evolution
        self.psi_total = []
        for n in range(self.steps):
            # Solve the linear system
            self.psi = np.linalg.solve(forward_matrix, backward_matrix @ self.psi)
            # Normalize the wave-function
            self.psi = self.psi / np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)
            # Append the wave function to psi total
            self.psi_total.append(self.psi)
        return self.psi_total

# Solving the SHO
t_steps = 100
resolution = 200
sim = QMSolver(dt=0.1, dx=0.1, n=resolution, steps=t_steps)
sim.create_grid(-5,5)
sim.initial_condition()
sim.sho_potential()
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
ani.save('simple_harmonic_oscillator.mp4', writer='ffmpeg', fps=30)

