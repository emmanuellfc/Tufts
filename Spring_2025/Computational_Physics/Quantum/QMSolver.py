# Main Class Object Containing
import numpy as np
import matplotlib.pyplot as plt

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
             'legend.frameon'  : True,     # activate frame on lengend?
            }
plt.rcParams.update(newParams) # Set new plotting parameters

class QMSolver:
    """
    Initialize the QM solver
        Parameters
            - dt:
            - dx:
            - n:
            - steps:
    """
    def __init__(self, dt: float, dx: float, n: int, steps: int) -> None:
        self.dt = dt
        self.dx = dx
        self.N = n
        self.steps = steps
        self.x = None
        self.psi = None
        self.psi_total = None
        self.potential = None
        self.hamiltonian = None

    def create_potential(self):
        self.x = np.linspace(0, self.N * self.dx, self.N)
        # Potential: infinite square well
        self.potential = np.zeros(self.N)
        self.potential[0] = np.inf
        self.potential[-1] = np.inf

    def create_hamiltonian(self):
        """

        Returns:
            Hamiltonian

        """
        h_bar, m = 1, 1

        # Make Hamiltonian Matrix (Complex Matrix)
        c = -h_bar ** 2 / (2 * m * self.dx ** 2)
        self.hamiltonian = np.zeros((self.N, self.N), dtype=complex)
        for i in range(1, self.N - 1):
            self.hamiltonian[i, i] = 2 * c + self.potential[i]
            self.hamiltonian[i, i + 1] = -c
            self.hamiltonian[i, i - 1] = -c

        # Set Dirichlet Boundary Conditions
        self.hamiltonian[0, 0] = 1e30
        self.hamiltonian[-1, -1] = 1e30
        return self.hamiltonian

    def initial_condition(self):
        """

        Returns:
            Initial condition

        """
        self.psi = np.exp(-((self.x - self.N * self.dx / 2) ** 2) / 2)
        self.psi = self.psi / np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)  # Normalized Gaussian
        return self.psi

    def solve(self):
        """

        Returns:
            Wave function at at all times

        """
        I = np.identity(self.N, dtype=complex)
        # Create matrices A and B
        A = I + (1j / 2) * self.hamiltonian * self.dt
        B = I - (1j / 2) * self.hamiltonian * self.dt
        # Time evolution

        self.psi_total = []
        for n in range(self.steps):
            # Solve the linear system
            self.psi = np.linalg.solve(A, B @ self.psi)

            # Normalize the wave-function (important for stability)
            self.psi = self.psi / np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)

            self.psi_total.append(self.psi)

        return self.psi_total

test  = QMSolver(dt=0.001, dx=0.1, n=100, steps=1000)
test.create_potential()
test.initial_condition()
test.create_hamiltonian()
sols = test.solve()

# Plot Solution
plt.figure()
plt.plot(test.x, np.real(sols[0]), label='Re(psi) @ t = t0')
plt.plot(test.x, np.real(sols[-1]), label='Re(psi) @ t = tf')
plt.legend()
plt.show()

