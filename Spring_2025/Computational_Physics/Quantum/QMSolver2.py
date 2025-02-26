import numpy as np
import matplotlib.pyplot as plt

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

    def create_potential(self):
        return self.potential

    def create_hamiltonian(self):
        """
        Constructs the Hamiltonian matrix
        Returns:
            - Square matrix of the Hamiltonian
        """
        h_bar, m, omega = 1, 1, 1
        # Construct the Hamiltonian matrix
        c: float = - h_bar ** 2 / (2 * m * self.dx ** 2)
        self.hamiltonian = np.zeros((self.n, self.n), dtype=complex)
        for i in range(1, self.n - 1):
            self.hamiltonian[i, i] = 2 * c + 0.5*self.potential[i]**2
            self.hamiltonian[i, i] = 2 * c 
            self.hamiltonian[i, i + 1] = -c
            self.hamiltonian[i, i - 1] = -c
        # Set Dirichlet Boundary Conditions
        self.hamiltonian[0, 0]   = 1e30
        self.hamiltonian[-1, -1] = 1e30
        return self.hamiltonian



sim = QMSolver(dt=0.1, dx=0.1, n=200, steps=200)
