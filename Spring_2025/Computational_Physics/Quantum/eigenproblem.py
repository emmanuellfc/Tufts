import numpy as np
import matplotlib.pyplot as plt

class EigenProblem:
    def __init__(self, N, x_max):
        self.N = N
        self.x_max = x_max
        self.hamiltonian = None

    def construct_hamiltonian(self, x, potential):
        hbar, m, omega = 1, 1, 1
        # Construct the Hamiltonian matrix
        self.hamiltonian = np.zeros((N, N))
        for i in range(N):
            self.hamiltonian[i, i] = hbar ** 2 / (m * dx ** 2) + potential(N, x[i])
            if i > 0:
                self.hamiltonian[i, i - 1] = -hbar ** 2 / (2 * m * dx ** 2)
            if i < N - 1:
                self.hamiltonian[i, i + 1] = -hbar ** 2 / (2 * m * dx ** 2)
        return self.hamiltonian

class Potentials:
    def __init__(self, N, x_max):
        self.N = N
        self.x_max = x_max
        self.potential = None

    def sho(self):
        hbar, m, omega = 1, 1, 1
        self.potential = 0.5 * m * omega ** 2 * x ** 2
        return

# Parameters
N = 1000  # Number of grid points
x_max = 5  # Maximum x value (adjust as needed)
dx = 2 * x_max / N
x = np.linspace(-x_max, x_max, N)

eigen = EigenProblem(N, x_max)
pot = Potentials(N, x_max)
sho = pot.sho()
H = eigen.construct_hamiltonian(x, sho)
# Solve for eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(H)

# Sort eigenvalues and eigenvectors in ascending order
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Plotting (example: first 3 eigenfunctions)
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(x, eigenvectors[:, i], label=f'E_{i} = {eigenvalues[i]:.3f}')

plt.xlabel('x')
plt.ylabel('ψ(x)')
plt.title('Harmonic Oscillator Eigenfunctions')
plt.legend()
plt.grid(True)
plt.xlim(-x_max, x_max)
plt.ylim(-0.1,0.1)
plt.show()








# hbar, m, omega = 1, 1, 1


# def potential(N, x):
#     return 0.5 * m * omega ** 2 * x ** 2

# Physical constants (can be set to 1 for simplicity)
# def create_Hamiltonian(N, x):
#     # Construct the Hamiltonian matrix
#     hamiltonian = np.zeros((N, N))
#     for i in range(N):
#         hamiltonian[i, i] = hbar ** 2 / (m * dx ** 2) + potential(N, x[i])
#         if i > 0:
#             hamiltonian[i, i - 1] = -hbar ** 2 / (2 * m * dx ** 2)
#         if i < N - 1:
#             hamiltonian[i, i + 1] = -hbar ** 2 / (2 * m * dx ** 2)
#     return hamiltonian
