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
             'legend.frameon'  : True,     # activate frame on lengend
            }
plt.rcParams.update(newParams) # Set new plotting parameters

# Parameters
N = 100  # Number of spatial points
dt = 0.001  # Time step
dx = 0.1  # Spatial step
hbar = 1.0
m = 1.0

# Spatial grid
x = np.linspace(-N*dx, N*dx, N)

# Potential: infinite square well
V = np.zeros(N)
V[0] = np.inf
V[-1] = np.inf


# Construct the Hamiltonian matrix (Complex Matrix)
c = -hbar**2 / (2 * m * dx**2)
H = np.zeros((N, N), dtype=complex)
for i in range(1, N - 1):
    H[i, i] = 2 * c + V[i]
    H[i, i + 1] = -c
    H[i, i - 1] = -c

# Boundary Conditions
H[0,0] = 1e30
H[-1,-1] = 1e30

# Identity matrix
I = np.identity(N, dtype=complex)

# Create matrices A and B
A = I + (1j/2) * H * dt  # 1j represents the imaginary unit
B = I - (1j/2) * H * dt

# Initial wave-function: Gaussian
psi = np.exp(-((x - N*dx/2)**2) / 2)
psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)  # Normalized Gaussian

# Time evolution
num_steps = 1000
psi_total = []
for n in range(num_steps):
    # Solve the linear system
    psi = np.linalg.solve(A, B @ psi)

    # Normalize the wave-function (important for stability)
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * dx)

    psi_total.append(psi)

    # You can plot psi here to visualize the evolution
    # if n % 100 == 0:
    #     plt.plot(x, np.abs(psi)**2)  # Plot probability density
    #     plt.show()
# Plot several wave-functions
plt.figure()
plt.plot(x, np.real(psi_total[0]), label='Re(psi) @ t = t0')
plt.plot(x, np.real(psi_total[-1]), label='Re(psi) @ t = tf')
plt.legend()
plt.show()

for i in range(num_steps):
    if i % 100 == 0:
        # print("P2 Norm at t = {}-----------".format(i))
        integral = np.trapezoid(np.abs(psi_total[i])**2, dx=dx)
        if (integral - 1) <= 1e-9:
            print("-----Norm is conserved-----")
            print("P2 Norm at t = {:0.1f}: {:.5f} ".format(i, integral))
        else:
            print("-----Norm is not conserved-----")
            break


