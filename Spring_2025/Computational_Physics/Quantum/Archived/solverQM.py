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


def initial_condition(x: np.ndarray[float], x0: float) -> np.ndarray[float]:
    """
    Compute the initial condition for the wave function.

    Parameters:
    x (np.ndarray[float]): The spatial grid points.
    x0 (float): The center of the Gaussian.

    Returns:
    np.ndarray[float]: The initial wave function values at the grid points.
    """
    return np.exp(-(x - x0)**2)


#define spacing and x,t arrays
dx = 1
dt=1
xi=0
xF = 10
t0 = 0
tF = 10
x0 = np.linspace(0,10,100)
t  = np.linspace(t0,tF,11)

# Define empty matrix where we hold x
x_mat = []

# Create the tri-diagonal matrix
gamma   = -1j*(dt/dx**2)
B       = np.zeros(len(x0), dtype=complex)
B[0]    = 1
B[-1]   = 1
B[1:-1] = (1-gamma)
u = np.zeros(len(x0)-1, dtype=complex)
u[0] = 0
u[-1] = .5*gamma
u[1:-1] = (.5*gamma)
L       = np.zeros(len(x0)-1, dtype=complex)
L[0]    = .5*gamma
L[-1]   = 0
L[1:-1] = (.5*gamma)
# Create the tridiagonal matrix
A = np.diag(B, 0) + np.diag(u, 1) + np.diag(L, -1)


psi = []
psi0: np.ndarray[complex] = initial_condition(x0, 5)
area = np.trapz(initial_condition(x0, 5)**2, dx=dx)
psi0 = psi0/np.sqrt(area)

psi.append(psi0)

for i in range(1000):
    psiN = np.linalg.solve(A, psi[i])
    psi.append(psiN)

plt.figure()
for i in range(1000):
    if i % 200 == 0:
        plt.plot(x0, np.real(psi[i]), label="Time {}".format(i))
    # plt.plot(x0, np.real(psi[i]), label="Time {}".format(i))
plt.title("Real Part of the Wave function")
plt.legend()
plt.grid()
plt.show()

plt.figure()
for i in range(1000):
    if i % 200 == 0:
        plt.plot(x0, np.imag(psi[i]), label="Time {}".format(i))
plt.title("Imaginary Part of the Wave function")
plt.legend()
plt.grid()
plt.show()

# Plot the norm of the wave-function
for i in range(1000):
    if i % 100 == 0:
        print(np.linalg.norm(psi[i]))

for i in range(1000):
    if i % 100 == 0:
        # print("P2 Norm at t = {}-----------".format(i))
        integral = np.trapezoid(np.abs(psi[i])**2, dx=dx)
        print("P2 Norm at t = {}: ".format(i) + str(integral))