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
# Define
# Define the tri-diagonal matrix
x0 = np.linspace(0,10,100)
t  = np.linspace(t0,tF,11)
L  = np.zeros(len(x0)-1, dtype=complex)
L[0]    = .5*gamma
L[-1]   = 0
L[1:-1] = (.5*gamma)
lower_diag = np.zeros()
diagonals = [lower_diag, main_diag, upper_diag]
offsets = [-1, 0, 1]
matrix = np.diagflat(diagonals, offsets)
