# Load modules
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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_time_series_data(solution, num_time_steps, num_points):
    """Generates sample time-series graph data."""
    data = []
    for t in range(num_time_steps):
        data.append(np.column_stack((solution.x, np.abs(solution.psi_total[t]))))
    return np.array(data)

def animate_psi_squared_series(time_series_data, t_steps, filename):
    """Creates and saves an animation of time-series data."""
    fig, ax = plt.subplots()
    line, = ax.plot(time_series_data[0, :, 0], time_series_data[0, :, 1], label="SHO")  # Initial plot
    ax.set_xlim(np.min(time_series_data[:, :, 0]), np.max(time_series_data[:, :, 0]))
    ax.set_ylim(np.min(time_series_data[:, :, 1]), np.max(time_series_data[:, :, 1]))
    ax.set_title(filename)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi(t)|^2$")

    def animate(i):
        line.set_xdata(time_series_data[i, :, 0])
        line.set_ydata(time_series_data[i, :, 1])
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=t_steps, interval=1, blit=True)
    ani.save(filename, writer='ffmpeg', fps=30)

