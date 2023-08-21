from scipy.integrate import odeint

from ODESolver.ode_solver import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
plt.style.use('seaborn-v0_8')

G = 9.81
L = 1.0


def simple_ode(x: float, t: float) -> float:
    return x * np.sin(t)


if __name__ == '__main__':
    # Set up plot
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')
    ax.set_title('Simple Pendulum')

    # Get all Methods
    methods = ODESolverBase.__subclasses__()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    method_colors = dict(zip(methods, colors))

    # Solve and plot for each method
    t_end = 10
    for method in methods:
        solver = method(simple_ode, [1], 0.1)
        solver.solve(t_end)
        solver.time_plot(ax, label=method.__name__, color=method_colors[method])

    # Plot Scipy solution for comparison
    t_values = np.linspace(0, t_end, 1000)
    scipy_solution = odeint(simple_ode, 1, t_values)
    ax.plot(t_values, scipy_solution, 'r', label='Scipy Solution',
            linestyle='--', zorder=10)

    # Show plot
    ax.legend()
    plt.show()
