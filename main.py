from scipy.integrate import odeint

from ODESolver.ode_solver import ForwardEuler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
plt.style.use('seaborn')

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

    # Set up solver
    solver = ForwardEuler(simple_ode, [1], 0.01)

    # Solve and plot
    solver.solve(100)
    solver.time_plot(ax, 0, 'b', 'Angle')

    # Plot Scipy solution for comparison
    t_values = np.linspace(0, 100, 1000)
    scipy_solution = odeint(simple_ode, 1, t_values)
    ax.plot(t_values, scipy_solution, 'r', label='Scipy Solution')

    # Show plot
    ax.legend()
    plt.show()
