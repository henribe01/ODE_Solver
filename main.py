from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib
from ODE import FirstOrderODE

matplotlib.use('TkAgg')

plt.style.use('seaborn-v0_8')


def dxdt(x: float, t: float) -> float:
    """
    Function that returns the derivative of x at a given x and t
    :param x: Value of x
    :param t: Value of t
    :return: Derivative of x at x and t
    """
    return x * np.sin(t)


def mse(x: list[float], y: list[float]) -> float:
    """
    Calculates the mean squared error between two lists of floats
    :param x: List of approximated floats
    :param y: List of true floats
    :return: Mean squared error between x and y
    """
    return np.mean((np.array(x) - np.array(y)) ** 2)


if __name__ == '__main__':
    # List of all ODE solving methods
    methods = [method for method in dir(FirstOrderODE) if
               not method.startswith('__')]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink',
              'gray', 'olive', 'cyan']
    method_colors = dict(zip(methods, colors))
    print(
        f'Available methods: {", ".join(method.capitalize() for method in methods)}')

    # Set up ODE
    ode = FirstOrderODE(dxdt, 1, 0)

    # Set up plot
    fig, ax = plt.subplots()
    fig: plt.Figure
    ax: plt.Axes
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title('Comparison of ODE solving methods')

    # Plot Scipy's odeint method for comparison
    t_end = 10
    t_resolution = 1000
    t = np.linspace(0, t_end, t_resolution)
    x_scipy = odeint(dxdt, ode.x_0, t)
    ax.plot(t, x_scipy, label='Scipy odeint', color='black', linewidth=2,
            zorder=0, linestyle='--')

    # Variables for calculating the mean squared error
    stepsize = 0.1
    t_with_steps = np.linspace(0, t_end, int(t_end / stepsize) + 1)
    x_scipy_with_steps = odeint(dxdt, ode.x_0, t_with_steps)

    # Plot all methods
    stepsize = 0.1
    for method in methods:
        # Calculate x values for each method
        x = getattr(ode, method)(t_end, stepsize)

        # Calculate mean squared error
        mse_value = mse(x, x_scipy_with_steps)

        # Plot x values
        t_steps = np.linspace(0, t_end, len(x))
        ax.plot(t_steps, x,
                label=f'{method.capitalize()} ({mse_value:.2e})',
                color=method_colors[method],
                zorder=2, alpha=0.5)
        ax.scatter(t_steps, x, color=method_colors[method], s=10, zorder=3,
                   alpha=0.5, marker='s')

    ax.legend()
    plt.show()
