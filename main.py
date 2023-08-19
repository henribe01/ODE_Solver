from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib
from ODE import ODE

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


def d2xdt2(x: float, x_dot: float, t: float) -> float:
    """
    Function that returns the second derivative of x at a given x, x_dot and t
    :param x: Value of x
    :param x_dot: Value of x_dot
    :param t: Value of t
    :return: Second derivative of x at x, x_dot and t
    """
    # ODE for a simple pendulum
    return -np.sin(x)


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
    methods = [method for method in dir(ODE) if
               not method.startswith('__')]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink',
              'gray', 'olive', 'cyan']
    method_colors = dict(zip(methods, colors))
    print(
        f'Available methods: {", ".join(method.capitalize() for method in methods)}')

    # Set up plot
    fig, ax = plt.subplots()
    fig: plt.Figure
    ax: plt.Axes
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title('Comparison of ODE solving methods')

    # Set up variables
    t_end = 10
    t_0 = 0
    inital_values = [1, 0]
    stepsize = 0.1
    t_res = 1000

    # Set up scipy
    t_scipy = np.linspace(t_0, t_end, t_res)
    def f_scipy(u, t):
        return (u[1], d2xdt2(u[0], u[1], t))
    x_scipy = odeint(f_scipy, inital_values, t_scipy)
    ax.plot(t_scipy, x_scipy[:, 0], label='Scipy', color='black', zorder=1)

    # Set up ODE
    ode = ODE(d2xdt2, inital_values, t_0)

    for method in methods:
        # Calculate x values for each method
        t_steps, x = getattr(ode, method)(t_end, stepsize)
        # Calculate mean squared error
        # mse_value = mse(x, x_scipy_with_steps)

        # Plot x values
        ax.plot(t_steps, x,
                label=f'{method.capitalize()}',
                color=method_colors[method],
                zorder=2, alpha=0.5)
        ax.scatter(t_steps, x, color=method_colors[method], s=10, zorder=3,
                   alpha=0.5, marker='s')

    leg = ax.legend(loc='upper left')
    leg.set_draggable(True)
    leg.set_alpha(0.5)
    plt.show()
