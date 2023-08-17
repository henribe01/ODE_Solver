from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib

matplotlib.use('TkAgg')

plt.style.use('seaborn-v0_8')


def euler(x_0: float, t_0: float, t_end: float, h: float, dxdt: Callable) -> \
        list[float]:
    """
    Implementation of the Euler method for solving ODEs
    Source: https://en.wikipedia.org/wiki/Euler_method
    :param x_0: Initial value of x
    :param t_0: Initial value of t
    :param t_end: Value of t at which to stop
    :param h: Step size
    :param dxdt: Function that returns the derivative of x at a given x and t
    -> Right hand side of the ODE
    :return: List of x values at steps h, 2h, 3h, ..., t_end
    """
    x_n = x_0
    t_n = t_0
    results = [x_0]
    # Tolerance for floating point errors
    while t_n + 1e-6 < t_end:
        x_n += h * dxdt(x_n, t_n)
        t_n += h
        results.append(x_n)
    return results


def heun(x_0: float, t_0: float, t_end: float, h: float, dxdt: Callable) -> \
        list[float]:
    """
    Implementation of the Heun method for solving ODEs
    Source: https://en.wikipedia.org/wiki/Heun%27s_method
    :param x_0: Initial value of x
    :param t_0: Initial value of t
    :param t_end: Value of t at which to stop
    :param h: Step size
    :param dxdt: Function that returns the derivative of x at a given x and t
    -> Right hand side of the ODE
    :return: List of x values at steps h, 2h, 3h, ..., t_end
    """
    x_n = x_0
    t_n = t_0
    t_next = t_0 + h
    intermediate_x = 0
    results = [x_0]
    while t_n + 1e-6 < t_end:
        intermediate_x = x_n + h * dxdt(x_n, t_n)
        x_n += h / 2 * (dxdt(x_n, t_n) + dxdt(intermediate_x, t_next))
        t_n += h
        t_next += h
        results.append(x_n)
    return results


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
    x_0 = 1
    t_0 = 0
    t_end = 4
    step_size = 1
    integration_methods = {'Euler': euler, 'Heun': heun}
    colors = {'Euler': 'red', 'Heun': 'blue'}

    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlabel('t')
    ax.set_ylabel('x')

    # Plot scipy solution for comparison
    t = np.linspace(t_0, t_end, 1000)
    x_exact = odeint(dxdt, x_0, t).reshape(-1).tolist()
    t_with_steps = np.linspace(t_0, t_end, int((t_end - t_0) / step_size) + 1)
    x_scipy_with_steps = odeint(dxdt, x_0, t_with_steps).reshape(-1).tolist()
    ax.plot(t, x_exact, label='Exact solution')

    # Plot Integration methods
    for method_name, method in integration_methods.items():
        x = method(x_0, t_0, t_end, step_size, dxdt)
        mse_value = mse(x, x_scipy_with_steps)
        ax.plot(t_with_steps, x,
                label=f'{method_name} method, MSE: {mse_value:.2e}',
                color=colors[method_name])
        ax.scatter(t_with_steps, x, color=colors[method_name])

    ax.legend()
    plt.show()
