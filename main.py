from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.style.use('seaborn')

def euler(x_0: float, t_0: float, t_end: float, h: float, dxdt: Callable) -> \
        list[float]:
    """
    Implementation of the Euler method for solving ODEs
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
    while t_n < t_end:
        x_n += h * dxdt(x_n, t_n)
        t_n += h
        results.append(x_n)
    return results


def dxdt(x: float, t: float) -> float:
    """
    Function that returns the derivative of x at a given x and t
    :param x: Value of x
    :param t: Value of t
    :return: Derivative of x at x and t
    """
    return x


if __name__ == '__main__':
    x_0 = 1
    t_0 = 0
    t_end = 4
    h = 1
    t_range = np.arange(t_0, t_end + h, h)

    # Solve using scipy -> Almost exact solution
    scipy_results = odeint(dxdt, x_0, t_range)

    # Solve using different methods
    euler_results = euler(x_0, t_0, t_end, h, dxdt)

    # Compare results
    fig, ax = plt.subplots()
    ax.plot(t_range, scipy_results, label='Scipy', color='blue')
    ax.plot(t_range, euler_results, label='Euler', color='red')
    ax.set_title(
        f'Comparison of solutions with initial condition x({t_0}) = {x_0}'
        f' and step size h = {h}')

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend()
    plt.show()
