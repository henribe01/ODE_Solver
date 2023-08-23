from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.style.use('seaborn-v0_8')


class ODESolverBase:
    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        """
        :param equation: Function that represents the ODE. \n
        The function must take the form f(x, x', ..., x^{(n-1)}, t) \n
        and return the value of x^{(n)} \n
        :param initial_values:
        :param step_size:
        """
        self.equation = equation
        self.initial_values = initial_values
        self.step_size = step_size
        self.order = len(initial_values)  # Order of the ODE

        # Variables for storing the solution
        self.t_values = np.array([0])
        self.derivative_values = np.array([initial_values])  # 2D array

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        methods = self.__class__.__subclasses__()
        method_colors = {method: colors[i] for i, method in enumerate(methods)}
        self.color = None
        if self.__class__ in methods:
            self.color = method_colors[self.__class__]

    def f(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Converts a high order ODE to a system of first order ODEs.
        :param t: The time value.
        :param y: The solution vector.
        :return: The derivative vector.
        """
        return np.array([y[i] for i in range(1, len(y))] + [
            self.equation(*y, t)])

    def step(self) -> None:
        """
        Perform a single step of the solver.
        :return:
        """
        # Append new derivative values to the array
        # new_values = [2, 4]
        # self.derivative_values = np.append(self.derivative_values, [new_values], axis=0)
        raise NotImplementedError

    def solve(self, t_end: float) -> None:
        """
        Solves the ODE up to a given time.
        :param t_end: The time to solve up to.
        :return: None
        """
        while self.t_values[-1] < t_end:
            self.step()

    def get_n_th_derivative(self, n: int) -> np.ndarray:
        """
        Returns the calculated solutions for the n-th derivative.
        :param n: The order of the derivative to return d^n x/dt^n
        :return: A numpy array of the derivative values.
        """
        return self.derivative_values[:, n]

    def get_t_values(self) -> np.ndarray:
        """
        Returns the time values for the solution.
        :return: A numpy array of the time values.
        """
        return self.t_values

    def time_plot(self, ax: plt.Axes, n: int = 0,
                  label: str = '') -> list[
        plt.Line2D, matplotlib.collections.PathCollection]:
        """
        Plots the n-th derivative against time.
        :param ax: A matplotlib axes object to plot on.
        :param n: The order of the derivative to plot.
        :param color: The color of the line.
        :param label: The label of the line.
        :return: None
        """
        if label == '':
            label = f'{self.__class__.__name__.capitalize()}'
        line = ax.plot(self.get_t_values(), self.get_n_th_derivative(n),
                       color=self.color, label=label, alpha=0.5, zorder=1)[0]

        # If step size is small, only plot the points with distance
        interval = 1
        distance = 0.5
        if self.step_size < distance:
            interval = int(distance / self.step_size)
        scatter = ax.scatter(self.get_t_values()[::interval],
                             self.get_n_th_derivative(n)[::interval],
                             color=self.color, alpha=0.5, s=10, marker='s')
        return line, scatter

    def phase_plot(self, ax: plt.Axes, n: int = 0, color: str = 'k',
                   label: str = '') -> plt.Line2D:
        """
        Plots the n-th derivative against the (n+1)-th derivative.
        :param ax: A matplotlib axes object to plot on.
        :param n: The order of the derivative to plot.
        :param color: The color of the line.
        :param label: The label of the line.
        :return: None
        """
        if label == '':
            label = f'd^{n}x/dt^{n} vs d^{n + 1}x/dt^{n + 1}'
        return ax.plot(self.get_n_th_derivative(n),
                       self.get_n_th_derivative(n + 1), color=color,
                       label=label)[0]

    def get_mse(self, scipy_solution: np.ndarray) -> float:
        """
        Calculates the mean squared error between the calculated solution and a scipy solution.
        :param scipy_solution: The scipy solution.
        :return: The mean squared error.
        """
        # Get shortest
        if len(self.t_values) > len(scipy_solution):
            length = len(scipy_solution)
        else:
            length = len(self.t_values)
        return 1 / length * np.sum(
            (self.get_n_th_derivative(0)[:length] - scipy_solution[
                                                    :length]) ** 2)


class ForwardEuler(ODESolverBase):
    def step(self) -> None:
        """
        Perform a single step of the solver using the Forward Euler method. \n
        x_{i+1} = x_i + h * x'_i \n
        x'_{i+1} = x'_i + h * x''_i \n
        ...
        x^{(n)}_{i+1} = x^{(n)}_i + h * f(x_i, x'_i, ..., x^{(n)}_i, t_i)
        :return: None
        """
        # Calculate new values
        new_values = self.derivative_values[-1] + self.step_size * self.f(
            self.derivative_values[-1], self.t_values[-1])

        # Append new values to the array
        self.derivative_values = np.append(self.derivative_values,
                                           [new_values], axis=0)
        self.t_values = np.append(self.t_values, self.t_values[-1] +
                                  self.step_size)


class Heun(ODESolverBase):
    def step(self) -> None:
        """
        Perform a single step of the solver using the Heun method. \n
        Source: https://en.wikipedia.org/wiki/Heun%27s_method
        :return: None
        """
        # Calculate intermediate values
        intermediate_values = self.derivative_values[
                                  -1] + self.step_size * self.f(
            self.derivative_values[-1], self.t_values[-1])

        # Calculate new values
        new_values = self.derivative_values[-1] + self.step_size / 2 * (self.f(
            self.derivative_values[-1], self.t_values[-1]) + self.f(
            intermediate_values, self.t_values[-1] + self.step_size))

        # Append new values to the array
        self.derivative_values = np.append(self.derivative_values,
                                           [new_values], axis=0)
        self.t_values = np.append(self.t_values, self.t_values[-1] +
                                  self.step_size)


class RK4(ODESolverBase):
    def step(self) -> None:
        """
        Perform a single step of the solver using the RK4 method. \n
        Source: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        :return: None
        """
        # Calculate k-values
        k1 = self.f(self.derivative_values[-1], self.t_values[-1])
        k2 = self.f(self.derivative_values[-1] + self.step_size / 2 * k1,
                    self.t_values[-1] + self.step_size / 2)
        k3 = self.f(self.derivative_values[-1] + self.step_size / 2 * k2,
                    self.t_values[-1] + self.step_size / 2)
        k4 = self.f(self.derivative_values[-1] + self.step_size * k3,
                    self.t_values[-1] + self.step_size)

        # Calculate new values
        new_values = self.derivative_values[-1] + self.step_size / 6 * (
                k1 + 2 * k2 + 2 * k3 + k4)

        # Append new values to the array
        self.derivative_values = np.append(self.derivative_values,
                                           [new_values], axis=0)
        self.t_values = np.append(self.t_values, self.t_values[-1] +
                                  self.step_size)


class TwoStepAdamBashforth(ODESolverBase):
    def step(self) -> None:
        """
        Perform a single step of the solver using the 2-step Adams-Bashforth
        method. \n
        Source: https://en.wikipedia.org/wiki/Linear_multistep_method
        :return: None
        """
        # Calculate first two values using Euler's method
        if len(self.derivative_values) < 2:
            new_values = self.derivative_values[-1] + self.step_size * self.f(
                self.derivative_values[-1], self.t_values[-1])
            self.derivative_values = np.append(self.derivative_values,
                                               [new_values], axis=0)
            self.t_values = np.append(self.t_values, self.t_values[-1] +
                                      self.step_size)
            return
        # Calculate new values
        new_values = self.derivative_values[
                         -1] + 3 / 2 * self.step_size * self.f(
            self.derivative_values[-1],
            self.t_values[-1]) - 1 / 2 * self.step_size * self.f(
            self.derivative_values[-2], self.t_values[-2])

        # Append new values to the array
        self.derivative_values = np.append(self.derivative_values,
                                           [new_values], axis=0)
        self.t_values = np.append(self.t_values, self.t_values[-1] +
                                  self.step_size)
