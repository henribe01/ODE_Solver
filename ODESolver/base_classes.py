from typing import Callable

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp, odeint

plt.style.use('seaborn-v0_8')


class ODESolverBase:
    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        """
        :param equation: Function that represents the ODE. \n
        The function must take the form f(x, x', ..., x^{(n-1)}, t) \n
        and return the value of x^{(n)} \n
        :param initial_values: A list of the initial values of the ODE. \n
        Must be of length n. \n
        :param step_size: The step size of the solver.
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
        raise NotImplementedError

    def solve(self, t_end: float) -> None:
        """
        Solves the ODE up to a given time.
        :param t_end: The time to solve up to.
        :return: None
        """
        # Account for floating point errors
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
            label = f'{self.__class__.__name__}'
        line = ax.plot(self.get_t_values(), self.get_n_th_derivative(n),
                       color=self.color, label=label, zorder=1)[0]

        # If step size is small, only plot the points with distance
        interval = 1
        distance = 0.5
        if self.step_size < distance:
            interval = int(distance / self.step_size)
        scatter = ax.scatter(self.get_t_values()[::interval],
                             self.get_n_th_derivative(n)[::interval],
                             color=self.color, s=10, marker='s')
        return line, scatter


    def get_mse(self) -> float:
        """
        Calculates the mean squared error between the calculated solution and a scipy solution.
        :param scipy_solution: The scipy solution.
        :return: The mean squared error.
        """
        # Calculate the scipy solution
        scipy_solution = odeint(self.f, self.initial_values, self.t_values)[:, 0]

        length = len(self.t_values)
        return np.sum(
            (self.get_n_th_derivative(0)[:length] - scipy_solution[
                                                    :length]) ** 2) / length

    def get_max_error(self, scipy_solution: np.ndarray) -> float:
        """
        Calculates the maximum error between the calculated solution and a scipy solution.
        :param scipy_solution: The scipy solution.
        :return: The maximum error.
        """
        length = len(self.t_values)
        return np.max(
            np.abs(self.get_n_th_derivative(0)[:length] - scipy_solution[
                                                          :length]))


class RKBase(ODESolverBase):
    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float, stages: int,
                 runge_kutta_matrix: np.ndarray, weights: np.ndarray,
                 nodes: np.ndarray) -> None:
        """
        For a list of Runge-Kutta methods, see: \n
        https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        :param equation: Function that represents the ODE. \n
        The function must take the form f(x, x', ..., x^{(n-1)}, t) \n
        and return the value of x^{(n)} \n
        :param initial_values: A list of the initial values of the ODE. \n
        Must be of length n. \n
        :param step_size: The step size of the solver.
        :param stages: The number of stages of the Runge-Kutta method.
        :param runge_kutta_matrix: The Runge-Kutta matrix. \n
        Must be of shape (stages, stages).
        :param weights: The weights of the Runge-Kutta method. \n
        Must be of length stages.
        :param nodes: The nodes of the Runge-Kutta method. \n
        Must be of length stages.
        """
        self.runge_kutta_matrix = runge_kutta_matrix
        self.weights = weights
        self.nodes = nodes
        self.stages = stages
        super().__init__(equation, initial_values, step_size)

    def step(self) -> None:
        """
        Perform a single step of the solver using the Runge-Kutta method. \n
        :return:
        """
        # Calculate k-values, rows are k-values, columns are derivatives
        # So k_values[0] is k_1
        # k_values[1] is k_2
        # etc.
        k_values = np.zeros((self.stages, len(self.initial_values)))
        for i in range(self.stages):
            # Calculate k_i
            k_values[i] = self.f(self.derivative_values[-1] + self.step_size *
                                 np.dot(self.runge_kutta_matrix[i, :i],
                                        k_values[:i]),
                                 self.t_values[-1] + self.step_size *
                                 self.nodes[i])

        # Calculate new values
        new_values = self.derivative_values[-1] + self.step_size * \
                     np.dot(self.weights, k_values)

        # Append new values to the array
        self.derivative_values = np.append(self.derivative_values,
                                           [new_values], axis=0)
        self.t_values = np.append(self.t_values, self.t_values[-1] +
                                  self.step_size)


class AdaptiveRKBase(RKBase):
    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float, p: int,
                 runge_kutta_matrix: np.ndarray, weights: np.ndarray,
                 weights2: np.ndarray,
                 nodes: np.ndarray, tolerance: float = 1e-3) -> None:
        """
        For a list of Runge-Kutta methods, see: \n
        https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Embedded_methods
        :param equation: Function that represents the ODE. \n
        The function must take the form f(x, x', ..., x^{(n-1)}, t) \n
        and return the value of x^{(n)} \n
        :param initial_values: A list of the initial values of the ODE. \n
        Must be of length n. \n
        :param step_size: The step size of the solver.
        :param p: The order of the method.
        :param runge_kutta_matrix: The Runge-Kutta matrix. \n
        Must be of shape (stages, stages).
        :param weights: The weights of the Runge-Kutta method. \n
        Must be of length stages.
        :param weights2: The weights of the Runge-Kutta method, which are used
        to calculate the error. \n
        Must be of length stages - 1.
        :param nodes: The nodes of the Runge-Kutta method. \n
        Must be of length stages.
        :param tolerance: If the error is below this value, the step is
        accepted. \n
        """
        self.weights2 = weights2
        self.tolerance = tolerance
        self.p = p
        super().__init__(equation, initial_values, step_size, p,
                         runge_kutta_matrix, weights, nodes)
        self.original_step_size = step_size

    def step(self) -> None:
        """
        Perform a single step of the solver using the Runge-Kutta method. \n
        :return:
        """
        # Calculate k-values
        k_values = np.zeros((self.runge_kutta_matrix.shape[0],
                             self.derivative_values.shape[1]))
        for i in range(self.stages):
            # Calculate k_i
            k_values[i] = self.f(self.derivative_values[-1] + self.step_size *
                                 np.dot(self.runge_kutta_matrix[i, :i],
                                        k_values[:i]),
                                 self.t_values[-1] + self.step_size *
                                 self.nodes[i])

        # Calculate values with order p
        values_p = self.derivative_values[-1] + self.step_size * \
                   np.dot(self.weights, k_values)

        # Calculate values with order p - 1
        values_p1 = self.derivative_values[-1] + self.step_size * \
                    np.dot(self.weights2, k_values)

        # Calculate error
        error = np.linalg.norm(values_p - values_p1)

        # Calculate new step size
        new_step_size = 0.9 * self.step_size * (self.tolerance / error) ** (
                1 / self.p)

        # If error is below tolerance, accept step
        if error < self.tolerance:
            # Append new values to the array
            self.derivative_values = np.append(self.derivative_values,
                                               [values_p], axis=0)
            self.t_values = np.append(self.t_values, self.t_values[-1] +
                                      self.step_size)
            self.step_size = new_step_size
        else:
            # Reject step
            self.step_size = new_step_size
