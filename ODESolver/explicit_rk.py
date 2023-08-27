from typing import Callable

import numpy as np

from .base_classes import RKBase


class Midpoint(RKBase):
    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        runge_kutta_matrix = np.array([[0, 0],
                                       [1 / 2, 0]])
        weights = np.array([0, 1])
        nodes = np.array([0, 1 / 2])
        super().__init__(equation, initial_values, step_size, 2,
                         runge_kutta_matrix, weights, nodes)

class RK3(RKBase):
    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        runge_kutta_matrix = np.array([[0, 0, 0],
                                       [1 / 2, 0, 0],
                                       [-1, 2, 0]])
        weights = np.array([1 / 6, 2 / 3, 1 / 6])
        nodes = np.array([0, 1 / 2, 1])
        super().__init__(equation, initial_values, step_size, 3,
                            runge_kutta_matrix, weights, nodes)


class SSPRK3(RKBase):
    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        runge_kutta_matrix = np.array([[0, 0, 0],
                                       [1, 0, 0],
                                       [1 / 4, 1 / 4, 0]])
        weights = np.array([1 / 6, 1 / 6, 2 / 3])
        nodes = np.array([0, 1, 1 / 2])
        super().__init__(equation, initial_values, step_size, 3,
                            runge_kutta_matrix, weights, nodes)


class RK4(RKBase):
    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        """
        Initialize the RK4 solver. \n
        Source: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        :param equation: Function that represents the ODE. \n
        The function must take the form f(x, x', ..., x^{(n-1)}, t) \n
        and return the value of x^{(n)} \n
        :param initial_values: A list of the initial values of the ODE. \n
        Must be of length n. \n
        :param step_size: The step size of the solver.
        """
        runge_kutta_matrix = np.array([[0, 0, 0, 0],
                                       [1 / 2, 0, 0, 0],
                                       [0, 1 / 2, 0, 0],
                                       [0, 0, 1, 0]])
        weights = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        nodes = np.array([0, 1 / 2, 1 / 2, 1])
        super().__init__(equation, initial_values, step_size, 4,
                         runge_kutta_matrix, weights, nodes)
