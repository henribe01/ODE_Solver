from typing import Callable

import numpy as np

from .base_classes import AdaptiveRKBase


class AdaptiveHeunEuler(AdaptiveRKBase):
    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        runge_kutta_matrix = np.array([[0, 0],
                                       [1, 0]])
        weights = np.array([1 / 2, 1 / 2])
        weights2 = np.array([1, 0])
        nodes = np.array([0, 1])
        lower_tolerance = 1 / 2
        upper_tolerance = 2
        super().__init__(equation, initial_values, step_size, 2,
                         runge_kutta_matrix, weights, weights2, nodes,
                         lower_tolerance, upper_tolerance)

