from typing import Callable

import numpy as np

from .base_classes import AdaptiveRKBase


class AdaptiveHeunEuler(AdaptiveRKBase):
    """
    Adaptive Heun-Euler method. \n
    Source: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Heun%E2%80%93Euler
    """

    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        p = 2
        runge_kutta_matrix = np.array([[0, 0],
                                       [1, 0]])
        weights = np.array([1 / 2, 1 / 2])
        weights_star = np.array([1, 0])
        nodes = np.array([0, 1])
        super().__init__(equation, initial_values, step_size, p,
                         runge_kutta_matrix, weights, weights_star, nodes)


class AdaptiveRKF45(AdaptiveRKBase):
    """
    Adaptive Runge-Kutta 4(5) method. \n
    Source: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta%E2%80%93Fehlberg_methods
    """

    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        stages = 5
        runge_kutta_matrix = np.array([[0, 0, 0, 0, 0, 0],
                                       [1 / 4, 0, 0, 0, 0, 0],
                                       [3 / 32, 9 / 32, 0, 0, 0, 0],
                                       [1932 / 2197, -7200 / 2197, 7296 / 2197,
                                        0, 0, 0],
                                       [439 / 216, -8, 3680 / 513, -845 / 4104,
                                        0, 0],
                                       [-8 / 27, 2, -3544 / 2565, 1859 / 4104,
                                        -11 / 40, 0]])
        weights = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])
        weights_star = np.array(
            [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
        nodes = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
        super().__init__(equation, initial_values, step_size, stages,
                         runge_kutta_matrix, weights, weights_star, nodes)


class AdaptiveFehlberg12(AdaptiveRKBase):
    """
    Adaptive Fehlberg 1(2) method. \n
    Source: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Fehlberg_RK1(2)
    """

    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        stages = 2
        runge_kutta_matrix = np.array([[0, 0, 0],
                                       [1 / 2, 0, 0],
                                       [1 / 256, 255 / 256, 0]])
        weights = np.array([1 / 256, 255 / 256, 0])
        weights_star = np.array([1 / 512, 255 / 256, 1 / 512])
        nodes = np.array([0, 1 / 2, 1])
        super().__init__(equation, initial_values, step_size, stages,
                         runge_kutta_matrix, weights, weights_star, nodes)


class AdaptiveDormandPrince(AdaptiveRKBase):
    """
    Adaptive Dormand-Prince method. \n
    Source: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Dormand%E2%80%93Prince
    """

    def __init__(self, equation: Callable[[..., float], float],
                 initial_values: list[float], step_size: float) -> None:
        stages = 6
        runge_kutta_matrix = np.array([[0, 0, 0, 0, 0, 0, 0],
                                       [1 / 5, 0, 0, 0, 0, 0, 0],
                                       [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                                       [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                                       [19372 / 6561, -25360 / 2187,
                                        64448 / 6561, -212 / 729, 0, 0, 0],
                                       [9017 / 3168, -355 / 33, 46732 / 5247,
                                        49 / 176, -5103 / 18656, 0, 0],
                                       [35 / 384, 0, 500 / 1113, 125 / 192,
                                        -2187 / 6784, 11 / 84, 0]])
        weights = np.array([35 / 384, 0, 500 / 1113, 125 / 192,
                            -2187 / 6784, 11 / 84, 0])
        weights_star = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640,
                                    -92097 / 339200, 187 / 2100, 1 / 40])
        nodes = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1 / 2])
        super().__init__(equation, initial_values, step_size, stages,
                            runge_kutta_matrix, weights, weights_star, nodes)
