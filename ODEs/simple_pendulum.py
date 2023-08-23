import numpy as np


def simple_pendulum(x: float, x_dot: float, t: float) -> float:
    """
    Returns the derivative of the simple pendulum.
    :param x: The angle of the pendulum.
    :param x_dot: The angular velocity of the pendulum.
    :param t: The time.
    :return: The derivative of the simple pendulum.
    """
    G = 9.81
    L = 1
    return -G / L * np.sin(x)