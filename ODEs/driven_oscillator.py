import numpy as np


def driven_oscillator(x: float, x_dot: float, t) -> float:
    """
    Return the derivative of the driven oscillator at x, x_dot, t.
    """
    freq = 1
    freq_driving = 0.4
    amplitude = 1
    return - freq ** 2 * x + amplitude * np.cos(freq_driving * t)