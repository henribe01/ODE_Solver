def damped_oscillator(x: float, x_dot: float, t: float) -> float:
    """
    Return the derivative of the damped oscillator at x, x_dot, t.
    """
    freq = 5
    gamma = 0.1
    return -2 * gamma * x_dot - freq ** 2 * x
