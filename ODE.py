from typing import Callable
import numpy as np


class ODE:
    def __init__(self, f: Callable[[float, ..., float], float],
                 initial_values: list[float], t_0: float) -> None:
        """
        :param f: Function that returns the nth derivative of x at a
        given x, x', ..., x^(n-1), t
        -> Right hand side of the ODE d^n x/dt^n = f(x, x', ..., x^(n-1), t)
        :param initial_values: List of initial values of x(t_0), x'(t_0), ...,
        x^(n-1)(t_0)
        :param t_0: Initial time
        """
        self.f = f
        self.initial_values = initial_values
        self.t_0 = t_0
        self.order = len(initial_values)

    def euler(self, t_end: float, step_size: float) -> tuple[
        list[float], list[float]]:
        t_values = np.linspace(self.t_0, t_end,
                               int((t_end - self.t_0) / step_size))
        x_values = np.zeros_like(t_values)
        x_values[0] = self.initial_values[0]
        derivatives_vector = np.array(self.initial_values, dtype=float)

        for i in range(1, len(t_values)):
            derivatives_vector[:-1] += step_size * derivatives_vector[1:]
            derivatives_vector[-1] += step_size * self.f(*derivatives_vector,
                                                         t_values[i - 1])
            x_values[i] = derivatives_vector[0]
        return t_values.tolist(), x_values.tolist()

    def heun(self, t_end: float, step_size: float) -> tuple[
        list[float], list[float]]:
        t_values = np.linspace(self.t_0, t_end,
                               int((t_end - self.t_0) / step_size))
        x_values = np.zeros_like(t_values)
        x_values[0] = self.initial_values[0]
        derivatives_vector = np.array(self.initial_values, dtype=float)
        predictor_vector = np.array(self.initial_values, dtype=float)

        for i in range(1, len(t_values)):
            predictor_vector[:-1] += step_size * derivatives_vector[1:]
            predictor_vector[-1] += step_size * self.f(*derivatives_vector,
                                                       t_values[i - 1])
            derivatives_vector[:-1] += 0.5 * step_size * (
                    derivatives_vector[1:] + predictor_vector[1:])
            derivatives_vector[-1] += 0.5 * step_size * (
                    self.f(*derivatives_vector, t_values[i - 1]) + self.f(
                *predictor_vector, t_values[i]))
            x_values[i] = derivatives_vector[0]
        return t_values.tolist(), x_values.tolist()

    def rk4(self, t_end: float, step_size: float) -> tuple[
        list[float], list[float]]:
        t_values = np.linspace(self.t_0, t_end,
                               int((t_end - self.t_0) / step_size))
        x_values = np.zeros_like(t_values)
        x_values[0] = self.initial_values[0]
        derivatives_vector = np.array(self.initial_values, dtype=float)
        k1 = np.zeros_like(derivatives_vector)
        k2 = np.zeros_like(derivatives_vector)
        k3 = np.zeros_like(derivatives_vector)
        k4 = np.zeros_like(derivatives_vector)

        for i in range(1, len(t_values)):
            k1[:-1] = step_size * derivatives_vector[1:]
            k1[-1] = step_size * self.f(*derivatives_vector, t_values[i - 1])
            k2[:-1] = step_size * (derivatives_vector[1:] + 0.5 * k1[1:])
            k2[-1] = step_size * self.f(
                *(derivatives_vector + 0.5 * k1),
                t_values[i - 1] + 0.5 * step_size)
            k3[:-1] = step_size * (derivatives_vector[1:] + 0.5 * k2[1:])
            k3[-1] = step_size * self.f(
                *(derivatives_vector + 0.5 * k2),
                t_values[i - 1] + 0.5 * step_size)
            k4[:-1] = step_size * (derivatives_vector[1:] + k3[1:])
            k4[-1] = step_size * self.f(
                *(derivatives_vector + k3), t_values[i - 1] + step_size)
            derivatives_vector += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            x_values[i] = derivatives_vector[0]
        return t_values.tolist(), x_values.tolist()

