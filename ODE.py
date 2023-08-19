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
        """
        Solves the ODE using Euler's method \n
        Source: https://en.wikipedia.org/wiki/Euler_method
        :param t_end: Time to stop calculation at
        :param step_size: Step size
        :return: Tuple of lists of t and x values
        """
        t = [self.t_0]
        x = [self.initial_values[0]]
        derivatives = self.initial_values.copy()
        while t[-1] < t_end:
            xn_next = derivatives[-1] + step_size * self.f(*derivatives, t[-1])
            for i in range(self.order - 1):
                derivatives[i] += step_size * derivatives[i + 1]
            derivatives[-1] = xn_next
            t.append(t[-1] + step_size)
            x.append(derivatives[0])
        return t, x


    def heun(self, t_end: float, step_size: float) -> tuple[
        list[float], list[float]]:
        """
        Solves the ODE using Heun's method \n
        Source: https://en.wikipedia.org/wiki/Heun%27s_method
        :param t_end: Time to stop calculation at
        :param step_size: Step size
        :return: Tuple of lists of t and x values
        """
        # intermediate+1 = x + step_size * x'
        # intermediate'+1 = x' + step_size * x''
        # ...
        # intermediate^(n)+1 = x^(n) + step_size * f(x, x', ..., x^(n), t)

        # x+1 = x + 0.5 * step_size * (x'(t) + intermediate+1)
        # x'+1 = x' + 0.5 * step_size * (x''(t) + intermediate'+1)
        # ...
        # x^(n)+1 = x^(n) + 0.5 * step_size * (f(x, x', ..., x^(n), t) + intermediate^(n)+1)

        t = [self.t_0]
        x = [self.initial_values[0]]
        derivatives = self.initial_values.copy()
        predictor = self.initial_values.copy()
        while t[-1] < t_end:
            pred_last = predictor[-1] + step_size * self.f(*derivatives, t[-1])
            for i in range(self.order - 1):
                predictor[i] += step_size * derivatives[i + 1]
            predictor[-1] = pred_last

            xn_next = derivatives[-1] + 0.5 * step_size * (
                    self.f(*derivatives, t[-1]) + self.f(*predictor,
                                                         t[-1] + step_size))
            for i in range(self.order - 1):
                derivatives[i] += 0.5 * step_size * (
                        derivatives[i + 1] + predictor[i + 1])
            derivatives[-1] = xn_next
            t.append(t[-1] + step_size)
            x.append(derivatives[0])
        return t, x
