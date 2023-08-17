from typing import Callable

TOL = 1e-6


class FirstOrderODE:
    def __init__(self, f: Callable[[float, float], float],
                 x_0: float, t_0: float):
        """
        :param f: Function that returns the derivative of x at a given x and t
        -> Right hand side of the ODE dx/dt = f(x, t)
        :param x_0: Initial value of x -> x(t_0)
        :param t_0: Initial value of t
        """
        self.f = f
        self.x_0 = x_0
        self.t_0 = t_0

    def euler(self, t_end: float, stepsize: float) -> list[float]:
        """
        Implementation of the Euler method for solving ODEs
        Source: https://en.wikipedia.org/wiki/Euler_method
        :param t_end: Value of t at which to stop
        :param stepsize: Step size
        :return: List of x values at steps h, 2h, 3h, ..., t_end
        """
        x_n = self.x_0
        t_n = self.t_0
        results = [self.x_0]
        # Add tolerance for floating point errors
        while t_n + TOL < t_end:
            x_n += stepsize * self.f(x_n, t_n)
            t_n += stepsize
            results.append(x_n)
        return results

    def heun(self, t_end: float, stepsize: float) -> list[float]:
        """
        Implementation of the Heun method for solving ODEs
        Source: https://en.wikipedia.org/wiki/Heun%27s_method
        :param t_end: Value of t at which to stop
        :param stepsize: Step size
        :return: List of x values at steps h, 2h, 3h, ..., t_end
        """
        x_n = self.x_0
        t_n = self.t_0
        t_next = self.t_0 + stepsize
        results = [self.x_0]
        while t_n + TOL < t_end:
            x_intermediate = x_n + stepsize * self.f(x_n, t_n)
            x_n += stepsize / 2 * (self.f(x_n, t_n) + self.f(x_intermediate, t_next))
            t_n += stepsize
            t_next += stepsize
            results.append(x_n)
        return results
