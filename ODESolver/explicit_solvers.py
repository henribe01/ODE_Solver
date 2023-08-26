import numpy as np

from .base_classes import ODESolverBase


class ForwardEuler(ODESolverBase):
    def step(self) -> None:
        """
        Perform a single step of the solver using the Forward Euler method. \n
        x_{i+1} = x_i + h * x'_i \n
        x'_{i+1} = x'_i + h * x''_i \n
        ...
        x^{(n)}_{i+1} = x^{(n)}_i + h * f(x_i, x'_i, ..., x^{(n)}_i, t_i)
        :return: None
        """
        # Calculate new values
        new_values = self.derivative_values[-1] + self.step_size * self.f(
            self.derivative_values[-1], self.t_values[-1])

        # Append new values to the array
        self.derivative_values = np.append(self.derivative_values,
                                           [new_values], axis=0)
        self.t_values = np.append(self.t_values, self.t_values[-1] +
                                  self.step_size)


class Heun(ODESolverBase):
    def step(self) -> None:
        """
        Perform a single step of the solver using the Heun method. \n
        Source: https://en.wikipedia.org/wiki/Heun%27s_method
        :return: None
        """
        # Calculate intermediate values
        intermediate_values = self.derivative_values[
                                  -1] + self.step_size * self.f(
            self.derivative_values[-1], self.t_values[-1])

        # Calculate new values
        new_values = self.derivative_values[-1] + self.step_size / 2 * (self.f(
            self.derivative_values[-1], self.t_values[-1]) + self.f(
            intermediate_values, self.t_values[-1] + self.step_size))

        # Append new values to the array
        self.derivative_values = np.append(self.derivative_values,
                                           [new_values], axis=0)
        self.t_values = np.append(self.t_values, self.t_values[-1] +
                                  self.step_size)


class TwoStepAdamBashforth(ODESolverBase):
    def step(self) -> None:
        """
        Perform a single step of the solver using the 2-step Adams-Bashforth
        method. \n
        Source: https://en.wikipedia.org/wiki/Linear_multistep_method
        :return: None
        """
        # Calculate first two values using Euler's method
        if len(self.derivative_values) < 2:
            new_values = self.derivative_values[-1] + self.step_size * self.f(
                self.derivative_values[-1], self.t_values[-1])
            self.derivative_values = np.append(self.derivative_values,
                                               [new_values], axis=0)
            self.t_values = np.append(self.t_values, self.t_values[-1] +
                                      self.step_size)
            return
        # Calculate new values
        new_values = self.derivative_values[
                         -1] + 3 / 2 * self.step_size * self.f(
            self.derivative_values[-1],
            self.t_values[-1]) - 1 / 2 * self.step_size * self.f(
            self.derivative_values[-2], self.t_values[-2])

        # Append new values to the array
        self.derivative_values = np.append(self.derivative_values,
                                           [new_values], axis=0)
        self.t_values = np.append(self.t_values, self.t_values[-1] +
                                  self.step_size)
