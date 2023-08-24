from typing import Callable

import matplotlib
import numpy as np

from ODESolver.ode_solver import ForwardEuler


class ODESolverSystemBase:
    def __init__(self, equation: Callable[[..., float], np.array],
                 initial_values: list[float], step_size: float) -> None:
        """
        """
        self.equation = equation
        self.initial_values = initial_values
        self.step_size = step_size
        self.order = len(initial_values)
        self.t_values = np.array([0])

    def step(self):
        """
        Perform a single step of the solver.
        :return:
        """
        raise NotImplementedError

    def solve(self, t_end: float) -> None:
        """
        Solves the ODE up to a given time.
        :param t_end: The time to solve up to.
        :return: None
        """
        while self.t_values[-1] < t_end:
            self.step()


class ForwardEulerSystem(ODESolverSystemBase):
    def __init__(self, equation: Callable[[..., float], np.array],
                 initial_values: list[float], step_size: float) -> None:
        """
        """
        super().__init__(equation, initial_values, step_size)

        # Variables for storing the solution
        self.t_values = np.array([0])
        self.solution_values = np.array([initial_values])

    def step(self):
        """
        Perform a single step of the solver.
        :return:
        """
        # Append new derivative values to the array
        # new_values = [2, 4]
        # self.derivative_values = np.append(self.derivative_values, [new_values], axis=0)
        for i in range(len(self.initial_values)):
            self.initial_values[i] += self.step_size * \
                                      self.equation(self.initial_values,
                                                    self.t_values[-1])[i]
        self.t_values = np.append(self.t_values,
                                  self.t_values[-1] + self.step_size)
        self.solution_values = np.append(self.solution_values,
                                         [self.initial_values], axis=0)


def lorentz_equation(y: np.ndarray, t: float):
    """
    :param y: The solution vector.
    :param t: The time value.
    :return: The derivative vector.
    """
    sigma = 10
    rho = 28
    beta = 8 / 3
    x, y, z = y
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


if __name__ == '__main__':
    solver = ForwardEulerSystem(lorentz_equation, [1, 0, 0], 0.01)
    solver.solve(100)
    print(solver.solution_values)

    # Plot the solution
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(solver.solution_values[:, 0], solver.solution_values[:, 1],
            solver.solution_values[:, 2])
    plt.show()
