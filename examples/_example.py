from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from ODE import ODE


class Example:
    def __init__(self, initial_values: list[float], t_0: float,
                 speed: float = 1) -> None:
        """
        :param initial_values: List of initial values of x(t_0), x'(t_0), ...,
        x^(n-1)(t_0)
        :param t_0: Initial time
        :param speed: Speed of the animation
        """
        self.phase_plot = None
        self.stream_plot = None
        self.time_plot = None
        self.t_values = None
        self.x_values = None
        self.speed = speed
        self.ode = ODE(self._f, initial_values, t_0)

    def _f(self, *args) -> float:
        raise NotImplementedError

    def solve(self, t_end: float, step_size: float,
              method: str = 'euler') -> None:
        """
        Solves the ODE using the specified method and stores the results in
        self.t_values and self.x_values
        :param t_end: Time to stop solving at
        :param step_size: Step size
        :param method: Numerical method to use to solve the ODE
        :return: None
        """
        # Check if the method is valid
        methods = {key: item for key, item in ODE.__dict__.items() if
                   '__' not in key}
        if method not in methods:
            raise ValueError(f'Invalid method {method}')

        # Solve the ODE
        self.t_values, self.x_values = methods[method](self.ode, t_end,
                                                       step_size)

    def plot_time(self, ax: plt.Axes, label: str = 'x(t)',
                  color: str = 'red') -> plt.plot:
        """
        Plots the first variable against time
        :param ax: Matplotlib axes to plot on
        :param label: Label for the plot
        :param color: Color of the plot
        :return: Matplotlib plot object
        """
        # Check if the ODE has been solved
        if self.t_values is None or self.x_values is None:
            raise ValueError('ODE has not been solved, call solve() first')

        # Plot the results
        self.time_plot, = ax.plot(self.t_values, self.x_values[0],
                                  label=label, color=color)
        return self.time_plot

    def plot_phase_space(self, ax: plt.Axes, stream_plot: bool = False,
                         label: str = 'Phase space',
                         color: str = 'blue') -> plt.plot:
        """
        Plots the phase space of the ODE using the first two variables. \n
        :param ax: Matplotlib axes to plot on
        :param stream_plot: Whether to use a stream plot, for custom settings
        call plot_stream_phase_space and then plot_phase_space
        :param label: Label for the plot
        :param color: Color of the plot
        :return: Matplotlib plot object
        """
        # Check if the ODE has been solved
        if self.t_values is None or self.x_values is None:
            raise ValueError('ODE has not been solved, call solve() first')

        # Check if there is already a stream plot
        if self.stream_plot is not None and stream_plot:
            raise ValueError('Stream plot already exists')

        # Plot the results
        if stream_plot:
            self.plot_stream_phase_space(ax)
        self.phase_plot, = ax.plot(self.x_values[0], self.x_values[1],
                                   label=label, color=color)
        return self.phase_plot

    def plot_stream_phase_space(self, ax: plt.Axes,
                                x_range: tuple, y_range: tuple,
                                step_size: float = 0.1,
                                density: tuple = (1, 1),
                                cmap: str = 'jet_r',
                                color: str | Callable = 'black') -> plt.streamplot:
        """
        Plots the phase space of the ODE using a stream plot. \n
        Only available if the ODE is of second order.
        :param ax: Matplotlib axes to plot on
        :param x_range: Minimum and maximum values for the x-axis
        :param y_range: Minimum and maximum values for the y-axis
        :param step_size: Step size for the stream plot
        :param density: Density of the stream plot, higher values mean more
        lines
        :param cmap: Colormap to use for the stream plot
        :param color: Color of the stream plot, if a string is passed then
        the color is constant, if a function is passed then the color is using
        the colormap and the function
        :return: Matplotlib streamplot object
        """
        # Check if the ODE is of second order
        if len(self.x_values) != 2:
            raise ValueError('ODE is not of second order')

        # Check if the ODE has been solved
        if self.t_values is None or self.x_values is None:
            raise ValueError('ODE has not been solved, call solve() first')

        # Calculate values for the stream plot
        x_values = np.linspace(x_range[0], x_range[1], int(
            (x_range[1] - x_range[0]) / step_size))
        y_values = np.linspace(y_range[0], y_range[1], int(
            (y_range[1] - y_range[0]) / step_size))
        X, Y = np.meshgrid(x_values, y_values)
        U = Y  # x' = y
        V = self._f(X, Y, 0)  # x'' = f(x, x', t)

        # Plot the results
        if isinstance(color, str):
            self.stream_plot = ax.streamplot(X, Y, U, V, density=density,
                                             color=color)
        else:
            self.stream_plot = ax.streamplot(X, Y, U, V, density=density,
                                             color=color(U, V), cmap=cmap)
        return self.stream_plot

    def animate(self, frame: int):
        """
        Animate the ODE, if plot_phase_space has been called then the phase
        space will be animated, otherwise the time plot will be animated.
        :param frame: Current frame
        :return:
        """
        i = frame % len(self.t_values)
        # Check if time plot has been initialized
        if self.time_plot is None:
            raise ValueError('Time plot has not been initialized')

        self.time_plot.set_data(self.t_values[:i * self.speed],
                                self.x_values[0][:i * self.speed])

        # Check if phase plot has been initialized
        if self.phase_plot is not None:
            self.phase_plot.set_data(self.x_values[0][:i * self.speed],
                                     self.x_values[1][:i * self.speed])

        return self.time_plot, self.phase_plot
