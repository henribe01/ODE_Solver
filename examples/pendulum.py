import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

from ODE import ODE


class Pendulum:
    def __init__(self, l: float, theta_0: float, theta_dot_0: float,
                 g: float = 9.81) -> None:
        """
        :param l: Length of the pendulum
        :param theta_0: Initial angle of the pendulum
        :param theta_dot_0: Initial angular velocity of the pendulum
        :param g: Gravitational acceleration
        """
        self.l = l
        self.theta_0 = theta_0
        self.theta_dot_0 = theta_dot_0
        self.g = g
        self.ode = ODE(self._f, [theta_0, theta_dot_0], 0)
        self.t_values = None
        self.theta_values = None
        self.theta_dot_values = None

    def _f(self, theta: float, theta_dot: float, t: float) -> float:
        """
        Second derivative of theta
        :param theta: Angle
        :param theta_dot: Angular velocity
        :param t: Time
        :return: Second derivative of theta at theta, theta_dot, t
        """
        # Damped pendulum
        return -self.g / self.l * np.sin(theta) - damping * theta_dot / self.l ** 2

    def solve(self, t_end: float, step_size: float,
              method: str = 'euler') -> None:
        """
        Solves the ODE for the pendulum using the given method and stores the
        results in self.t_values, self.theta_values, self.theta_dot_values
        :param t_end: End time
        :param step_size: Step size
        :param method: Method to solve the ODE
        :return: None
        """
        methods = {key: item for key, item in ODE.__dict__.items() if
                   '__' not in key}
        if method not in methods:
            raise ValueError(f'Invalid method {method}')
        self.t_values, self.theta_values, self.theta_dot_values = methods[
            method](self.ode, t_end, step_size)

    def plot(self, ax: plt.Axes) -> plt.plot:
        """
        Plots the results of the ODE
        :param ax: Axes to plot on
        :return: None
        """
        assert self.t_values is not None and self.theta_values is not None
        self.time_plot, = ax.plot(self.t_values, self.theta_values,
                                  label='theta',
                                  color='red')
        return self.time_plot

    def plot_phase_space(self, ax: plt.Axes) -> plt.plot:
        """
        Plots the phase space of the pendulum
        :param ax: Axes to plot on
        :return: None
        """
        if self.theta_dot_values is None:
            self.calculate_theta_dot_values()
        self.phase_plot, = ax.plot(self.theta_values, self.theta_dot_values,
                                   label='phase space', color='blue')
        return self.phase_plot

    def calculate_theta_dot_values(self) -> None:
        """
        Calculates the angular velocity of the pendulum
        :return: None
        """
        assert self.t_values is not None and self.theta_values is not None
        self.theta_dot_values = np.gradient(self.theta_values, self.t_values)

    def animate(self, frame):
        """
        Animates the pendulum
        :param frame: Frame number
        :return: None
        """
        assert None not in (
            self.t_values, self.theta_values, self.theta_dot_values)
        i = int(frame)
        self.time_plot.set_data(self.t_values[:i * speed],
                                self.theta_values[:i * speed])
        self.phase_plot.set_data(self.theta_values[:i * speed],
                                 self.theta_dot_values[:i * speed])
        return self.time_plot, self.phase_plot

    def plot_vector_field(self, ax: plt.Axes, x_range: tuple, y_range: tuple,
                          step_size: float = 0.01) -> None:
        """
        Plots the vector field of the pendulum
        :param ax: Axes to plot on
        :param x_range: Range of x values
        :param y_range: Range of y values
        :param step_size: Step size
        :return: None
        """
        x_values = np.arange(*x_range, step_size)
        y_values = np.arange(*y_range, step_size)
        X, Y = np.meshgrid(x_values, y_values)
        U = Y
        V = self._f(X, Y, 0)
        # Color map is the magnitude of the vector field
        C = np.sqrt(U ** 2 + V ** 2)
        ax.streamplot(X, Y, U, V, linewidth=1, density=1, color=C,
                        cmap='jet_r')


if __name__ == '__main__':
    # Set up variables
    t_end = 20
    stepsize = 0.01
    l = 1
    theta_0 = -np.pi / 2 - 2 * np.pi
    theta_dot_0 = 6
    speed = 5
    damping = 0.1

    # Set up plot
    fig, (time_ax, phase_ax) = plt.subplots(1, 2, figsize=(10, 5))
    fig: plt.Figure
    time_ax: plt.Axes
    phase_ax: plt.Axes
    time_ax.set_xlabel('Time')
    time_ax.set_ylabel('Angle')
    phase_ax.set_xlabel('Angle')
    phase_ax.set_ylabel('Angular velocity')
    time_ax.set_xlim(0, t_end)
    phase_lim_x = 4 * np.pi
    phase_lim_y = 4 * np.pi
    phase_ax.set_xlim(-phase_lim_x, phase_lim_x)
    phase_ax.set_ylim(-phase_lim_y, phase_lim_y)

    # Plot
    pendulum = Pendulum(l, theta_0, theta_dot_0)
    pendulum.solve(t_end, stepsize)
    pendulum.plot(time_ax)
    pendulum.plot_phase_space(phase_ax)
    pendulum.plot_vector_field(phase_ax, (-phase_lim_x, phase_lim_x),
                                 (-phase_lim_y, phase_lim_y), 0.01)

    # Animate
    from matplotlib.animation import FuncAnimation

    anim = FuncAnimation(fig, pendulum.animate, interval=1,
                         frames=len(pendulum.t_values))
    plt.show()
