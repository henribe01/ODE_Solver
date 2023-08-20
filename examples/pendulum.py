import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
plt.style.use('seaborn-v0_8')

from ODE import ODE
from _example import Example
from matplotlib.animation import FuncAnimation


class Pendulum(Example):
    def __init__(self, l: float, theta_0: float, theta_dot_0: float,
                 g: float = 9.81,damping: float = 0, speed: float = 1):
        """
        :param l: Length of the pendulum
        :param theta_0: Initial angle
        :param theta_dot_0: Initial angular velocity
        :param g: Acceleration due to gravity
        :param damping: Damping coefficient
        :param speed: Speed of the animation
        """
        self.l = l
        self.damping = damping
        self.g = g
        super().__init__([theta_0, theta_dot_0], 0, speed)

    def _f(self, theta: float, theta_dot: float, t: float) -> list[float]:
        """
        :param theta: Angle
        :param theta_dot: Angular velocity
        :param t: Time
        :return: [theta_dot, theta_dot_dot]
        """
        return -self.g / self.l * np.sin(theta) - damping * theta_dot / self.l ** 2


if __name__ == '__main__':
    # Set up variables
    t_end = 20
    step_size = 0.01
    l = 1
    theta_0 = np.pi / 2
    theta_dot_0 = 0
    speed = 5
    damping = 0.1
    phase_limits = [(-4 * np.pi, 4 * np.pi), (-4 * np.pi, 4 * np.pi)]

    # Set up plot
    fig, (time_ax, phase_ax) = plt.subplots(1, 2, figsize=(10, 5))
    time_ax.set_xlabel('Time')
    time_ax.set_ylabel('Angle')
    time_ax.set_title('Angle over time')
    phase_ax.set_xlabel('Angle')
    phase_ax.set_ylabel('Angular velocity')
    phase_ax.set_title('Phase space')
    time_ax.set_xlim(0, t_end)
    phase_ax.set_xlim(*phase_limits[0])
    phase_ax.set_ylim(*phase_limits[1])

    # Plot
    pendulum = Pendulum(l, theta_0, theta_dot_0, damping=damping, speed=speed)
    pendulum.solve(t_end, step_size, method='euler')
    pendulum.plot_time(time_ax)
    pendulum.plot_stream_phase_space(phase_ax, phase_limits[0],
                                     phase_limits[1], density=(2, 2))
    pendulum.plot_phase_space(phase_ax)

    # Animate
    ani = FuncAnimation(fig, pendulum.animate, interval=1,
                        frames=len(pendulum.t_values))

    plt.show()
