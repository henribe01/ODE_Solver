import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Animation(FuncAnimation):
    """
    Class to animate data on multiple subplots.
    """

    def __init__(self, fig: plt.Figure, speed: int = 1, interval: int = 30) -> None:
        """
        :param fig: The figure to animate.
        :param speed: The speed of the animation.
        :param interval: The interval between frames in milliseconds.
        """
        self.fig = fig
        self._objects = {}
        self.speed = speed
        super().__init__(fig, self._update, interval=interval, blit=True)

    def add_line(self, ax: plt.Axes, x: list[float], y: list[float],
                 color: str = 'k', label: str = '') -> plt.Line2D:
        """
        Adds a line to the animation.
        :param ax: The axes to add the line to.
        :param x: The x values of the line.
        :param y: The y values of the line.
        :param color: The color of the line.
        :param label: The label of the line.
        :return: The line object.
        """
        line = ax.plot([], [], color=color, label=label)[0]
        self._objects[line] = (x, y)
        return line

    def _update(self, frame: int) -> list[plt.Artist]:
        """
        Updates the animation.
        :param frame: The frame number.
        :return: A list of artists to update.
        """
        for line in self._objects:
            frame = min(frame, len(self._objects[line][0]) - 1) % len(
                self._objects[line][0])
            line.set_data(self._objects[line][0][:frame * self.speed],
                          self._objects[line][1][:frame * self.speed])
        return list(self._objects.keys())
