import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from collections import defaultdict
import numpy as np

class PlotWidget(QWidget):
    """
    A custom widget that embeds a Matplotlib plot.
    It provides methods to dynamically add and update lines on the plot.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Matplotlib Figure ---
        # plt.style.use('seaborn-v0_8-whitegrid')
        self.figure = plt.figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # --- Data Storage ---
        self.lines = {}
        self.data = defaultdict(lambda: {'x': [], 'y': []})

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def add_point(self, line_name, x_val, y_val, label=None):
        """Adds a single point to a line's data and redraws the plot."""
        if line_name not in self.lines:
            # Create a new line if it doesn't exist
            self.lines[line_name], = self.ax.plot(
                [], [], marker='o', linestyle='-', label=label or line_name
            )
            self.ax.legend()

        # Append data
        self.data[line_name]['x'].append(x_val)
        self.data[line_name]['y'].append(y_val)

        # Update plot data
        self.lines[line_name].set_data(self.data[line_name]['x'], self.data[line_name]['y'])

        # Rescale and redraw
        self._rescale_and_redraw()

    def _rescale_and_redraw(self):
        """Helper to rescale axes and redraw the canvas."""
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def clear_plot(self):
        """Removes all lines and data from the plot."""
        # Clear data
        self.data.clear()
        self.lines.clear()

        # Clear axes
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Live Metrics")

        # Redraw the empty plot
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def initialize_plot(self):
        """Sets up the initial state of the plot."""
        self.clear_plot()
