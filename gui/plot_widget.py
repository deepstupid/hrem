import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from collections import defaultdict

class PlotWidget(QWidget):
    """
    A custom widget that embeds a pyqtgraph plot.
    It provides methods to dynamically add and update lines on the plot,
    optimized for live data.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # --- pyqtgraph PlotWidget ---
        self.plot_widget = pg.PlotWidget()

        # --- Data Storage ---
        self.lines = {}
        self.data = defaultdict(lambda: {'x': [], 'y': []})

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)

        self.initialize_plot()

    def add_point(self, line_name, x_val, y_val, label=None):
        """Adds a single point to a line's data and updates the plot."""
        if line_name not in self.lines:
            # Create a new line if it doesn't exist
            # Use different colors for different lines for better visibility
            pen = pg.mkPen(color=(len(self.lines) % 9, 9), width=2)
            self.lines[line_name] = self.plot_widget.plot(
                [], [], pen=pen, name=label or line_name
            )

        # Append data
        self.data[line_name]['x'].append(x_val)
        self.data[line_name]['y'].append(y_val)

        # Update plot data
        self.lines[line_name].setData(self.data[line_name]['x'], self.data[line_name]['y'])

    def clear_plot(self):
        """Removes all lines and data from the plot."""
        self.plot_widget.clear()
        self.data.clear()
        self.lines.clear()
        self.initialize_plot() # Re-initialize plot styles

    def initialize_plot(self):
        """Sets up the initial state and style of the plot."""
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Live Metrics", color='k', size='12pt')
        self.plot_widget.setLabel('left', 'Value', color='k')
        self.plot_widget.setLabel('bottom', 'Step', color='k')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()
