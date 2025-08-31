from typing import List, Dict, Tuple, Any
import plotext as plt
import numpy as np
from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Horizontal, VerticalScroll
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..pca_utils import get_pca_2d


class DiscoveryDashboard(Static):
    """A dashboard to visualize the scientific discovery process."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.insights_data: List[Tuple[int, str, str, float, float]] = []
        self.hparam_points: List[Dict[str, Any]] = []
        self.patience_curve_data: List[Tuple[float, int]] = []
        self.insights_table = self._create_insights_table()

    def compose(self) -> ComposeResult:
        """Create child widgets for the dashboard."""
        yield Static("Discovery Dashboard", id="dashboard-header", classes="header")
        with Horizontal(id="main-container"):
            with VerticalScroll(id="left-pane"):
                yield Static(id="patience-insight-graph")
                yield Static(id="exploration-space")
            with VerticalScroll(id="right-pane"):
                yield Static(id="insights-panel")

    def on_mount(self) -> None:
        """Set up the dashboard when it's mounted."""
        self.update_insights_panel()
        self.update_patience_plot()
        self.update_exploration_space_plot()

    def _create_insights_table(self) -> Table:
        """Creates the initial structure of the insights table."""
        table = Table(expand=True)
        table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
        table.add_column("Insight", style="magenta", width=40)
        table.add_column("Category", style="green")
        table.add_column("Confidence", justify="right", style="yellow")
        table.add_column("Potential", justify="right")
        return table

    # --- Public API for updating dashboard ---

    def set_challenge(self, challenge_name: str):
        """Sets the header for the current challenge."""
        header = self.query_one("#dashboard-header", Static)
        header.update(f"Discovery Dashboard for Challenge: [u]{challenge_name}[/u]")

    def add_insight(self, insight: Dict[str, Any]):
        """Adds a new insight to the table and refreshes."""
        new_insight_tuple = (
            insight.get("rank", len(self.insights_data) + 1),
            insight.get("text", "N/A"),
            insight.get("category", "N/A"),
            insight.get("confidence", 0.0),
            insight.get("potential", 0.0),
        )
        self.insights_data.append(new_insight_tuple)
        self.update_insights_panel()

    def add_hparam_point(self, point: Dict[str, Any]):
        """Adds a new hyperparameter point and refreshes the plot."""
        self.hparam_points.append(point)
        self.update_exploration_space_plot()

    def add_patience_point(self, point: Tuple[float, int]):
        """Adds a new point to the patience curve."""
        self.patience_curve_data.append(point)
        self.update_patience_plot()

    def clear_all(self):
        """Clears all data from the dashboard."""
        self.insights_data.clear()
        self.hparam_points.clear()
        self.patience_curve_data.clear()
        self.update_insights_panel()
        self.update_patience_plot()
        self.update_exploration_space_plot()

    # --- Internal update methods ---

    def update_insights_panel(self):
        """Renders the insights table with the current data."""
        insights_panel = self.query_one("#insights-panel", Static)
        self.insights_table.rows.clear()

        if not self.insights_data:
            content = Text("Waiting for insights...", justify="center")
        else:
            sorted_insights = sorted(self.insights_data, key=lambda x: x[0])
            for rank, insight, category, confidence, potential in sorted_insights:
                potential_style = "bold green" if potential > 0.8 else "yellow" if potential > 0.7 else "white"
                self.insights_table.add_row(
                    str(rank),
                    insight,
                    category,
                    f"{confidence:.2f}",
                    Text(f"{potential:.2f}", style=potential_style)
                )
            content = self.insights_table

        insights_panel.update(Panel(content, title="[b]Scientific Insights[/b]", border_style="blue", padding=(1, 2)))

    def update_exploration_space_plot(self):
        """Generates and renders a PCA plot of the hyperparameter space."""
        space_view = self.query_one("#exploration-space", Static)

        if not self.hparam_points or len(self.hparam_points) < 2:
            content = Text("Not enough hyperparameter data for PCA yet.", justify="center")
        else:
            try:
                hparams = [d['params'] for d in self.hparam_points]
                performance = [d['performance'] for d in self.hparam_points]
                pca_result = get_pca_2d(hparams)

                if pca_result.size == 0:
                    content = Text("PCA calculation failed.", justify="center")
                else:
                    x = pca_result[:, 0]
                    y = pca_result[:, 1]
                    colors = ['red' if p < 0.3 else 'yellow' if p < 0.7 else 'green' for p in performance]

                    plt.clf()
                    plt.scatter(x, y, color=colors)
                    plt.plotsize(60, 20)
                    plt.title("Hyperparameter Exploration")
                    content = plt.build()
            except Exception:
                content = Text("Error generating PCA plot.", justify="center")

        space_view.update(Panel(content, title="[b]Exploration Space (PCA)[/b]", border_style="blue"))

    def update_patience_plot(self):
        """Generates and renders a plot of patience vs. insights."""
        graph = self.query_one("#patience-insight-graph", Static)

        if not self.patience_curve_data:
            content = Text("Awaiting discovery progress...", justify="center")
        else:
            self.patience_curve_data.sort(key=lambda x: x[0])
            patience, insights = zip(*self.patience_curve_data)
            plt.clf()
            plt.plot(list(patience), list(insights), color='cyan')
            plt.plotsize(60, 20)
            plt.title("Patience vs. Insight Curve")
            content = plt.build()

        graph.update(Panel(content, title="[b]Discovery Progress[/b]", border_style="blue"))
