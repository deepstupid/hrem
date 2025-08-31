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

    def compose(self) -> ComposeResult:
        """Create child widgets for the dashboard."""
        yield Static("Discovery Dashboard for Challenge: [u]Synthetic Sort[/u]", id="dashboard-header", classes="header")
        with Horizontal(id="main-container"):
            with VerticalScroll(id="left-pane"):
                yield Static(id="patience-insight-graph")
                yield Static(id="exploration-space")
            with VerticalScroll(id="right-pane"):
                yield Static(id="insights-panel")

    def on_mount(self) -> None:
        """Set up the dashboard when it's mounted."""
        self.update_insights_table()
        self.update_plot()
        self.update_exploration_space()

    def update_insights_table(self):
        """Creates and renders a rich Table for scientific insights."""
        insights_panel = self.query_one("#insights-panel", Static)

        table = Table(expand=True)
        table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
        table.add_column("Insight", style="magenta", width=40)
        table.add_column("Category", style="green")
        table.add_column("Confidence", justify="right", style="yellow")
        table.add_column("Potential", justify="right")

        # Sample data
        insights_data = [
            (1, "HREM outperforms HRM on long sequences", "Performance", 0.95, 0.9),
            (2, "HRM is more efficient on short sequences", "Efficiency", 0.87, 0.8),
            (3, "HREM shows better generalization", "Generalization", 0.82, 0.75),
            (4, "HRM converges faster on simple tasks", "Convergence", 0.91, 0.7),
        ]

        for rank, insight, category, confidence, potential in insights_data:
            potential_style = "bold green" if potential > 0.8 else "yellow" if potential > 0.7 else "white"
            table.add_row(
                str(rank),
                insight,
                category,
                f"{confidence:.2f}",
                Text(f"{potential:.2f}", style=potential_style)
            )

        insights_panel.update(Panel(table, title="[b]Scientific Insights[/b]", border_style="blue", padding=(1, 2)))

    def update_exploration_space(self):
        """Generates and renders a PCA plot of the hyperparameter space."""
        space_view = self.query_one("#exploration-space", Static)

        # Sample hyperparameter data
        hparams = [
            {'lr': 0.01, 'dropout': 0.1, 'epochs': 10},
            {'lr': 0.005, 'dropout': 0.2, 'epochs': 20},
            {'lr': 0.02, 'dropout': 0.15, 'epochs': 15},
            {'lr': 0.015, 'dropout': 0.25, 'epochs': 25},
            {'lr': 0.008, 'dropout': 0.3, 'epochs': 30},
        ]

        pca_result = get_pca_2d(hparams)

        if pca_result.size == 0:
            space_view.update(Panel("[No data for exploration space]", title="[b]Exploration Space (PCA)[/b]", border_style="blue"))
            return

        x = pca_result[:, 0]
        y = pca_result[:, 1]
        performance = np.random.rand(len(hparams))
        colors = ['red' if p < 0.3 else 'yellow' if p < 0.7 else 'green' for p in performance]

        plt.clf()
        plt.scatter(x, y, color=colors)
        plt.plotsize(60, 20)
        plt.title("Hyperparameter Exploration")

        space_view.update(Panel(plt.build(), title="[b]Exploration Space (PCA)[/b]", border_style="blue"))

    def update_plot(self):
        """Generates and renders a plot of patience vs. insights."""
        graph = self.query_one("#patience-insight-graph", Static)

        patience = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        insights = [0, 1, 1, 2, 3, 5, 6, 8, 9, 10]

        plt.clf()
        plt.plot(patience, insights, color='cyan')
        plt.plotsize(60, 20)
        plt.title("Patience vs. Insight Curve")

        graph.update(Panel(plt.build(), title="[b]Discovery Progress[/b]", border_style="blue"))
