import plotext as plt
from textual.app import ComposeResult
from textual.widgets import Static, DataTable
from textual.containers import Horizontal, Vertical

class DiscoveryDashboard(Static):
    """A dashboard to visualize the scientific discovery process."""

    def compose(self) -> ComposeResult:
        """Create child widgets for the dashboard."""
        with Horizontal():
            with Vertical(id="left-pane"):
                yield Static("Patience vs. Insight Graph", classes="header")
                yield Static(id="patience-insight-graph")
                yield Static("Exploration Space", classes="header")
                yield Static(id="exploration-space")
            with Vertical(id="right-pane"):
                yield Static("Ranked Insights", classes="header")
                yield DataTable(id="insights-table")

    def on_mount(self) -> None:
        """Set up the dashboard when it's mounted."""
        # Set up table
        table = self.query_one(DataTable)
        table.add_columns("Rank", "Insight", "Potential")
        table.add_row("1", "HREM outperforms HRM on long sequences", "0.9")
        table.add_row("2", "HRM is more efficient on short sequences", "0.8")

        # Set up plot
        self.update_plot()
        self.update_exploration_space()

    def update_exploration_space(self):
        from ..pca_utils import get_pca_2d
        import numpy as np

        space_view = self.query_one("#exploration-space", Static)

        # Sample hyperparameter data
        hparams = [
            {'lr': 0.01, 'dropout': 0.1, 'epochs': 10},
            {'lr': 0.005, 'dropout': 0.2, 'epochs': 20},
            {'lr': 0.02, 'dropout': 0.15, 'epochs': 15},
            {'lr': 0.015, 'dropout': 0.25, 'epochs': 25},
            {'lr': 0.008, 'dropout': 0.3, 'epochs': 30},
        ]

        # Get 2D coordinates from PCA
        pca_result = get_pca_2d(hparams)

        if pca_result.size == 0:
            space_view.update("[No data for exploration space]")
            return

        x = pca_result[:, 0]
        y = pca_result[:, 1]

        # Sample performance data (e.g., accuracy)
        performance = np.random.rand(len(hparams))
        colors = ['red' if p < 0.3 else 'yellow' if p < 0.7 else 'green' for p in performance]

        plt.clf()
        plt.scatter(x, y, color=colors)
        plt.title("Hyperparameter Space (PCA)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

        space_view.update(plt.build())

    def update_plot(self):
        graph = self.query_one("#patience-insight-graph", Static)

        # Sample data
        patience = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        insights = [0, 1, 1, 2, 3, 5, 6, 8, 9, 10]

        plt.clf()
        plt.plot(patience, insights)
        plt.title("Patience vs. Insight")
        plt.xlabel("Patience (minutes)")
        plt.ylabel("Insights Gained")

        graph.update(plt.build())
