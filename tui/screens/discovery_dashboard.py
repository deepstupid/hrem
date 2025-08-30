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
                yield Static("[TODO: Exploration space visualization]", id="exploration-space")
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
        space_view = self.query_one("#exploration-space", Static)

        # Sample data
        space = [
            ['O', 'O', '.', '.'],
            ['O', 'X', 'O', '.'],
            ['.', 'O', '.', '.'],
            ['.', '.', '.', '.'],
        ]

        space_str = ""
        for row in space:
            space_str += " ".join(row) + "\n"

        space_view.update(space_str)

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
