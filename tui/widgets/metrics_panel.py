from textual.widgets import Static, DataTable
from textual.app import ComposeResult
from typing import Dict, Any

class MetricsPanel(Static):
    """A widget to display algorithm metrics in a table."""

    def compose(self) -> ComposeResult:
        """Render the metrics panel."""
        yield DataTable(id="metrics-table")

    def on_mount(self) -> None:
        """Initialize the DataTable."""
        table = self.query_one(DataTable)
        table.add_columns("Algorithm", "Metric", "Value")

    def update_metrics(self, algorithm: str, metrics: Dict[str, Any]) -> None:
        """Updates the table with new metrics."""
        table = self.query_one(DataTable)
        for key, value in metrics.items():
            display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            # A unique key for each row to allow updates
            row_key = f"{algorithm}_{key}"

            # Check if the row already exists
            if table.is_valid_row_index(table.get_row_index(row_key)):
                 table.update_cell(row_key, "Value", display_value)
            else:
                table.add_row(algorithm, key, display_value, key=row_key)

    def clear_metrics(self) -> None:
        """Clears all rows from the metrics table."""
        self.query_one(DataTable).clear()
