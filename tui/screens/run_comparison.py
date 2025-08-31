import os
import json
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Checkbox, Button, Static, Label
from textual.containers import Vertical, Horizontal, VerticalScroll
from rich.table import Table
from rich.panel import Panel

LOG_DIR = "logs"

class RunComparisonScreen(Screen):
    """A screen to compare historical experiment runs."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Horizontal(id="comparison-container", classes="main-container"):
            with VerticalScroll(id="selection-pane"):
                yield Label("Select runs to compare:", classes="header")
                yield Vertical(id="run-checklist-container")
                yield Button("Compare Selected Runs", id="compare-button", variant="primary")
            with VerticalScroll(id="results-pane"):
                yield Static(id="comparison-results")

    def on_mount(self) -> None:
        """Populate the checklist with available log files."""
        checklist_container = self.query_one("#run-checklist-container")
        log_files = self._find_log_files()
        for i, log_file in enumerate(log_files):
            # Use Checkbox instead of Checklist for more control
            checkbox = Checkbox(log_file, id=f"run_checkbox_{i}")
            checkbox.meta['path'] = log_file
            yield checklist_container.mount(checkbox)

    def _find_log_files(self) -> list[str]:
        """Find all .json log files in the log directory."""
        json_files = []
        for root, _, files in os.walk(LOG_DIR):
            for file in files:
                if file.endswith(".json"):
                    relative_path = os.path.relpath(os.path.join(root, file), start=os.getcwd())
                    json_files.append(relative_path)
        return sorted(json_files)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the compare button press."""
        if event.button.id == "compare-button":
            selected_runs = []
            for checkbox in self.query(Checkbox):
                if checkbox.value:
                    selected_runs.append(checkbox.meta['path'])

            if not selected_runs:
                self.query_one("#comparison-results", Static).update(Panel("Please select at least one run to compare.", title="[b]Comparison Results[/b]", border_style="red"))
                return

            self.update_comparison_table(selected_runs)

    def _parse_log_file(self, file_path: str) -> dict:
        """Parse a log file and extract key summary metrics."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Find the summary dictionary, which is the last element containing an "all" key
            summary = {}
            for entry in reversed(data):
                if "all" in entry:
                    summary = entry["all"]
                    break

            return {
                "file": os.path.basename(file_path),
                "accuracy": summary.get("accuracy"),
                "exact_accuracy": summary.get("exact_accuracy"),
                "lm_loss": summary.get("lm_loss"),
                "steps": summary.get("steps"),
            }
        except (json.JSONDecodeError, IndexError, FileNotFoundError):
            return {"file": os.path.basename(file_path), "error": "Could not parse"}

    def update_comparison_table(self, selected_runs: list[str]) -> None:
        """Update the results panel with a comparison table."""
        results_panel = self.query_one("#comparison-results", Static)

        table = Table(title="Run Comparison", expand=True, show_header=True, header_style="bold magenta")
        table.add_column("Run", style="cyan", min_width=20)
        table.add_column("Accuracy", justify="right")
        table.add_column("Exact Acc.", justify="right")
        table.add_column("LM Loss", justify="right")
        table.add_column("Steps", justify="right")

        for run_path in selected_runs:
            metrics = self._parse_log_file(run_path)
            if "error" in metrics or any(v is None for v in metrics.values()):
                table.add_row(metrics["file"], "[red]Parse Error[/red]", "[red]N/A[/red]", "[red]N/A[/red]", "[red]N/A[/red]")
            else:
                table.add_row(
                    metrics["file"],
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['exact_accuracy']:.4f}",
                    f"{metrics['lm_loss']:.4f}",
                    str(metrics['steps'])
                )

        results_panel.update(Panel(table, title="[b]Comparison Results[/b]", border_style="green"))
