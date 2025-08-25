import numpy as np
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Input, Button, Log

class DatasetVisualizer(Static):
    """A screen to visualize datasets."""

    def compose(self) -> ComposeResult:
        yield Static("Enter the path to a dataset folder to visualize it.", classes="header")
        yield Input(placeholder="/path/to/your/dataset/folder", id="dataset_path_input")
        yield Button("Load Dataset", variant="primary", id="load_dataset_button")
        yield Log(id="visualizer_log", classes="log_view", auto_scroll=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "load_dataset_button":
            self.load_dataset()

    def load_dataset(self) -> None:
        log = self.query_one("#visualizer_log", Log)
        log.clear()

        path_str = self.query_one("#dataset_path_input", Input).value
        if not path_str:
            log.write("[bold red]Please enter a path.[/bold red]")
            return

        dataset_path = Path(path_str)
        if not dataset_path.is_dir():
            log.write(f"[bold red]Error: Path is not a directory: {dataset_path}[/bold red]")
            return

        log.write(f"Loading dataset from: {dataset_path}")

        try:
            # In a real app, we would now parse the .npy files.
            # For now, we just show a success message.
            # Example of what would go here:
            # train_inputs_path = dataset_path / "train" / "all__inputs.npy"
            # if train_inputs_path.exists():
            #     inputs_arr = np.load(train_inputs_path)
            #     log.write(f"Loaded train/all__inputs.npy with shape: {inputs_arr.shape}")
            # else:
            #     log.write(f"[bold yellow]Warning: {train_inputs_path} not found.[/bold yellow]")

            log.write("[bold green]Dataset loaded (simulation).[/bold green]")
            log.write("Rendering logic will be implemented next.")

        except Exception as e:
            log.write(f"[bold red]Failed to load dataset: {e}[/bold red]")
