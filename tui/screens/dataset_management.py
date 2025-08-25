import os
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Input, Button, Log, DirectoryTree, DataTable
from textual import work


class DatasetManagementScreen(Static):
    """The screen for managing datasets."""

    def compose(self) -> ComposeResult:
        yield Static("Manage and build datasets for HRM System.", classes="header")
        
        with Horizontal():
            with Vertical(classes="dataset-tree"):
                yield Static("Dataset Directory:")
                yield DirectoryTree("./data", id="dataset_tree")
            
            with Vertical(classes="dataset-controls"):
                yield Static("Dataset Builder:")
                yield Static("Dataset Type:")
                with Horizontal():
                    yield Button("Synthetic", id="build_synthetic")
                    yield Button("ARC", id="build_arc")
                    yield Button("Sudoku", id="build_sudoku")
                    yield Button("Maze", id="build_maze")
                
                yield Static("Options:")
                with Horizontal():
                    yield Input(placeholder="Output directory", id="output_dir", value="data/custom")
                    yield Input(placeholder="Number of samples", id="num_samples", value="1000")
                
                yield Button("Build Dataset", variant="primary", id="build_dataset")
                
                yield Static("Status:")
                yield Log(id="dataset_log", classes="log_view", auto_scroll=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        log_widget = self.query_one("#dataset_log", Log)
        
        if event.button.id == "build_synthetic":
            self.build_dataset("synthetic")
        elif event.button.id == "build_arc":
            self.build_dataset("arc")
        elif event.button.id == "build_sudoku":
            self.build_dataset("sudoku")
        elif event.button.id == "build_maze":
            self.build_dataset("maze")
        elif event.button.id == "build_dataset":
            # Get selected dataset type from context or default to synthetic
            self.build_dataset("synthetic")

    def build_dataset(self, dataset_type: str) -> None:
        """Build a dataset of the specified type."""
        log_widget = self.query_one("#dataset_log", Log)
        log_widget.clear()
        
        output_dir = self.query_one("#output_dir", Input).value
        num_samples = self.query_one("#num_samples", Input).value
        
        log_widget.write(f"Building {dataset_type} dataset...")
        log_widget.write(f"Output directory: {output_dir}")
        log_widget.write(f"Number of samples: {num_samples}")
        
        # Build command
        script_map = {
            "synthetic": "dataset/build_synthetic_dataset.py",
            "arc": "dataset/build_arc_dataset.py",
            "sudoku": "dataset/build_sudoku_dataset.py",
            "maze": "dataset/build_maze_dataset.py"
        }
        
        script_path = script_map.get(dataset_type)
        if not script_path:
            log_widget.write(f"[bold red]Unknown dataset type: {dataset_type}[/bold red]")
            return
            
        if not os.path.exists(script_path):
            log_widget.write(f"[bold red]Dataset builder script not found: {script_path}[/bold red]")
            return
            
        cmd = ["python", script_path, f"--output-dir={output_dir}"]
        
        if dataset_type == "synthetic":
            cmd.append(f"--num-samples={num_samples}")
        
        log_widget.write(f"Command: {' '.join(cmd)}")
        self.execute_build_command(cmd, dataset_type, output_dir)

    @work(thread=True, exclusive=True)
    def execute_build_command(self, cmd: list, dataset_type: str, output_dir: str) -> None:
        """Execute the build command in a worker thread."""
        import subprocess
        
        log_widget = self.query_one("#dataset_log", Log)
        
        try:
            # Run the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd()
            )
            
            # Stream output to the log
            for line in process.stdout:
                self.post_message(self.LogMessage(line.rstrip()))
            
            # Wait for completion
            process.wait()
            
            if process.returncode == 0:
                self.post_message(self.LogMessage("[bold green]Dataset built successfully![/bold green]"))
                self.post_message(self.LogMessage(f"Dataset location: {output_dir}"))
            else:
                self.post_message(self.LogMessage(f"[bold red]Dataset build failed with exit code {process.returncode}[/bold red]"))
                
        except Exception as e:
            self.post_message(self.LogMessage(f"[bold red]Error building dataset: {str(e)}[/bold red]"))

    class LogMessage(Static):
        """A message to be posted to the log."""
        def __init__(self, message: str) -> None:
            super().__init__()
            self.message = message

    def on_dataset_management_screen_log_message(self, message: LogMessage) -> None:
        """Handle a log message from a worker."""
        log_widget = self.query_one("#dataset_log", Log)
        log_widget.write(message.message)