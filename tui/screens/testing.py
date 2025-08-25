from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Button, Log, DataTable, RadioButton, RadioSet
from textual import work
import subprocess
import os


class TestingScreen(Static):
    """The screen for running unit tests."""

    def compose(self) -> ComposeResult:
        yield Static("Run unit tests for the HRM System.", classes="header")
        
        with Horizontal():
            with Vertical(classes="test-selection"):
                yield Static("Test Selection:")
                with RadioSet(id="test_selector"):
                    yield RadioButton("All Tests", value=True)
                    yield RadioButton("Config Tests")
                    yield RadioButton("Evaluation Tests")
                    yield RadioButton("Optimization Tests")
                    yield RadioButton("Integration Tests")
                    yield RadioButton("Smoke Tests Only")
                
                with Horizontal():
                    yield Button("Run Selected Tests", variant="primary", id="run_tests_button")
                    yield Button("Run All Tests", variant="success", id="run_all_tests_button")
            
            with Vertical():
                yield Static("Test Results:")
                yield Log(id="test_log", classes="log_view", auto_scroll=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "run_tests_button":
            self.run_selected_tests()
        elif event.button.id == "run_all_tests_button":
            self.run_all_tests()

    def run_selected_tests(self) -> None:
        """Run the selected tests."""
        radio_set = self.query_one("#test_selector", RadioSet)
        selected_button = radio_set.pressed_button
        
        if selected_button.label == "All Tests":
            self.run_all_tests()
        elif selected_button.label == "Config Tests":
            self.run_tests_by_file("tests/test_config.py")
        elif selected_button.label == "Evaluation Tests":
            self.run_tests_by_file("tests/test_evaluation.py")
        elif selected_button.label == "Optimization Tests":
            self.run_tests_by_file("tests/test_optimization.py")
        elif selected_button.label == "Integration Tests":
            self.run_tests_by_file("tests/test_integration.py")
        elif selected_button.label == "Smoke Tests Only":
            self.run_smoke_tests()

    def run_all_tests(self) -> None:
        """Run all tests."""
        self.run_pytest_command(["tests/"])

    def run_tests_by_file(self, file_path: str) -> None:
        """Run tests from a specific file."""
        self.run_pytest_command([file_path])

    def run_smoke_tests(self) -> None:
        """Run only smoke tests."""
        self.run_pytest_command(["tests/", "-k", "smoke"])

    def run_pytest_command(self, cmd_args: list) -> None:
        """Run pytest with the given arguments."""
        log_widget = self.query_one("#test_log", Log)
        log_widget.clear()
        log_widget.write(f"Running tests: {' '.join(cmd_args)}")
        
        # Build the command
        cmd = ["python", "-m", "pytest"] + cmd_args + ["-v"]
        
        log_widget.write(f"Command: {' '.join(cmd)}")
        self.execute_test_command(cmd)

    @work(thread=True, exclusive=True)
    def execute_test_command(self, cmd: list) -> None:
        """Execute the test command in a worker thread."""
        log_widget = self.query_one("#test_log", Log)
        
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
                self.post_message(self.LogMessage("[bold green]All tests passed![/bold green]"))
            else:
                self.post_message(self.LogMessage(f"[bold red]Tests failed with exit code {process.returncode}[/bold red]"))
                
        except Exception as e:
            self.post_message(self.LogMessage(f"[bold red]Error running tests: {str(e)}[/bold red]"))

    class LogMessage(Static):
        """A message to be posted to the log."""
        def __init__(self, message: str) -> None:
            super().__init__()
            self.message = message

    def on_testing_screen_log_message(self, message: LogMessage) -> None:
        """Handle a log message from a worker."""
        log_widget = self.query_one("#test_log", Log)
        log_widget.write(message.message)