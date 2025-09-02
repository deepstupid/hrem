import traceback
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Header, Footer, Button, Log
from textual.containers import Container

class ErrorScreen(Screen):
    """A modal screen to display a fatal error from the experiment worker."""

    def __init__(
        self,
        error: Exception,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        self.error = error
        super().__init__(name=name, id=id, classes=classes)

    def compose(self) -> ComposeResult:
        """Render the error screen."""
        yield Header()
        with Container(id="error-container"):
            yield Log(id="error-log", highlight=True)
            yield Button("Return to Setup", variant="primary", id="return-button")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the error log with the traceback."""
        log = self.query_one(Log)
        log.write("[bold red]An unexpected error occurred in the experiment![/bold red]\n")
        log.write(traceback.format_exc())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the return button press."""
        if event.button.id == "return-button":
            self.dismiss()
