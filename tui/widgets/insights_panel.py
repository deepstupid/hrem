from textual.widgets import Static
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from rich.panel import Panel
from rich.text import Text
from typing import List, Any, Optional

class InsightsPanel(Static):
    """A widget to display the final scientific insights from an experiment."""

    def compose(self) -> ComposeResult:
        """Render the insights panel."""
        # The content will be populated dynamically.
        yield VerticalScroll(id="insights-container")

    def show_results(self, insights: List[Any], plot_path: Optional[str], report_path: Optional[str]) -> None:
        """
        Renders the final insights, plot path, and report path into the panel.
        """
        container = self.query_one("#insights-container")
        # Clear any previous results
        container.remove_children()

        if not insights:
            container.mount(Static("[bold yellow]No significant scientific insights were generated.[/bold yellow]"))
            return

        container.mount(Static("[bold]Scientific Insights Generated:[/bold]\n"))

        for i, insight in enumerate(insights, 1):
            content = Text()
            content.append(f"Type: {insight.type.upper()}\n", style="bold")
            content.append(f"Confidence: {insight.confidence:.2f}\n\n", style="italic")
            content.append("Implications:\n", style="underline")
            for implication in insight.implications:
                content.append(f"• {implication}\n")

            panel = Panel(
                content,
                title=f"Insight {i}: {insight.name}",
                border_style="green",
                expand=True,
                padding=(1, 2)
            )
            container.mount(panel)

        if plot_path or report_path:
            container.mount(Static("\n[bold]Artifacts:[/bold]\n"))
            if plot_path:
                container.mount(Static(f"📊 Performance plot: [u]{plot_path}[/u]"))
            if report_path:
                container.mount(Static(f"📄 Scientific report: [u]{report_path}[/u]"))
