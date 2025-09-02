import plotext as plt
from textual.widgets import Static
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from rich.panel import Panel
from rich.text import Text
from typing import List, Any, Optional, Dict

class InsightsPanel(Static):
    """A widget to display the final scientific insights from an experiment."""

    def compose(self) -> ComposeResult:
        """Render the insights panel."""
        yield VerticalScroll(id="insights-container")

    def _generate_plot(self, log_histories: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generates a textual plot from log histories."""
        plt.clf()
        for alg, history in log_histories.items():
            steps = [record.get('step', 0) for record in history if 'train/lm_loss' in record]
            losses = [record.get('train/lm_loss') for record in history if 'train/lm_loss' in record]
            if steps and losses:
                plt.plot(steps, losses, label=alg)

        plt.title("Training Performance")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        return plt.build()

    def show_results(
        self,
        insights: List[Any],
        plot_path: Optional[str],
        report_path: Optional[str],
        log_histories: Optional[Dict[str, List[Dict[str, Any]]]]
    ) -> None:
        """
        Renders the final insights, plot, and report path into the panel.
        """
        container = self.query_one("#insights-container")
        container.remove_children()

        # 1. Display the plot
        if log_histories:
            plot_text = self._generate_plot(log_histories)
            container.mount(Static(plot_text, id="performance-plot"))
        elif plot_path:
             container.mount(Static(f"📊 Performance plot available at: [u]{plot_path}[/u]"))


        # 2. Display the insights
        if not insights:
            container.mount(Static("[bold yellow]No significant scientific insights were generated.[/bold yellow]"))
        else:
            container.mount(Static("\n[bold]Scientific Insights Generated:[/bold]\n"))
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

        # 3. Display artifacts
        if report_path:
            container.mount(Static("\n[bold]Artifacts:[/bold]\n"))
            container.mount(Static(f"📄 Scientific report: [u]{report_path}[/u]"))
