from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.rule import Rule

from sc_engine.core.progress_handler import ProgressHandler
from typing import Dict, Any, Optional
import json

class CUIProgressHandler(ProgressHandler):
    """A progress handler that prints live updates to the console using rich."""

    def __init__(self):
        self.console = Console()
        self.current_algorithm: Optional[str] = None
        self.current_phase: Optional[str] = None
        self.progress_bar: Optional[Progress] = None
        self.trial_progress: Optional[int] = None

    def on_progress(self, event_type: str, data: Dict[str, Any]):
        """Receives an event and prints a corresponding message to the console."""

        # Session Events
        if event_type == 'start_session':
            challenge_name = data.get('challenge', 'Unknown Challenge')
            self.console.print(Panel(f"🔬 Starting Scientific Discovery Session for [bold]{challenge_name}[/bold]", style="bold blue"))

        # Phase Events
        elif event_type == 'start_phase':
            self.current_phase = data.get('phase', 'Unknown Phase').replace('_', ' ').title()
            self.console.print(Rule(f"[bold cyan]Phase: {self.current_phase}[/bold cyan]"))

        elif event_type == 'end_phase':
            phase = data.get('phase', 'Unknown Phase').replace('_', ' ').title()
            self.console.print(Rule(f"[bold cyan]Finished Phase: {phase}[/bold cyan]", style="dim cyan"))

        # Evaluation Events
        elif event_type == 'start_evaluation':
            title = data.get('title', 'Starting Evaluation')
            self.console.print(f"\n[bold yellow]🧪 {title}[/bold yellow]")

        elif event_type == 'end_evaluation':
            title = data.get('title', 'Finished Evaluation')
            self.console.print(f"[bold yellow]✅ {title} Complete[/bold yellow]\n")

        elif event_type == 'evaluation_cancelled':
            reason = data.get('reason', 'No reason provided.')
            self.console.print(f"[bold red]🚫 Evaluation Cancelled:[/bold red] {reason}")

        # Algorithm Events
        elif event_type == 'start_algorithm':
            self.current_algorithm = data.get('algorithm', 'Unknown Algorithm')
            progress = data.get('progress', 0)
            self.console.print(f"\n▶️  Running algorithm: [bold green]{self.current_algorithm}[/bold green] ({int(progress*100)}%)")

        elif event_type == 'end_algorithm':
            alg = data.get('algorithm', 'Unknown Algorithm')
            metrics = data.get('metrics', {})
            self.console.print(f"✅ Finished algorithm: [bold green]{alg}[/bold green]")
            if metrics:
                table = Table(title=f"{alg} Final Metrics", show_header=True, header_style="bold magenta")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="yellow")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        table.add_row(key, f"{value:.4f}")
                    else:
                        table.add_row(key, str(value))
                self.console.print(table)
            self.current_algorithm = None

        elif event_type == 'start_algorithm_step':
            alg = data.get('algorithm', 'Unknown Algorithm')
            step = data.get('step', 0)
            self.console.print(f"  [dim] Interleaved Step for {alg}: {step}[/dim]")

        # Optimization Events
        elif event_type == 'start_optimization_alg':
            alg = data.get('algorithm', 'Unknown Algorithm')
            self.console.print(f"\n[bold magenta]📈 Optimizing: {alg}[/bold magenta]")
            self.trial_progress = 0

        elif event_type == 'end_optimization_alg':
            alg = data.get('algorithm', 'Unknown Algorithm')
            best_params = data.get('best_params', {})
            self.console.print(f"\n[bold magenta]🏁 Finished Optimizing: {alg}[/bold magenta]")
            self.console.print(Panel(
                Syntax(json.dumps(best_params, indent=2), "json", theme="monokai", line_numbers=True),
                title="Best Hyperparameters",
                border_style="magenta"
            ))
            self.trial_progress = None

        elif event_type == 'skip_optimization':
            alg = data.get('algorithm', 'Unknown Algorithm')
            reason = data.get('reason', 'Not specified')
            self.console.print(f"⏭️  Skipping optimization for [yellow]{alg}[/yellow]: {reason}")

        elif event_type == 'start_trial':
            self.trial_progress += 1
            params = data.get('params', {})
            param_str = ", ".join(f"{k}={v}" for k,v in params.items())
            self.console.print(f"  [dim]Trial {self.trial_progress} starting with params: {param_str}[/dim]")

        elif event_type == 'end_trial':
            loss = data.get('loss', float('inf'))
            self.console.print(f"  [dim]Trial {self.trial_progress} finished. Loss: {loss:.4f}[/dim]")

        elif event_type == 'optimization_cancelled':
            reason = data.get('reason', 'No reason provided.')
            self.console.print(f"[bold red]🚫 Optimization Cancelled:[/bold red] {reason}")

        # Trainer Events
        elif event_type == 'trainer:train_batch':
            if self.current_algorithm:
                metrics = data.get('metrics', {})
                step = data.get('step', 0)
                total_steps = data.get('total_steps', 0)
                metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                self.console.print(f"  [dim]Step {step}/{total_steps} | {metrics_str}[/dim]")

        # Insight Events
        elif event_type == 'insights_generated':
            self.console.print(Rule("[bold green]🔬 Scientific Insights Generated[/bold green]"))
            insights = data.get('insights', [])
            report_path = data.get('report_path')

            for insight in insights:
                self.console.print(Panel(str(insight), title="Insight", border_style="green"))

            if report_path:
                self.console.print(f"\nFull report available at: [blue underline]{report_path}[/blue underline]")

        elif event_type == 'no_insights':
            self.console.print(Panel("No significant insights were generated from this run.", title="No Insights", border_style="yellow"))

        # Default handler for any other events
        else:
            # This can be used for debugging, but should ideally be empty in production
            # self.console.print(f"[dim]Event: {event_type} | Data: {data}[/dim]")
            pass
