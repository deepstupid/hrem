from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from typing import List

from sc_engine.core.model_runner import ScientificModelRunner
from cui.progress import CUIProgressHandler
from ui.common.runner import BaseUIRunner
from ui.common.utils import get_available_challenges, get_available_models
from dataset_manager import dataset_manager

class CUIRunner(BaseUIRunner):
    def __init__(self, ctx):
        self.ctx = ctx
        self.console = ctx.obj.console
        self.challenge_registry = ctx.obj.challenge_registry
        self.config_manager = ctx.obj.config_manager

    def _run_evaluation(self, challenge_id, models, patience, smoke_test=False):
        """A helper function to run an evaluation, handling validation and execution."""
        if not models:
            self.console.print("[bold red]Error: At least one model must be specified.[/bold red]")
            return

        available_challenges = {c.id: c for c in get_available_challenges(self.challenge_registry)}
        if challenge_id not in available_challenges:
            self.console.print(f"[bold red]Error: Challenge ID '{challenge_id}' not found.[/bold red]")
            self.console.print("Use 'list-challenges' to see available IDs.")
            return

        available_models = get_available_models(self.config_manager).keys()
        for model_name in models:
            if model_name not in available_models:
                self.console.print(f"[bold red]Error: Model '{model_name}' not found.[/bold red]")
                self.console.print("Use 'list-models' to see available models.")
                return

        self.console.print(Rule(f"[bold green]🚀 Launching Evaluation[/bold green]"))
        self.console.print(f"Challenge: [bold yellow]{challenge_id}[/bold yellow]")
        self.console.print(f"Models: [bold yellow]{', '.join(models)}[/bold yellow]")
        self.console.print(f"Patience: [bold yellow]{patience}[/bold yellow]")

        progress_handler = CUIProgressHandler()
        runner = ScientificModelRunner()
        runner.run(
            run_type="comparison",
            challenge_id=challenge_id,
            patience_level=patience,
            smoke_test=smoke_test,
            models=list(models),
            progress_handler=progress_handler
        )
        self.console.print("[green]✅ Evaluation completed successfully![/green]")

    def _select_challenge(self):
        """Interactively select a challenge."""
        self.console.print(Rule("[bold cyan]Step 1: Select a Challenge[/bold cyan]"))
        challenges = get_available_challenges(self.challenge_registry)
        challenge_table = Table(title="🔬 Available Scientific Challenges")
        challenge_table.add_column("Index", style="magenta")
        challenge_table.add_column("ID", style="cyan")
        challenge_table.add_column("Name", style="green")
        challenge_table.add_column("Description", style="yellow")
        for i, challenge in enumerate(challenges):
            challenge_table.add_row(str(i), challenge.id, challenge.name, challenge.description)
        self.console.print(challenge_table)
        challenge_idx = Prompt.ask("Enter the index of the challenge you want to run", choices=[str(i) for i in range(len(challenges))], show_choices=False)
        return challenges[int(challenge_idx)]

    def _select_models(self) -> List[str]:
        """Interactively select models."""
        self.console.print(Rule("[bold cyan]Step 2: Select Models[/bold cyan]"))
        model_configs = get_available_models(self.config_manager)
        models = list(model_configs.keys())
        model_table = Table(title="🤖 Available Models")
        model_table.add_column("Index", style="magenta")
        model_table.add_column("Name", style="cyan")
        for i, model_name in enumerate(models):
            model_table.add_row(str(i), model_name)
        self.console.print(model_table)
        model_indices_str = Prompt.ask("Enter the indices of the models you want to run (e.g., '0, 2')")
        selected_model_indices = [int(i.strip()) for i in model_indices_str.split(',')]
        return [models[i] for i in selected_model_indices]

    def _get_patience_level(self) -> str:
        """Interactively get the patience level."""
        self.console.print(Rule("[bold cyan]Step 3: Set Patience Level[/bold cyan]"))
        return Prompt.ask("Choose a patience level", choices=["low", "medium", "high"], default="medium")

    def run_interactive(self):
        """Interactively run a scientific evaluation."""
        self.console.print("[bold blue]Welcome to the Interactive Experiment Runner![/bold blue]")
        self.console.print("Let's set up a new scientific evaluation.")

        selected_challenge = self._select_challenge()
        selected_models = self._select_models()
        patience = self._get_patience_level()

        self._run_evaluation(selected_challenge.id, selected_models, patience)

    def run_once(self, challenge_id: str, models: List[str], patience: str, smoke_test: bool):
        """Run a scientific evaluation non-interactively."""
        self._run_evaluation(challenge_id, list(models), patience, smoke_test)

    def list_challenges(self):
        """Lists all available scientific challenges."""
        challenges = get_available_challenges(self.challenge_registry)

        if not challenges:
            self.console.print("[yellow]No challenges found.[/yellow]")
            return

        table = Table(title="🔬 Available Scientific Challenges", header_style="bold magenta")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Dataset", style="blue")

        for challenge in challenges:
            table.add_row(challenge.id, challenge.name, challenge.description, challenge.dataset.dataset)

        self.console.print(table)

    def list_models(self):
        """Lists all available models."""
        models = get_available_models(self.config_manager)

        if not models:
            self.console.print("[yellow]No models found.[/yellow]")
            return

        table = Table(title="🤖 Available Models", header_style="bold magenta")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Class Path", style="green")

        for name, model_schema in models.items():
            table.add_row(name, model_schema.algorithm_class)

        self.console.print(table)

    def list_datasets(self):
        """Lists all available datasets."""
        datasets = dataset_manager.list_available_datasets()

        if not datasets:
            self.console.print("[yellow]No datasets found.[/yellow]")
            return

        table = Table(title="📚 Available Datasets", header_style="bold magenta")
        table.add_column("Name", style="cyan", no_wrap=True)

        for dataset_name in datasets:
            table.add_row(dataset_name)

        self.console.print(table)
