import click
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule
from typing import List, Tuple

from sc_engine.core.runner import ScientificModelRunner
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry
from sc_engine.core.ui_utils import get_available_challenges, get_available_models
from sc_engine.core.schemas import ChallengeSchema
from dataset_manager import dataset_manager
from cui.progress import CUIProgressHandler

@click.group()
@click.pass_context
def cui(ctx):
    """A console-based interface for the Scientific Discovery Engine."""
    pass

def _prompt_for_challenge(console: Console, challenge_registry: ChallengeRegistry) -> ChallengeSchema:
    """Interactively prompts the user to select a challenge."""
    console.print(Rule("[bold cyan]Step 1: Select a Challenge[/bold cyan]"))
    challenges = get_available_challenges(challenge_registry)

    table = Table(title="🔬 Available Scientific Challenges")
    table.add_column("Index", style="magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")

    for i, challenge in enumerate(challenges):
        table.add_row(str(i), challenge.id, challenge.name)
    console.print(table)

    idx = Prompt.ask("Enter the index of the challenge", choices=[str(i) for i in range(len(challenges))], show_choices=False)
    return challenges[int(idx)]

def _prompt_for_models(console: Console, config_manager: ConfigManager) -> List[str]:
    """Interactively prompts the user to select models."""
    console.print(Rule("[bold cyan]Step 2: Select Models[/bold cyan]"))
    model_configs = get_available_models(config_manager)
    models = list(model_configs.keys())

    table = Table(title="🤖 Available Models")
    table.add_column("Index", style="magenta")
    table.add_column("Name", style.cyan)

    for i, name in enumerate(models):
        table.add_row(str(i), name)
    console.print(table)

    indices_str = Prompt.ask("Enter model indices (e.g., '0, 2')")
    return [models[int(i.strip())] for i in indices_str.split(',')]

@cui.command(name="list-challenges")
@click.pass_context
def list_challenges(ctx):
    """Lists all available scientific challenges."""
    console = ctx.obj.console
    challenges = get_available_challenges(ctx.obj.challenge_registry)
    if not challenges:
        console.print("[yellow]No challenges found.[/yellow]")
        return
    table = Table(title="🔬 Available Scientific Challenges", header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Description", style="yellow")
    table.add_column("Dataset", style="blue")
    for challenge in challenges:
        table.add_row(challenge.id, challenge.name, challenge.description, challenge.dataset.dataset)
    console.print(table)

@cui.command(name="list-models")
@click.pass_context
def list_models(ctx):
    """Lists all available models."""
    console = ctx.obj.console
    models = get_available_models(ctx.obj.config_manager)
    if not models:
        console.print("[yellow]No models found.[/yellow]")
        return
    table = Table(title="🤖 Available Models", header_style="bold magenta")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Class Path", style="green")
    for name, model_schema in models.items():
        table.add_row(name, model_schema.algorithm_class)
    console.print(table)

@cui.command(name="run-interactive")
@click.pass_context
def run_interactive(ctx):
    """Interactively run a scientific evaluation."""
    console = ctx.obj.console
    console.print("[bold blue]Welcome to the Interactive Experiment Runner![/bold blue]")

    selected_challenge = _prompt_for_challenge(console, ctx.obj.challenge_registry)
    selected_models = _prompt_for_models(console, ctx.obj.config_manager)

    console.print(Rule("[bold cyan]Step 3: Set Patience Level[/bold cyan]"))
    patience = Prompt.ask("Choose a patience level", choices=["low", "medium", "high"], default="medium")

    console.print(Rule(f"[bold green]🚀 Launching Evaluation[/bold green]"))
    console.print(f"Challenge: [bold yellow]{selected_challenge.name}[/bold yellow]")
    console.print(f"Models: [bold yellow]{', '.join(selected_models)}[/bold yellow]")
    console.print(f"Patience: [bold yellow]{patience}[/bold yellow]")

    runner = ScientificModelRunner()
    runner.run(
        run_type="comparison",
        challenge_id=selected_challenge.id,
        patience_level=patience,
        smoke_test=False,
        models=selected_models,
        progress_handler=CUIProgressHandler()
    )
    console.print("[green]✅ Interactive evaluation completed successfully![/green]")

@cui.command(name="list-datasets")
def list_datasets():
    """Lists all available datasets."""
    console = Console()
    datasets = dataset_manager.list_available_datasets()
    if not datasets:
        console.print("[yellow]No datasets found.[/yellow]")
        return
    table = Table(title="📚 Available Datasets", header_style="bold magenta")
    table.add_column("Name", style="cyan", no_wrap=True)
    for dataset_name in datasets:
        table.add_row(dataset_name)
    console.print(table)

@cui.command(name="run-once")
@click.option('--challenge-id', required=True, help='The ID of the challenge to run.')
@click.option('--model', 'models', multiple=True, help='The name of a model to run. Can be specified multiple times.')
@click.option('--patience', type=click.Choice(['low', 'medium', 'high']), default='medium', help='The patience level for the run.')
@click.option('--smoke-test', is_flag=True, default=False, help='Run in smoke test mode.')
@click.pass_context
def run_once(ctx, challenge_id, models, patience, smoke_test):
    """Run a scientific evaluation non-interactively."""
    console = ctx.obj.console

    console.print(Rule(f"[bold green]🚀 Launching Non-Interactive Evaluation[/bold green]"))
    console.print(f"Challenge: [bold yellow]{challenge_id}[/bold yellow]")
    console.print(f"Models: [bold yellow]{', '.join(models)}[/bold yellow]")
    console.print(f"Patience: [bold yellow]{patience}[/bold yellow]")
    console.print(f"Smoke Test: [bold yellow]{smoke_test}[/bold yellow]")

    runner = ScientificModelRunner()
    try:
        runner.run(
            run_type="comparison",
            challenge_id=challenge_id,
            patience_level=patience,
            smoke_test=smoke_test,
            models=list(models),
            progress_handler=CUIProgressHandler()
        )
        console.print("[green]✅ Non-interactive evaluation completed successfully![/green]")
    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

if __name__ == "__main__":
    cui()
