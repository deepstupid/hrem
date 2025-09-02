import click
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule

from sc_engine.core.model_runner import ScientificModelRunner
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry
from sc_engine.core.ui_utils import get_available_challenges, get_available_models
from dataset_manager import dataset_manager
from cui.progress import CUIProgressHandler
from .interactive import run_interactive_session

@click.group()
@click.pass_context
def cui(ctx):
    """A console-based interface for the Scientific Discovery Engine."""
    # The context is now passed down from the main CLI group in run.py
    pass

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
    run_interactive_session(ctx)


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

    if not models:
        console.print("[bold red]Error: At least one --model must be specified.[/bold red]")
        return

    available_challenges = {c.id: c for c in get_available_challenges(ctx.obj.challenge_registry)}
    if challenge_id not in available_challenges:
        console.print(f"[bold red]Error: Challenge ID '{challenge_id}' not found.[/bold red]")
        console.print("Use 'list-challenges' to see available IDs.")
        return

    available_models = get_available_models(ctx.obj.config_manager).keys()
    for model_name in models:
        if model_name not in available_models:
            console.print(f"[bold red]Error: Model '{model_name}' not found.[/bold red]")
            console.print("Use 'list-models' to see available models.")
            return

    console.print(Rule(f"[bold green]🚀 Launching Non-Interactive Evaluation[/bold green]"))
    console.print(f"Challenge: [bold yellow]{challenge_id}[/bold yellow]")
    console.print(f"Models: [bold yellow]{', '.join(models)}[/bold yellow]")
    console.print(f"Patience: [bold yellow]{patience}[/bold yellow]")
    console.print(f"Smoke Test: [bold yellow]{smoke_test}[/bold yellow]")

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

    console.print("[green]✅ Non-interactive evaluation completed successfully![/green]")


if __name__ == "__main__":
    cui()
