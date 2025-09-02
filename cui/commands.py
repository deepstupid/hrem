import click
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule

from sc_engine.core.engine import ScientificDiscoveryEngine
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry
from sc_engine.core.ui_utils import get_available_challenges, get_available_models
from dataset_manager import dataset_manager
from cui.progress import CUIProgressHandler
from cui.utils import prompt_for_challenge, prompt_for_models
from sc_engine.core.components import OptunaOptimizer, ScientificTimingManager

@click.group()
@click.pass_context
def cui(ctx):
    """A console-based interface for the Scientific Discovery Engine."""
    ctx.obj = {
        "console": Console(),
        "config_manager": ConfigManager(),
        "challenge_registry": ChallengeRegistry(ConfigManager())
    }

@cui.command(name="list-challenges")
@click.pass_context
def list_challenges(ctx):
    """Lists all available scientific challenges."""
    console = ctx.obj["console"]
    challenges = get_available_challenges(ctx.obj["challenge_registry"])
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
    console = ctx.obj["console"]
    models = get_available_models(ctx.obj["config_manager"])
    if not models:
        console.print("[yellow]No models found.[/yellow]")
        return
    table = Table(title="🤖 Available Models", header_style="bold magenta")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Class Path", style="green")
    for name, model_schema in models.items():
        table.add_row(name, model_schema.algorithm_class)
    console.print(table)

from sc_engine.core.interactive import InteractiveEvaluator

@cui.command(name="run-interactive-puzzle")
@click.option('--dataset-path', required=True, help='The path to the dataset to use.')
@click.option('--puzzle-index', required=True, type=int, help='The index of the puzzle to run.')
@click.option('--challenge-id', required=True, help='The ID of the challenge to run.')
@click.option('--model', 'models', multiple=True, help='The name of a model to run. Can be specified multiple times.')
@click.pass_context
def run_interactive_puzzle(ctx, dataset_path, puzzle_index, challenge_id, models):
    """Run a single puzzle interactively."""
    console = ctx.obj["console"]
    console.print("[bold blue]Welcome to the Interactive Puzzle Runner![/bold blue]")

    evaluator = InteractiveEvaluator()
    evaluator.run(
        dataset_path=dataset_path,
        puzzle_index=puzzle_index,
        challenge_id=challenge_id,
        models=models
    )
    console.print("[green]✅ Interactive puzzle run completed successfully![/green]")

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
    console = ctx.obj["console"]

    console.print(Rule(f"[bold green]🚀 Launching Non-Interactive Evaluation[/bold green]"))
    console.print(f"Challenge: [bold yellow]{challenge_id}[/bold yellow]")
    console.print(f"Models: [bold yellow]{', '.join(models)}[/bold yellow]")
    console.print(f"Patience: [bold yellow]{patience}[/bold yellow]")
    console.print(f"Smoke Test: [bold yellow]{smoke_test}[/bold yellow]")

    config_manager = ConfigManager()
    engine = ScientificDiscoveryEngine(
        config_manager=config_manager,
        progress_handler=CUIProgressHandler(),
        optimizer=OptunaOptimizer(),
        timing_manager=ScientificTimingManager()
    )
    try:
        engine.run(
            challenge_id=challenge_id,
            patience_level=patience,
            smoke_test=smoke_test,
            models=list(models),
        )
        console.print("[green]✅ Non-interactive evaluation completed successfully![/green]")
    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
