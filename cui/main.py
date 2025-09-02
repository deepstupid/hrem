import click
from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule

from sc_engine.core.model_runner import ScientificModelRunner
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry
from dataset_manager import dataset_manager
from cui.progress import CUIProgressHandler

@click.group()
def cui():
    """A console-based interface for the Scientific Discovery Engine."""
    pass

@cui.command(name="list-challenges")
def list_challenges():
    """Lists all available scientific challenges."""
    console = Console()
    config_manager = ConfigManager('config')
    challenge_registry = ChallengeRegistry(config_manager)
    challenges = challenge_registry.get_all_challenges()

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
def list_models():
    """Lists all available models."""
    console = Console()
    config_manager = ConfigManager('config')
    models = config_manager.load_model_configs()

    if not models:
        console.print("[yellow]No models found.[/yellow]")
        return

    table = Table(title="🤖 Available Models", header_style="bold magenta")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Class Path", style="green")

    for name, model_schema in models.items():
        table.add_row(name, model_schema.algorithm_class)

    console.print(table)


@cui.command()
def run():
    """Interactively run a scientific evaluation."""
    console = Console()
    console.print("[bold blue]Welcome to the Interactive Experiment Runner![/bold blue]")
    console.print("Let's set up a new scientific evaluation.")

    # Get Challenge
    console.print(Rule("[bold cyan]Step 1: Select a Challenge[/bold cyan]"))
    config_manager = ConfigManager('config')
    challenge_registry = ChallengeRegistry(config_manager)
    challenges = challenge_registry.get_all_challenges()

    challenge_table = Table(title="🔬 Available Scientific Challenges")
    challenge_table.add_column("Index", style="magenta")
    challenge_table.add_column("ID", style="cyan")
    challenge_table.add_column("Name", style="green")
    challenge_table.add_column("Description", style="yellow")

    for i, challenge in enumerate(challenges):
        challenge_table.add_row(str(i), challenge.id, challenge.name, challenge.description)
    console.print(challenge_table)

    challenge_idx = Prompt.ask("Enter the index of the challenge you want to run", choices=[str(i) for i in range(len(challenges))], show_choices=False)
    selected_challenge = challenges[int(challenge_idx)]

    # Get Models
    console.print(Rule("[bold cyan]Step 2: Select Models[/bold cyan]"))
    model_configs = config_manager.load_model_configs()
    models = list(model_configs.keys())

    model_table = Table(title="🤖 Available Models")
    model_table.add_column("Index", style="magenta")
    model_table.add_column("Name", style="cyan")

    for i, model_name in enumerate(models):
        model_table.add_row(str(i), model_name)
    console.print(model_table)

    model_indices_str = Prompt.ask("Enter the indices of the models you want to run (e.g., '0, 2')")
    selected_model_indices = [int(i.strip()) for i in model_indices_str.split(',')]
    selected_models = [models[i] for i in selected_model_indices]

    # Get Patience
    console.print(Rule("[bold cyan]Step 3: Set Patience Level[/bold cyan]"))
    patience = Prompt.ask("Choose a patience level", choices=["low", "medium", "high"], default="medium")

    # Run Evaluation
    console.print(Rule(f"[bold green]🚀 Launching Evaluation[/bold green]"))
    console.print(f"Challenge: [bold yellow]{selected_challenge.name}[/bold yellow]")
    console.print(f"Models: [bold yellow]{', '.join(selected_models)}[/bold yellow]")
    console.print(f"Patience: [bold yellow]{patience}[/bold yellow]")

    progress_handler = CUIProgressHandler()
    runner = ScientificModelRunner()
    runner.run(
        run_type="comparison",
        challenge_id=selected_challenge.id,
        patience_level=patience,
        smoke_test=False,  # Or make this an option
        models=selected_models,
        progress_handler=progress_handler
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


if __name__ == "__main__":
    cui()
