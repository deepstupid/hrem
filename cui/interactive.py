from rich.table import Table
from rich.prompt import Prompt
from rich.rule import Rule

from sc_engine.core.ui_utils import get_available_challenges, get_available_models
from sc_engine.core.model_runner import ScientificModelRunner
from cui.progress import CUIProgressHandler

def run_interactive_session(ctx):
    """Interactively run a scientific evaluation."""
    console = ctx.obj.console
    config_manager = ctx.obj.config_manager
    challenge_registry = ctx.obj.challenge_registry

    console.print("[bold blue]Welcome to the Interactive Experiment Runner![/bold blue]")
    console.print("Let's set up a new scientific evaluation.")

    # Get Challenge
    console.print(Rule("[bold cyan]Step 1: Select a Challenge[/bold cyan]"))
    challenges = get_available_challenges(challenge_registry)

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
    model_configs = get_available_models(config_manager)
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
        smoke_test=False,
        models=selected_models,
        progress_handler=progress_handler
    )

    console.print("[green]✅ Interactive evaluation completed successfully![/green]")
