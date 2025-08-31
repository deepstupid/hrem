import click
from rich.console import Console
from functools import wraps

# Correctly import the ScientificModelRunner
from sc_engine.core.model_runner import ScientificModelRunner

console = Console()

def handle_exceptions(func):
    """A decorator to handle common exceptions for CLI commands."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ValueError as e:
            console.print(f"[red]Configuration Error: {e}[/red]")
        except ImportError as e:
            console.print(f"[red]Import Error: {e}[/red]")
            console.print("[yellow]Please ensure all required modules are installed and accessible.[/yellow]")
        except Exception as e:
            console.print(f"[red]An unexpected error occurred: {e}[/red]")
    return wrapper

@click.group()
def cli():
    """HRM System: A unified interface for scientific comparisons."""
    pass

@cli.command()
@click.option("--challenge", type=str, default="synthetic_sort", help="Challenge ID to run.")
@click.option("--patience", type=click.Choice(["low", "medium", "high"]), default="medium", 
              help="Patience level for the comparison.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--dataset", type=str, default=None, help="Dataset to use.")
@click.option("--models", type=str, multiple=True, help="Models to evaluate.")
@click.option("--arch-overrides", type=str, default=None, help="JSON string of architecture overrides.")
@handle_exceptions
def evaluate(challenge: str, patience: str, smoke_test: bool, dataset: str, models: list[str], arch_overrides: str):
    """Run a scientific algorithm evaluation for a specified challenge."""
    console.print(f"[bold blue]🔬 Starting Scientific Evaluation for Challenge: {challenge}[/bold blue]")
    
    runner = ScientificModelRunner()
    runner.run(
        run_type="comparison",
        challenge_id=challenge,
        patience_level=patience,
        smoke_test=smoke_test,
        dataset=dataset,
        models=list(models) if models else None,
        arch_overrides=arch_overrides
    )
    console.print("[green]✅ Scientific evaluation completed successfully![/green]")

@cli.command()
@click.option("--challenge", type=str, default="synthetic_sort", help="Challenge ID to run.")
@click.option("--model-to-optimize", type=str, help="The name of the model to optimize.")
@click.option("--n-trials", type=int, default=10, help="Number of optimization trials.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@handle_exceptions
def optimize(challenge: str, model_to_optimize: str, n_trials: int, smoke_test: bool):
    """Run hyperparameter optimization for a specified model and challenge."""
    console.print(f"[bold blue]🚀 Starting Hyperparameter Optimization for Model: {model_to_optimize} on Challenge: {challenge}[/bold blue]")

    runner = ScientificModelRunner()
    runner.run(
        run_type="optimization",
        challenge_id=challenge,
        model_to_optimize=model_to_optimize,
        n_trials=n_trials,
        smoke_test=smoke_test
    )
    console.print("[green]✅ Hyperparameter optimization completed successfully![/green]")



@cli.command()
def gui():
    """Launch the PyQt6 Graphical User Interface."""
    try:
        from gui.main import main as gui_main
        gui_main()
    except ImportError as e:
        console.print(f"[red]GUI Error: {e}[/red]")
        console.print("[yellow]Make sure PyQt6 is installed: pip install PyQt6[/yellow]")


if __name__ == "__main__":
    cli()