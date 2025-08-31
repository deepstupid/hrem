import click
from rich.console import Console

# Correctly import the ScientificModelRunner
from sc_engine.core.model_runner import ScientificModelRunner

console = Console()

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
def evaluate(challenge: str, patience: str, smoke_test: bool, dataset: str, models: list[str], arch_overrides: str):
    """Run a scientific algorithm evaluation for a specified challenge."""
    console.print(f"[bold blue]🔬 Starting Scientific Evaluation for Challenge: {challenge}[/bold blue]")
    
    try:
        runner = ScientificModelRunner()
        results = runner.run(
            run_type="comparison",
            challenge_id=challenge,
            patience_level=patience,
            smoke_test=smoke_test,
            dataset=dataset,
            models=list(models) if models else None,
            arch_overrides=arch_overrides
        )
        console.print("[green]✅ Scientific evaluation completed successfully![/green]")
        
    except ValueError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
    except ImportError as e:
        console.print(f"[red]Import Error: {e}[/red]")
        console.print("[yellow]Please ensure all required modules are installed and accessible.[/yellow]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")

@cli.command()
@click.option("--challenge", type=str, default="synthetic_sort", help="Challenge ID to run.")
@click.option("--model-to-optimize", type=str, help="The name of the model to optimize.")
@click.option("--n-trials", type=int, default=10, help="Number of optimization trials.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
def optimize(challenge: str, model_to_optimize: str, n_trials: int, smoke_test: bool):
    """Run hyperparameter optimization for a specified model and challenge."""
    console.print(f"[bold blue]🚀 Starting Hyperparameter Optimization for Model: {model_to_optimize} on Challenge: {challenge}[/bold blue]")

    try:
        runner = ScientificModelRunner()
        results = runner.run(
            run_type="optimization",
            challenge_id=challenge,
            model_to_optimize=model_to_optimize,
            n_trials=n_trials,
            smoke_test=smoke_test
        )
        console.print("[green]✅ Hyperparameter optimization completed successfully![/green]")

    except ValueError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
    except ImportError as e:
        console.print(f"[red]Import Error: {e}[/red]")
        console.print("[yellow]Please ensure all required modules are installed and accessible.[/yellow]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")

@cli.command()
@click.option("--challenge", type=str, default="quick_comparison", help="Challenge ID to run for the demo.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--model", type=str, multiple=True, help="Model to run. Can be specified multiple times.")
def demo(challenge: str, smoke_test: bool, model: list[str]):
    """Run a non-interactive, scripted demonstration of a full workflow."""
    console.print(f"[bold blue]🎬 Starting Demo for Challenge: {challenge}[/bold blue]")

    try:
        runner = ScientificModelRunner()
        results = runner.run(
            run_type="demo",
            challenge_id=challenge,
            smoke_test=smoke_test,
            models=list(model) if model else None
        )
        console.print("[green]✅ Demo completed successfully![/green]")

    except ValueError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
    except ImportError as e:
        console.print(f"[red]Import Error: {e}[/red]")
        console.print("[yellow]Please ensure all required modules are installed and accessible.[/yellow]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")

@cli.command()
def tui():
    """Launch the Textual User Interface."""
    try:
        from tui.main import main as tui_main
        tui_main()
    except ImportError as e:
        console.print(f"[red]TUI Error: {e}[/red]")
        console.print("[yellow]Make sure TUI dependencies are installed.[/yellow]")

if __name__ == "__main__":
    cli()