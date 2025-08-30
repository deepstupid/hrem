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
def compare(challenge: str, patience: str, smoke_test: bool):
    """Run a scientific algorithm comparison for a specified challenge."""
    console.print(f"[bold blue]🔬 Starting Scientific Comparison for Challenge: {challenge}[/bold blue]")
    
    try:
        runner = ScientificModelRunner()
        results = runner.run_comparison_from_config(
            challenge_id=challenge,
            patience_level=patience,
            smoke_test=smoke_test
        )
        console.print("[green]✅ Scientific comparison completed successfully![/green]")
        
    except ValueError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
    except ImportError as e:
        console.print(f"[red]Import Error: {e}[/red]")
        console.print("[yellow]Please ensure all required modules are installed and accessible.[/yellow]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())

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