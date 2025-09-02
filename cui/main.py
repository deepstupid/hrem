import click
from sc_engine.core.model_runner import ScientificModelRunner
from cui.progress import CUIProgressHandler

@click.group()
def cui():
    """A console-based interface for the Scientific Discovery Engine."""
    pass

@cui.command()
@click.option("--challenge", type=str, default="synthetic_sort", help="Challenge ID to run.")
@click.option("--patience", type=click.Choice(["low", "medium", "high"]), default="medium",
              help="Patience level for the comparison.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--dataset", type=str, default=None, help="Dataset to use.")
@click.option("--models", type=str, multiple=True, help="Models to evaluate.")
@click.option("--arch-overrides", type=str, default=None, help="JSON string of architecture overrides.")
def evaluate(challenge: str, patience: str, smoke_test: bool, dataset: str, models: list[str], arch_overrides: str):
    """Run a scientific algorithm evaluation for a specified challenge."""
    progress_handler = CUIProgressHandler()
    progress_handler.console.print(f"[bold blue]🔬 Starting Scientific Evaluation for Challenge: {challenge}[/bold blue]")

    runner = ScientificModelRunner()
    runner.run(
        run_type="comparison",
        challenge_id=challenge,
        patience_level=patience,
        smoke_test=smoke_test,
        dataset=dataset,
        models=list(models) if models else ["HRM", "HREM"],
        arch_overrides=arch_overrides,
        progress_handler=progress_handler
    )

    progress_handler.console.print("[green]✅ Scientific evaluation completed successfully![/green]")

@cui.command()
@click.option("--challenge", type=str, default="synthetic_sort", help="Challenge ID to run.")
@click.option("--patience", type=click.Choice(["low", "medium", "high"]), default="medium",
              help="Patience level for the optimization.")
@click.option("--model-to-optimize", type=str, required=True, help="The name of the model to optimize.")
@click.option("--n-trials", type=int, default=50, help="Number of optimization trials.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
def optimize(challenge: str, patience: str, model_to_optimize: str, n_trials: int, smoke_test: bool):
    """Run hyperparameter optimization for a specified model and challenge."""
    progress_handler = CUIProgressHandler()
    progress_handler.console.print(f"[bold blue]🚀 Starting Hyperparameter Optimization for Model: {model_to_optimize} on Challenge: {challenge}[/bold blue]")

    runner = ScientificModelRunner()
    runner.run(
        run_type="optimization",
        challenge_id=challenge,
        patience_level=patience,
        model_to_optimize=model_to_optimize,
        n_trials=n_trials,
        smoke_test=smoke_test,
        progress_handler=progress_handler
    )
    progress_handler.console.print("[green]✅ Hyperparameter optimization completed successfully![/green]")

if __name__ == "__main__":
    cui()
