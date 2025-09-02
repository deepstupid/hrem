import click
from sc_engine.core.model_runner import ScientificModelRunner
from cui.progress import CUIProgressHandler

@click.command()
@click.option("--challenge", type=str, default="synthetic_sort", help="Challenge ID to run.")
@click.option("--patience", type=click.Choice(["low", "medium", "high"]), default="medium",
              help="Patience level for the comparison.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--dataset", type=str, default=None, help="Dataset to use.")
@click.option("--models", type=str, multiple=True, help="Models to evaluate.")
@click.option("--arch-overrides", type=str, default=None, help="JSON string of architecture overrides.")
def main(challenge: str, patience: str, smoke_test: bool, dataset: str, models: list[str], arch_overrides: str):
    """Run a scientific algorithm evaluation with a console-based UI."""

    progress_handler = CUIProgressHandler()

    progress_handler.console.print(f"[bold blue]🔬 Starting Scientific Evaluation for Challenge: {challenge}[/bold blue]")

    runner = ScientificModelRunner()
    runner.run(
        run_type="comparison",
        challenge_id=challenge,
        patience_level=patience,
        smoke_test=smoke_test,
        dataset=dataset,
        models=list(models) if models else None,
        arch_overrides=arch_overrides,
        progress_handler=progress_handler
    )

    progress_handler.console.print("[green]✅ Scientific evaluation completed successfully![/green]")

if __name__ == "__main__":
    main()
