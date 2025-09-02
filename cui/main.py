import click
from cui.runner import CUIRunner

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
    runner = CUIRunner(ctx)
    runner.list_challenges()

@cui.command(name="list-models")
@click.pass_context
def list_models(ctx):
    """Lists all available models."""
    runner = CUIRunner(ctx)
    runner.list_models()

@cui.command(name="run-interactive")
@click.pass_context
def run_interactive(ctx):
    """Interactively run a scientific evaluation."""
    runner = CUIRunner(ctx)
    runner.run_interactive()

@cui.command(name="list-datasets")
@click.pass_context
def list_datasets(ctx):
    """Lists all available datasets."""
    runner = CUIRunner(ctx)
    runner.list_datasets()

@cui.command(name="run-once")
@click.option('--challenge-id', required=True, help='The ID of the challenge to run.')
@click.option('--model', 'models', multiple=True, help='The name of a model to run. Can be specified multiple times.')
@click.option('--patience', type=click.Choice(['low', 'medium', 'high']), default='medium', help='The patience level for the run.')
@click.option('--smoke-test', is_flag=True, default=False, help='Run in smoke test mode.')
@click.pass_context
def run_once(ctx, challenge_id, models, patience, smoke_test):
    """Run a scientific evaluation non-interactively."""
    runner = CUIRunner(ctx)
    runner.run_once(challenge_id, list(models), patience, smoke_test)

if __name__ == "__main__":
    cui()
