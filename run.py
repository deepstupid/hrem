import click
from rich.console import Console
from functools import wraps
import os
import importlib

# Correctly import the ScientificModelRunner
from sc_engine.core.model_runner import ScientificModelRunner
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry

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

class AppContext:
    """A context object to hold shared resources."""
    def __init__(self):
        self.config_manager = ConfigManager('config')
        self.challenge_registry = ChallengeRegistry(self.config_manager)
        self.console = Console()

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """HRM System: A unified interface for scientific comparisons."""
    ctx.obj = AppContext()

try:
    from cui.main import cui as cui_group
    cli.add_command(cui_group, 'cui')
except ImportError as e:
    @cli.command()
    def cui():
        """(CUI not available)"""
        console.print(f"[red]CUI Error: {e}[/red]")


if __name__ == "__main__":
    cli()