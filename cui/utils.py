from rich.table import Table
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule
from typing import List, Dict

from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry
from sc_engine.core.schemas import ChallengeSchema, ModelConfigSchema

def prompt_for_challenge(console: Console, challenge_registry: ChallengeRegistry) -> ChallengeSchema:
    """Interactively prompts the user to select a challenge."""
    console.print(Rule("[bold cyan]Step 1: Select a Challenge[/bold cyan]"))
    challenges = get_available_challenges(challenge_registry)

    table = Table(title="🔬 Available Scientific Challenges")
    table.add_column("Index", style="magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")

    for i, challenge in enumerate(challenges):
        table.add_row(str(i), challenge.id, challenge.name)
    console.print(table)

    idx = Prompt.ask("Enter the index of the challenge", choices=[str(i) for i in range(len(challenges))], show_choices=False)
    return challenges[int(idx)]

def prompt_for_models(console: Console, config_manager: ConfigManager) -> List[str]:
    """Interactively prompts the user to select models."""
    console.print(Rule("[bold cyan]Step 2: Select Models[/bold cyan]"))
    model_configs = get_available_models(config_manager)
    models = list(model_configs.keys())

    table = Table(title="🤖 Available Models")
    table.add_column("Index", style="magenta")
    table.add_column("Name", style.cyan)

    for i, name in enumerate(models):
        table.add_row(str(i), name)
    console.print(table)

    indices_str = Prompt.ask("Enter model indices (e.g., '0, 2')")
    return [models[int(i.strip())] for i in indices_str.split(',')]
