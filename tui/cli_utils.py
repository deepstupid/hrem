import sys
from typing import Dict, Any, List
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from challenges import (
    Challenge,
    ChallengeDifficulty,
    get_all_challenges_sorted,
)

console = Console()

class ChallengeSelector:
    """Handles challenge selection and display, using a config object."""

    def __init__(self, ui_config: Dict[str, Any]):
        self.ui_config = ui_config["challenge_selector"]

    def _display_challenge_list(self, challenges: List[Challenge]):
        """Display the list of challenges grouped by difficulty."""
        difficulty_display = self.ui_config["difficulty_display"]
        hardware_icon = self.ui_config["hardware_icon"]
        duration_icon = self.ui_config["duration_icon"]

        console.clear()
        console.print(Panel(self.ui_config["title"], expand=False))

        # Display challenges grouped by difficulty
        for difficulty in [ChallengeDifficulty.BEGINNER, ChallengeDifficulty.INTERMEDIATE,
                          ChallengeDifficulty.ADVANCED, ChallengeDifficulty.RESEARCH]:
            difficulty_challenges = [c for c in challenges if c.difficulty == difficulty]
            if difficulty_challenges:
                console.print(f"\n[bold]{difficulty_display[difficulty.value.upper()]} Challenges:[/bold]")
                for i, challenge in enumerate(difficulty_challenges, 1):
                    global_index = challenges.index(challenge) + 1
                    console.print(f"  {global_index:2d}. [cyan]{challenge.name}[/cyan]")
                    console.print(f"      [dim]{challenge.description}[/dim]")
                    console.print(f"      {hardware_icon}  {challenge.recommended_hardware} | {duration_icon}  {challenge.expected_duration}")

    def _get_user_selection(self, challenges: List[Challenge]) -> Challenge:
        """Get user selection from the challenge list."""
        while True:
            try:
                choice = Prompt.ask(self.ui_config["prompt"])

                if choice.isdigit():
                    index = int(choice) - 1
                    if 0 <= index < len(challenges):
                        return challenges[index]

                for challenge in challenges:
                    if challenge.name.lower() == choice.lower():
                        return challenge

                console.print(self.ui_config["invalid_selection"])
            except KeyboardInterrupt:
                console.print(self.ui_config["exit_message"])
                sys.exit(0)

    def display_challenge_menu(self) -> Challenge:
        """Display a menu for selecting a challenge and return the selected challenge."""
        challenges = get_all_challenges_sorted()
        self._display_challenge_list(challenges)
        return self._get_user_selection(challenges)
