import yaml
from rich.console import Console

from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.engine import ScientificDiscoveryEngine
from dataset_manager import dataset_manager

console = Console()

def main():
    """
    Temporary script to test the new interactive puzzle API with model execution.
    """
    # --- Configuration ---
    challenge_name = "synthetic_sort"
    algorithm_names = ["HRM", "HREM"]
    puzzle_index = 5

    # --- Load Configs ---
    try:
        config_manager = ConfigManager()
        challenge_configs = config_manager.load_challenge_configs()
        model_configs = config_manager.load_model_configs()
        search_spaces = config_manager.load_search_spaces()

        challenge_config = next((c for c in challenge_configs.challenges if c.id == challenge_name), None)
        if not challenge_config:
            console.print(f"[red]Challenge '{challenge_name}' not found.[/red]")
            return

        algorithm_configs = []
        for name in algorithm_names:
            model_config = model_configs.get(name)
            if not model_config:
                console.print(f"[red]Model '{name}' not found.[/red]")
                return

            from types import SimpleNamespace
            algorithm_configs.append(SimpleNamespace(
                name=model_config.name,
                config=model_config.model_dump(),
                search_space=search_spaces.get(name)
            ))

        with open("config/demo_config.yaml", 'r') as f:
            demo_config = yaml.safe_load(f)

    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        return

    # --- Get Dataset Path ---
    dataset_path = dataset_manager.get_dataset_path(challenge_config.dataset.dataset, smoke_test=True)
    console.print(f"Using dataset path: {dataset_path}")

    # --- Initialize Engine ---
    engine = ScientificDiscoveryEngine(
        challenge=challenge_config,
        algorithms=algorithm_configs,
        config=demo_config
    )

    # --- Run Interactive Puzzle ---
    console.print(f"\n[bold cyan]--- Testing run_interactive_puzzle with model execution ---[/bold cyan]")
    try:
        results = engine.run_interactive_puzzle(
            dataset_path=dataset_path,
            puzzle_index=puzzle_index,
            algorithm_configs=algorithm_configs
        )
        console.print("[green]Successfully executed run_interactive_puzzle.[/green]")
        console.print("Results:")
        console.print(results)

    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red]")
        console.log(e, log_locals=True)


if __name__ == "__main__":
    main()
