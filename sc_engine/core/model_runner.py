from typing import Dict, Any, List
from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .engine import ScientificDiscoveryEngine, DiscoveryResults
from .config_manager import ConfigManager
from rich.console import Console

console = Console()

class ScientificModelRunner:
    """Runner for discovery-oriented algorithm comparison."""

    def __init__(self, config_dir: str = 'config'):
        self.config_manager = ConfigManager(config_dir)
        self.challenge_configs = self.config_manager.load_challenge_configs()
        self.model_configs = self.config_manager.load_model_configs()
        self.search_spaces = self.config_manager.load_search_spaces()

    def run(self, run_type: str, **kwargs):
        """
        Run a discovery session based on the specified run type.

        Args:
            run_type: The type of run to execute. Can be 'comparison', 'optimization', or 'demo'.
            **kwargs: Additional arguments for the run, such as 'challenge_id', 'smoke_test', etc.
        """
        challenge_id = kwargs.get("challenge_id", "synthetic_sort")
        smoke_test = kwargs.get("smoke_test", False)

        challenge_data = self.challenge_configs.get(challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")

        engine_config = {}
        patience_level = "medium"

        if run_type == "optimization":
            model_to_optimize = kwargs.get("model_to_optimize")
            if not model_to_optimize:
                raise ValueError("model_to_optimize must be specified for optimization runs")
            challenge_data['models'] = [model_to_optimize]
            engine_config["n_trials"] = kwargs.get("n_trials", 10)
            patience_level = "high"

        elif run_type == "demo":
            patience_level = "high"

        elif run_type != "comparison":
            raise ValueError(f"Invalid run type: {run_type}")

        challenge = self._create_challenge_config(challenge_data, smoke_test)
        algorithms = self._load_algorithms(challenge_data.get("models", []))
        patience_budget = PatienceBudget(level=patience_level)

        engine = ScientificDiscoveryEngine(
            challenge=challenge,
            algorithms=algorithms,
            config=engine_config
        )

        results = engine.execute_discovery_session(patience_budget)
        self._display_results(results)
        return results


    def _create_challenge_config(self, challenge_data: Dict[str, Any], smoke_test: bool) -> ChallengeConfig:
        dataset_name = challenge_data.get("dataset", "synthetic")
        data_config = {
            "dataset": dataset_name,
            "smoke_test": smoke_test,
        }
        from .config import ChallengeLevel
        return ChallengeConfig(
            name=challenge_data["name"],
            id=challenge_data["id"],
            description=challenge_data["description"],
            dataset=data_config,
            scientific_question=challenge_data.get("scientific_question", ""),
            hypothesis_space=challenge_data.get("hypothesis_space", []),
            difficulty=ChallengeLevel.INTERMEDIATE
        )

    def _load_algorithms(self, algorithm_names: List[str]) -> List[AlgorithmConfig]:
        """Load algorithm configurations."""
        algorithms = []
        for name in algorithm_names:
            model_data = self.model_configs.get(name)
            if not model_data:
                console.print(f"[yellow]⚠️  Model '{name}' not found in registry[/yellow]")
                continue
            
            alg_config = AlgorithmConfig(
                name=name,
                algorithm_class=model_data['algorithm_class'],
                search_space=self.search_spaces.get(name, {}),
                theoretical_advantages=model_data.get('theoretical_advantages', []),
                theoretical_limitations=model_data.get('theoretical_limitations', []),
                config=model_data
            )
            algorithms.append(alg_config)
        return algorithms

    def _display_results(self, results: DiscoveryResults):
        """Display the results of the discovery session."""
        console.print("\n[bold green]=== SCIENTIFIC DISCOVERY RESULTS ===[/bold green]")
        console.print(f"\n[bold]Challenge:[/bold] {results.challenge.name}")
        
        if results.insights:
            console.print(f"\n[bold]Scientific Insights:[/bold]")
            for i, insight in enumerate(results.insights, 1):
                console.print(f"  {i}. [{insight.type.upper()}] (Confidence: {insight.confidence:.2f})")
                for implication in insight.implications:
                    console.print(f"      - {implication}")
        else:
            console.print(f"\n[yellow]No significant scientific insights generated[/yellow]")
        
        console.print(f"\n[bold]Session Summary:[/bold]")
        console.print(f"  Total Time: {results.metadata.get('total_elapsed_time', 0):.2f}s")