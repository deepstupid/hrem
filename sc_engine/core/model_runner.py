import yaml
import os
from typing import Dict, Any, List
from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .engine import ScientificDiscoveryEngine, DiscoveryResults
from rich.console import Console

console = Console()

class ScientificModelRunner:
    """Runner for discovery-oriented algorithm comparison."""

    def __init__(self, config_dir: str = 'config'):
        self.config_dir = config_dir
        self.challenge_configs = self._load_yaml(os.path.join(config_dir, 'challenge_config.yaml'))
        self.model_configs = {}
        hrm_config = self._load_yaml(os.path.join(config_dir, 'models', 'hrm.yaml'))
        hrem_config = self._load_yaml(os.path.join(config_dir, 'models', 'hrem.yaml'))
        if hrm_config:
            self.model_configs[hrm_config['name']] = hrm_config
        if hrem_config:
            self.model_configs[hrem_config['name']] = hrem_config

        self.search_spaces = self._load_yaml(os.path.join(config_dir, 'search', 'hrm_search_space.yaml'))
        if self.search_spaces:
            hrem_search_space = self._load_yaml(os.path.join(config_dir, 'search', 'hrem_search_space.yaml'))
            if hrem_search_space:
                self.search_spaces.update(hrem_search_space)


    def _load_yaml(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            console.print(f"[red]Error: Config file not found at {path}[/red]")
            return {}
        except yaml.YAMLError as e:
            console.print(f"[red]Error parsing YAML file at {path}: {e}[/red]")
            return {}

    def run_comparison_from_config(self, challenge_id: str,
                                 patience_level: str = "medium",
                                 smoke_test: bool = False) -> DiscoveryResults:
        """
        Run comparison based on configuration files.
        """
        challenge_data = self.challenge_configs.get(challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")

        challenge = self._create_challenge_config(challenge_data, smoke_test)
        algorithms = self._load_algorithms(challenge_data.get("models", []))
        patience_budget = PatienceBudget(level=patience_level)

        engine = ScientificDiscoveryEngine(
            challenge=challenge,
            algorithms=algorithms
        )

        results = engine.execute_discovery_session(patience_budget)
        self._display_results(results)
        return results

    def run_optimization_from_config(self, challenge_id: str, model_to_optimize: str, n_trials: int, smoke_test: bool):
        """
        Run optimization based on configuration files.
        """
        challenge_data = self.challenge_configs.get(challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")

        # We only want to run the model to be optimized.
        challenge_data['models'] = [model_to_optimize]

        challenge = self._create_challenge_config(challenge_data, smoke_test)
        algorithms = self._load_algorithms(challenge_data.get("models", []))

        engine = ScientificDiscoveryEngine(
            challenge=challenge,
            algorithms=algorithms,
            config={"n_trials": n_trials}
        )

        results = engine.execute_discovery_session(PatienceBudget(level="high"))
        self._display_results(results)
        return results

    def run_demo_from_config(self, challenge_id: str, smoke_test: bool):
        """
        Run a demo based on configuration files.
        """
        # A demo is just a comparison with high patience.
        return self.run_comparison_from_config(challenge_id, patience_level="high", smoke_test=smoke_test)


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