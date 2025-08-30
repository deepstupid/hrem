import yaml
import os
from typing import Dict, Any, List
from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .engine import ScientificDiscoveryEngine, DiscoveryResults
from hrm_system.runner import run_single_model
from hrm_system.optimization import run_optimization
from hrm_system.config import RunConfig, DataConfig, ModelConfig, TrainingConfig, ExperimentConfig, OptimizationConfig
from rich.console import Console

console = Console()

class ScientificModelRunner:
    """Runner for discovery-oriented algorithm comparison."""

    def __init__(self, config_dir: str = 'config'):
        self.config_dir = config_dir
        self.challenge_configs = self._load_yaml(os.path.join(config_dir, 'challenge_config.yaml'))
        self.model_configs = self._load_yaml(os.path.join(config_dir, 'models', 'hrm.yaml'))
        self.model_configs.update(self._load_yaml(os.path.join(config_dir, 'models', 'hrem.yaml')))
        self.search_spaces = self._load_yaml(os.path.join(config_dir, 'search', 'hrm_search_space.yaml'))
        self.search_spaces.update(self._load_yaml(os.path.join(config_dir, 'search', 'hrem_search_space.yaml')))


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
            algorithms=algorithms,
            model_runner=self.run_model,
            optimization_runner=self.run_optimization_wrapper
        )

        results = engine.execute_discovery_session(patience_budget)
        self._display_results(results)
        return results

    def _create_challenge_config(self, challenge_data: Dict[str, Any], smoke_test: bool) -> ChallengeConfig:
        dataset_name = challenge_data.get("dataset", "synthetic")
        data_config = DataConfig(
            dataset=dataset_name,
            smoke_test=smoke_test,
        )
        return ChallengeConfig(
            name=challenge_data["name"],
            id=challenge_data["id"],
            description=challenge_data["description"],
            dataset=data_config,
            scientific_question=challenge_data.get("scientific_question", ""),
            hypothesis_space=challenge_data.get("hypothesis_space", [])
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
                theoretical_limitations=model_data.get('theoretical_limitations', [])
            )
            algorithms.append(alg_config)
        return algorithms

    def run_model(self, model_config: ModelConfig, data_config: DataConfig, run_config: RunConfig, training_config: TrainingConfig) -> Dict[str, Any]:
        """
        Wrapper to call the hrm_system model runner.
        """
        return run_single_model(
            run_config=run_config,
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
        )

    def run_optimization_wrapper(self, model_config: ModelConfig, data_config: DataConfig, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper for the optimization runner.
        """
        run_config = RunConfig(study_name=f"{model_config.name}_optimization")
        training_config = TrainingConfig()

        optimization_config = OptimizationConfig(
            n_trials=10, # This should be configurable
            model_to_optimize=model_config,
            search_space=search_space
        )

        experiment_config = ExperimentConfig(
            mode="optimize",
            run_config=run_config,
            data_config=data_config,
            training_config=training_config,
            optimization_config=optimization_config
        )

        return run_optimization(experiment_config)

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