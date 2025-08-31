from typing import Dict, Any, List, Optional, Callable
from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget
from .engine import ScientificDiscoveryEngine, DiscoveryResults
from .config_manager import ConfigManager
from .challenge_registry import ChallengeRegistry
from rich.console import Console

console = Console()

class ScientificModelRunner:
    """Runner for discovery-oriented algorithm comparison."""

    def __init__(self, config_dir: str = 'config'):
        self.config_manager = ConfigManager(config_dir)
        self.challenge_registry = ChallengeRegistry()
        self.model_configs = self.config_manager.load_model_configs()
        self.search_spaces = self.config_manager.load_search_spaces()

    def run(self, run_type: str, progress_callback: Optional[Callable] = None, **kwargs):
        """
        Run a discovery session based on the specified run type.
        This method now delegates to specific handlers for each run type.
        """
        run_handlers = {
            "comparison": self._run_comparison,
            "optimization": self._run_optimization,
            "demo": self._run_demo,
        }
        handler = run_handlers.get(run_type)
        if not handler:
            raise ValueError(f"Invalid run type: {run_type}")

        return handler(progress_callback=progress_callback, **kwargs)

    def _execute_engine(self, challenge_data: Dict, engine_config: Dict, patience_level: str, arch_overrides: Optional[str], dataset_override: Optional[str], progress_callback: Optional[Callable]):
        """Helper to configure and run the scientific discovery engine."""
        smoke_test = engine_config.get("smoke_test", False)

        challenge = self._create_challenge_config(challenge_data, smoke_test, dataset_override=dataset_override)
        algorithms = self._load_algorithms(challenge_data.get("models", []), arch_overrides=arch_overrides)
        patience_budget = PatienceBudget(level=patience_level)

        engine = ScientificDiscoveryEngine(
            challenge=challenge,
            algorithms=algorithms,
            config=engine_config,
            progress_callback=progress_callback
        )

        results = engine.execute_discovery_session(patience_budget)
        self._display_results(results)
        return results

    def _run_comparison(self, progress_callback: Optional[Callable], **kwargs):
        challenge_id = kwargs.get("challenge_id", "quick_comparison")
        challenge_data = self.challenge_registry.get_challenge_by_id(challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")

        if kwargs.get("models"):
            challenge_data['models'] = kwargs.get("models")

        engine_config = {"smoke_test": kwargs.get("smoke_test", False)}
        patience_level = kwargs.get("patience_level", "medium")

        return self._execute_engine(
            challenge_data=challenge_data,
            engine_config=engine_config,
            patience_level=patience_level,
            arch_overrides=kwargs.get("arch_overrides"),
            dataset_override=kwargs.get("dataset"),
            progress_callback=progress_callback
        )

    def _run_optimization(self, progress_callback: Optional[Callable], **kwargs):
        challenge_id = kwargs.get("challenge_id")
        challenge_data = self.challenge_registry.get_challenge_by_id(challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")

        model_to_optimize = kwargs.get("model_to_optimize")
        if not model_to_optimize:
            raise ValueError("model_to_optimize must be specified for optimization runs")
        challenge_data['models'] = [model_to_optimize]

        engine_config = {
            "smoke_test": kwargs.get("smoke_test", False),
            "n_trials": kwargs.get("n_trials", 10)
        }

        return self._execute_engine(
            challenge_data=challenge_data,
            engine_config=engine_config,
            patience_level="high",
            arch_overrides=kwargs.get("arch_overrides"),
            dataset_override=kwargs.get("dataset"),
            progress_callback=progress_callback
        )

    def _run_demo(self, progress_callback: Optional[Callable], **kwargs):
        challenge_id = kwargs.get("challenge_id", "quick_comparison")
        challenge_data = self.challenge_registry.get_challenge_by_id(challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")

        if kwargs.get("models"):
            challenge_data['models'] = kwargs.get("models")

        engine_config = {"smoke_test": kwargs.get("smoke_test", False)}

        return self._execute_engine(
            challenge_data=challenge_data,
            engine_config=engine_config,
            patience_level="high",
            arch_overrides=kwargs.get("arch_overrides"),
            dataset_override=kwargs.get("dataset"),
            progress_callback=progress_callback
        )


    def _create_challenge_config(self, challenge_data: Dict[str, Any], smoke_test: bool, dataset_override: Optional[str] = None) -> ChallengeConfig:
        dataset_info = challenge_data.get("dataset", {})

        if dataset_override:
            dataset_name = dataset_override
        elif isinstance(dataset_info, str):
            dataset_name = dataset_info
        else:
            dataset_name = dataset_info.get("dataset", "synthetic")

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

    def _load_algorithms(self, algorithm_names: List[str], arch_overrides: Optional[str] = None) -> List[AlgorithmConfig]:
        """Load algorithm configurations, applying overrides if provided."""
        import json

        overrides = {}
        if arch_overrides:
            try:
                overrides = json.loads(arch_overrides)
            except json.JSONDecodeError:
                console.print(f"[bold red]Error: Invalid JSON in arch-overrides: {arch_overrides}[/bold red]")
                overrides = {}

        algorithms = []
        for name in algorithm_names:
            model_data = self.model_configs.get(name, {}).copy()
            if not model_data:
                console.print(f"[yellow]⚠️  Model '{name}' not found in registry[/yellow]")
                continue
            
            # Deep merge overrides
            if overrides:
                # A simple update is not enough for nested dictionaries (like 'arch')
                for key, value in overrides.items():
                    if isinstance(value, dict) and isinstance(model_data.get(key), dict):
                        model_data[key].update(value)
                    else:
                        model_data[key] = value

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