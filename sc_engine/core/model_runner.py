from typing import Dict, Any, List, Optional
import json
from rich.console import Console

from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget, ChallengeLevel
from .engine import ScientificDiscoveryEngine, DiscoveryResults
from .config_manager import ConfigManager
from .challenge_registry import ChallengeRegistry
from .schemas import ChallengeSchema
from .progress_handler import ProgressHandler

console = Console()


class ScientificModelRunner:
    """Runner for discovery-oriented algorithm comparison."""

    def __init__(self, config_dir: str = 'config'):
        self.config_manager = ConfigManager(config_dir)
        self.challenge_registry = ChallengeRegistry(self.config_manager)
        self.model_configs = self.config_manager.load_model_configs()
        self.search_spaces = self.config_manager.load_search_spaces()

    def run(self, run_type: str, progress_handler: Optional[ProgressHandler] = None, **kwargs):
        """Run a discovery session based on the specified run type."""
        run_handlers = {
            "comparison": self._run_comparison,
            "optimization": self._run_optimization,
            "demo": self._run_demo,
        }
        handler = run_handlers.get(run_type)
        if not handler:
            raise ValueError(f"Invalid run type: {run_type}")
        return handler(progress_handler=progress_handler, **kwargs)

    def _get_challenge_data(self, challenge_id: Optional[str], default_id: Optional[str] = "quick_comparison") -> ChallengeSchema:
        """Fetches and validates challenge data from the registry."""
        final_challenge_id = challenge_id or default_id
        if not final_challenge_id:
            raise ValueError("A challenge ID must be provided for this operation.")

        challenge_data = self.challenge_registry.get_challenge_by_id(final_challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{final_challenge_id}' not found")
        return challenge_data

    def _execute_engine(self, challenge_schema: ChallengeSchema, models_to_run: List[str], engine_config: Dict, patience_level: str, arch_overrides: Optional[str], dataset_override: Optional[str], progress_handler: Optional[ProgressHandler]):
        """Helper to configure and run the scientific discovery engine."""
        smoke_test = engine_config.get("smoke_test", False)

        challenge = self._create_challenge_config(challenge_schema, smoke_test, dataset_override=dataset_override)
        algorithms = self._load_algorithms(models_to_run, arch_overrides=arch_overrides)
        patience_budget = PatienceBudget(level=patience_level)

        engine = ScientificDiscoveryEngine(
            challenge=challenge,
            algorithms=algorithms,
            config=engine_config,
            progress_handler=progress_handler
        )

        results = engine.execute_discovery_session(patience_budget)
        self._display_results(results)
        return results

    def _run_comparison(self, progress_handler: Optional[ProgressHandler], **kwargs):
        challenge_schema = self._get_challenge_data(kwargs.get("challenge_id"), default_id="quick_comparison")
        models_to_run = kwargs.get("models") or challenge_schema.models

        engine_config = {"smoke_test": kwargs.get("smoke_test", False)}
        patience_level = kwargs.get("patience_level", "medium")

        return self._execute_engine(
            challenge_schema=challenge_schema,
            models_to_run=models_to_run,
            engine_config=engine_config,
            patience_level=patience_level,
            arch_overrides=kwargs.get("arch_overrides"),
            dataset_override=kwargs.get("dataset"),
            progress_handler=progress_handler
        )

    def _run_optimization(self, progress_handler: Optional[ProgressHandler], **kwargs):
        challenge_schema = self._get_challenge_data(kwargs.get("challenge_id"), default_id=None)

        model_to_optimize = kwargs.get("model_to_optimize")
        if not model_to_optimize:
            raise ValueError("model_to_optimize must be specified for optimization runs")
        models_to_run = [model_to_optimize]

        engine_config = {
            "smoke_test": kwargs.get("smoke_test", False),
            "n_trials": kwargs.get("n_trials", 10)
        }

        return self._execute_engine(
            challenge_schema=challenge_schema,
            models_to_run=models_to_run,
            engine_config=engine_config,
            patience_level="high",
            arch_overrides=kwargs.get("arch_overrides"),
            dataset_override=kwargs.get("dataset"),
            progress_handler=progress_handler
        )

    def _run_demo(self, progress_handler: Optional[ProgressHandler], **kwargs):
        challenge_schema = self._get_challenge_data(kwargs.get("challenge_id"), default_id="quick_comparison")
        models_to_run = kwargs.get("models") or challenge_schema.models

        engine_config = {"smoke_test": kwargs.get("smoke_test", False)}

        return self._execute_engine(
            challenge_schema=challenge_schema,
            models_to_run=models_to_run,
            engine_config=engine_config,
            patience_level="high",
            arch_overrides=kwargs.get("arch_overrides"),
            dataset_override=kwargs.get("dataset"),
            progress_handler=progress_handler
        )


    def _create_challenge_config(self, challenge_schema: ChallengeSchema, smoke_test: bool, dataset_override: Optional[str] = None) -> ChallengeConfig:
        dataset_name = dataset_override or challenge_schema.dataset.dataset
        data_config = {
            "dataset": dataset_name,
            "smoke_test": smoke_test,
        }

        return ChallengeConfig(
            name=challenge_schema.name,
            id=challenge_schema.id,
            description=challenge_schema.description,
            dataset=data_config,
            scientific_question=challenge_schema.scientific_question or "",
            hypothesis_space=challenge_schema.hypothesis_space or [],
            difficulty=ChallengeLevel(challenge_schema.difficulty.upper())
        )

    def _load_algorithms(self, algorithm_names: List[str], arch_overrides: Optional[str] = None) -> List[AlgorithmConfig]:
        """Load algorithm configurations, applying overrides if provided."""
        overrides = {}
        if arch_overrides:
            try:
                overrides = json.loads(arch_overrides)
            except json.JSONDecodeError:
                console.print(f"[bold red]Error: Invalid JSON in arch-overrides: {arch_overrides}[/bold red]")
                overrides = {}

        algorithms = []
        for name in algorithm_names:
            model_schema = self.model_configs.get(name)
            if not model_schema:
                console.print(f"[yellow]⚠️  Model '{name}' not found in registry[/yellow]")
                continue
            
            model_dict = model_schema.dict()

            if overrides:
                # This is a simple override, for deep merge, a more complex utility would be needed
                model_dict.update(overrides)

            alg_config = AlgorithmConfig(
                name=model_schema.name,
                algorithm_class=model_schema.algorithm_class,
                search_space=self.search_spaces.get(name, {}),
                theoretical_advantages=model_schema.theoretical_advantages,
                theoretical_limitations=model_schema.theoretical_limitations,
                config=model_dict
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

        # Display plot path if it exists
        plot_path = results.metadata.get('plot_path')
        if plot_path:
            console.print(f"\n[bold]Performance Plot:[/bold] 📊")
            console.print(f"  A plot of the training performance has been saved to: [u]{plot_path}[/u]")