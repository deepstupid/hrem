from typing import Dict, Any, List, Optional
import json
from rich.console import Console

from .config import ChallengeConfig, AlgorithmConfig, PatienceBudget, ChallengeLevel
from .engine import ScientificDiscoveryEngine
from .orchestrator import EngineOrchestrator, DiscoveryResults
from .config_manager import ConfigManager
from .challenge_registry import ChallengeRegistry
from .interactive_mode_runner import InteractiveModeRunner
from .schemas import ChallengeSchema
from .progress_handler import ProgressHandler
from dataset_manager import dataset_manager
import threading

console = Console()


import yaml

class ScientificModelRunner:
    """Runner for discovery-oriented algorithm comparison."""

    def __init__(self, config_dir: str = 'config'):
        self.config_manager = ConfigManager(config_dir)
        self.challenge_registry = ChallengeRegistry(self.config_manager)
        self.model_configs = self.config_manager.load_model_configs()
        self.search_spaces = self.config_manager.load_search_spaces()
        with open("config/training/default.yaml", 'r') as f:
            self.default_training_config = yaml.safe_load(f)

    def run(self, run_type: str, progress_handler: Optional[ProgressHandler] = None, cancel_event: Optional[threading.Event] = None, **kwargs):
        """
        Run a discovery session based on the specified run type.
        This is the main entry point for kicking off an experiment.
        """
        if run_type in ["comparison", "optimization"]:
            # Essential parameters that must be provided by the caller (GUI/CLI)
            if not kwargs.get("challenge_id"):
                raise ValueError("A 'challenge_id' must be provided.")
            if not kwargs.get("patience_level"):
                raise ValueError("A 'patience_level' must be provided.")
            return self._run_discovery_session(run_type, progress_handler=progress_handler, cancel_event=cancel_event, **kwargs)

        elif run_type == "interactive":
            if not kwargs.get("dataset_path"):
                raise ValueError("A 'dataset_path' must be provided for interactive mode.")
            if kwargs.get("puzzle_index") is None:
                raise ValueError("A 'puzzle_index' must be provided for interactive mode.")
            return self._run_interactive_session(**kwargs)

        else:
            raise ValueError(f"Invalid run type: {run_type}")

    def _run_interactive_session(self, **kwargs):
        """Helper to configure and run an interactive puzzle session."""
        challenge_schema = self._get_challenge_data(kwargs.get("challenge_id"))
        challenge = self._create_challenge_config(challenge_schema, smoke_test=False)
        algorithms = self._load_algorithms(kwargs.get("models", []))

        runner = InteractiveModeRunner(self.default_training_config, challenge.dataset)
        return runner.run(
            dataset_path=kwargs["dataset_path"],
            puzzle_index=kwargs["puzzle_index"],
            algorithm_configs=algorithms
        )

    def _get_challenge_data(self, challenge_id: str) -> ChallengeSchema:
        """Fetches and validates challenge data from the registry."""
        challenge_data = self.challenge_registry.get_challenge_by_id(challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")
        return challenge_data

    def _run_discovery_session(self, run_type: str, progress_handler: Optional[ProgressHandler], cancel_event: Optional[threading.Event], **kwargs):
        """Helper to configure and run the scientific discovery engine."""
        challenge_schema = self._get_challenge_data(kwargs.get("challenge_id"))

        # Determine models to run based on the run type
        if run_type == "optimization":
            model_to_optimize = kwargs.get("model_to_optimize")
            if not model_to_optimize:
                raise ValueError("'model_to_optimize' must be specified for optimization runs.")
            models_to_run = [model_to_optimize]
        else:  # comparison
            models_to_run = kwargs.get("models")
            if not models_to_run:
                raise ValueError("'models' must be specified for comparison runs.")

        # Configure engine
        engine_config = {"smoke_test": kwargs.get("smoke_test", False)}
        if run_type == "optimization":
            engine_config["n_trials"] = kwargs.get("n_trials", 10)

        # Common setup
        smoke_test = engine_config.get("smoke_test", False)
        dataset_override = kwargs.get("dataset")
        arch_overrides = kwargs.get("arch_overrides")
        patience_level = kwargs.get("patience_level") # Already validated in run()

        challenge = self._create_challenge_config(challenge_schema, smoke_test, dataset_override=dataset_override)
        algorithms = self._load_algorithms(models_to_run, arch_overrides=arch_overrides)
        patience_budget = PatienceBudget(level=patience_level)

        engine = ScientificDiscoveryEngine(
            challenge=challenge,
            algorithms=algorithms,
            default_training_config=self.default_training_config,
            config=engine_config,
            progress_handler=progress_handler,
            cancel_event=cancel_event
        )

        orchestrator = EngineOrchestrator(engine)
        results = orchestrator.run(patience_budget)
        self._display_results(results)
        return results


    def _create_challenge_config(self, challenge_schema: ChallengeSchema, smoke_test: bool, dataset_override: Optional[str] = None) -> ChallengeConfig:
        """Creates the final ChallengeConfig object from the schema."""
        dataset_name = dataset_override or challenge_schema.dataset.dataset

        # Use the dataset manager to get the path, which will generate the data if it doesn't exist.
        dataset_path = dataset_manager.get_dataset_path(dataset_name, smoke_test=smoke_test)

        data_config = {
            "dataset": dataset_path,
            "smoke_test": smoke_test,
        }

        # The 'difficulty' field is intentionally omitted here.
        # It will use the default value specified in the ChallengeConfig dataclass.
        return ChallengeConfig(
            name=challenge_schema.name,
            id=challenge_schema.id,
            description=challenge_schema.description,
            dataset=data_config,
            scientific_question=challenge_schema.scientific_question or "",
            hypothesis_space=challenge_schema.hypothesis_space or []
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