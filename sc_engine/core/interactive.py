import torch
import numpy as np
from rich.console import Console
from typing import Dict, Any, List, Tuple, Optional
import json
import yaml

from .config import ChallengeConfig, AlgorithmConfig
from .schemas import ChallengeSchema
from .trainer import Trainer
from dataset_manager import dataset_manager
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from models.hrm.hrem import HREM
from .interactive_runner import InteractiveRunner
from .challenge_registry import ChallengeRegistry
from .config_manager import ConfigManager


console = Console()

class InteractiveEvaluator:
    """Runs a single puzzle through models for interactive visualization."""

    def __init__(self, config_dir: str = 'config'):
        self.config_manager = ConfigManager(config_dir)
        self.challenge_registry = ChallengeRegistry(self.config_manager)
        self.model_configs = self.config_manager.load_model_configs()
        self.search_spaces = self.config_manager.load_search_spaces()
        with open("config/training/default.yaml", 'r') as f:
            self.default_training_config = yaml.safe_load(f)

    def run(self, **kwargs) -> Dict[str, Any]:
        """Main entry point for the interactive evaluator."""
        if not all(k in kwargs for k in ["dataset_path", "puzzle_index", "challenge_id", "models"]):
             raise ValueError("Missing required arguments for interactive mode.")

        dataset_path, puzzle_index = kwargs["dataset_path"], kwargs["puzzle_index"]
        challenge_schema = self._get_challenge_data(kwargs["challenge_id"])
        challenge_config = self._create_challenge_config(challenge_schema, smoke_test=False)
        algorithm_configs = self._load_algorithms(kwargs["models"])

        console.print(f"Running interactive puzzle {puzzle_index} from {dataset_path}")
        puzzle_data, dummy_metadata = self._load_interactive_puzzle_data(dataset_path, puzzle_index)

        all_results = {}
        for algo_config in algorithm_configs:
            console.print(f"[bold cyan]Running model: {algo_config.name}[/bold cyan]")
            trainer = self._initialize_trainer_for_puzzle(algo_config, dummy_metadata, challenge_config.dataset)
            model = trainer.train_state.model
            is_hrem = isinstance(model.model, HREM)
            runner = InteractiveRunner(model, is_hrem=is_hrem)
            batch = {k: v.to('cpu') for k, v in puzzle_data.items()}
            runner.reset(batch)
            step_results = []
            for i in range(puzzle_data['inputs'].shape[1]):
                single_step_batch = {'inputs': batch['inputs'][:, i:i+1], 'labels': batch['labels'][:, i:i+1], 'puzzle_identifiers': batch['puzzle_identifiers']}
                result, metrics = runner.run_step(single_step_batch)
                result["metrics"] = metrics
                step_results.append(result)
            all_results[algo_config.name] = step_results
            console.print(f"Finished running {algo_config.name}. Collected {len(step_results)} steps.")
        return all_results

    def _load_interactive_puzzle_data(self, dataset_path: str, puzzle_index: int) -> Tuple[Dict[str, torch.Tensor], PuzzleDatasetMetadata]:
        """Loads a single puzzle and creates dummy metadata for it."""
        dataset_config = PuzzleDatasetConfig(seed=42, dataset_path=dataset_path, global_batch_size=1, test_set_mode=True, epochs_per_iter=1, rank=0, num_replicas=1)
        dataset = PuzzleDataset(config=dataset_config, split="test")
        dataset._lazy_load_dataset()
        set_data = dataset._data["all"]
        puzzle_start = set_data["puzzle_indices"][puzzle_index]
        puzzle_end = set_data["puzzle_indices"][puzzle_index + 1]
        puzzle_data = {
            "inputs": torch.from_numpy(set_data["inputs"][puzzle_start:puzzle_end]),
            "labels": torch.from_numpy(set_data["labels"][puzzle_start:puzzle_end]),
            "puzzle_identifiers": torch.from_numpy(np.array([set_data["puzzle_identifiers"][puzzle_index]]))
        }
        dummy_metadata = PuzzleDatasetMetadata(pad_id=0, ignore_label_id=-1, blank_identifier_id=0, vocab_size=32, seq_len=puzzle_data['inputs'].shape[1], num_puzzle_identifiers=1, total_groups=1, mean_puzzle_examples=1.0, sets=['all'])
        return puzzle_data, dummy_metadata

    def _initialize_trainer_for_puzzle(self, algo_config: AlgorithmConfig, metadata: PuzzleDatasetMetadata, challenge_dataset_config: Dict[str, Any]) -> Trainer:
        """Initializes a trainer instance for a given algorithm and puzzle metadata."""
        run_config = {"study_name": f"interactive_{algo_config.name}", "output_dir": "experiments"}
        trainer = Trainer(self.default_training_config, algo_config.config, challenge_dataset_config, run_config)
        trainer.train_metadata = metadata
        trainer.build_model()
        return trainer

    def _get_challenge_data(self, challenge_id: str) -> ChallengeSchema:
        """Fetches and validates challenge data from the registry."""
        challenge_data = self.challenge_registry.get_challenge_by_id(challenge_id)
        if not challenge_data:
            raise ValueError(f"Challenge with ID '{challenge_id}' not found")
        return challenge_data

    def _create_challenge_config(self, challenge_schema: ChallengeSchema, smoke_test: bool, dataset_override: Optional[str] = None) -> ChallengeConfig:
        """Creates the final ChallengeConfig object from the schema."""
        dataset_name = dataset_override or challenge_schema.dataset.dataset
        dataset_path = dataset_manager.get_dataset_path(dataset_name, smoke_test=smoke_test)
        data_config = {"dataset": dataset_path, "smoke_test": smoke_test}
        return ChallengeConfig(
            name=challenge_schema.name, id=challenge_schema.id, description=challenge_schema.description,
            dataset=data_config, scientific_question=challenge_schema.scientific_question or "",
            hypothesis_space=challenge_schema.hypothesis_space or []
        )

    def _load_algorithms(self, algorithm_names: List[str], arch_overrides: Optional[str] = None) -> List[AlgorithmConfig]:
        """Load algorithm configurations, applying overrides if provided."""
        overrides = json.loads(arch_overrides) if arch_overrides and isinstance(arch_overrides, str) else (arch_overrides or {})
        algorithms = []
        for name in algorithm_names:
            model_schema = self.model_configs.get(name)
            if not model_schema:
                console.print(f"[yellow]⚠️  Model '{name}' not found in registry[/yellow]")
                continue
            model_dict = model_schema.model_dump()
            model_dict.update(overrides)
            algorithms.append(AlgorithmConfig(
                name=model_schema.name, algorithm_class=model_schema.algorithm_class,
                search_space=self.search_spaces.get(name, {}),
                theoretical_advantages=model_schema.theoretical_advantages,
                theoretical_limitations=model_schema.theoretical_limitations,
                config=model_dict
            ))
        return algorithms
