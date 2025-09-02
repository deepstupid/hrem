from typing import List, Any, Dict, Tuple
import torch
import numpy as np
from rich.console import Console

from .config import AlgorithmConfig
from .trainer import Trainer
from .interactive_runner import InteractiveRunner
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from dataset.common import PuzzleDatasetMetadata
from models.hrm.hrem import HREM

console = Console()

class InteractiveModeRunner:
    """
    Runs a single puzzle through models for interactive visualization.
    """
    def __init__(self, default_training_config: Dict[str, Any], challenge_dataset_config: Dict[str, Any]):
        self.default_training_config = default_training_config
        self.challenge_dataset_config = challenge_dataset_config

    def run(self, dataset_path: str, puzzle_index: int, algorithm_configs: List[AlgorithmConfig]) -> Dict[str, Any]:
        """
        Runs a single puzzle through the models for interactive visualization.
        """
        console.print(f"Running interactive puzzle {puzzle_index} from {dataset_path}")

        puzzle_data, dummy_metadata = self._load_interactive_puzzle_data(dataset_path, puzzle_index)

        all_results = {}
        for algo_config in algorithm_configs:
            console.print(f"[bold cyan]Running model: {algo_config.name}[/bold cyan]")

            trainer = self._initialize_trainer_for_puzzle(algo_config, dummy_metadata)
            model = trainer.train_state.model
            is_hrem = isinstance(model.model, HREM)

            runner = InteractiveRunner(model, is_hrem=is_hrem)
            batch = {k: v.to('cpu') for k, v in puzzle_data.items()}
            runner.reset(batch)

            step_results = []
            for i in range(puzzle_data['inputs'].shape[1]):
                single_step_batch = {
                    'inputs': batch['inputs'][:, i:i+1],
                    'labels': batch['labels'][:, i:i+1],
                    'puzzle_identifiers': batch['puzzle_identifiers']
                }
                result, metrics = runner.run_step(single_step_batch)
                result["metrics"] = metrics
                step_results.append(result)

            all_results[algo_config.name] = step_results
            console.print(f"Finished running {algo_config.name}. Collected {len(step_results)} steps.")

        return all_results

    def _load_interactive_puzzle_data(self, dataset_path: str, puzzle_index: int) -> Tuple[Dict[str, torch.Tensor], PuzzleDatasetMetadata]:
        """Loads a single puzzle and creates dummy metadata for it."""
        split = "test"
        set_name = "all"
        dataset_config = PuzzleDatasetConfig(
            seed=42, dataset_path=dataset_path, global_batch_size=1, test_set_mode=True,
            epochs_per_iter=1, rank=0, num_replicas=1,
        )
        dataset = PuzzleDataset(config=dataset_config, split=split)
        dataset._lazy_load_dataset()
        set_data = dataset._data[set_name]
        puzzle_start = set_data["puzzle_indices"][puzzle_index]
        puzzle_end = set_data["puzzle_indices"][puzzle_index + 1]

        puzzle_identifier = set_data["puzzle_identifiers"][puzzle_index]
        puzzle_data = {
            "inputs": torch.from_numpy(set_data["inputs"][puzzle_start:puzzle_end]),
            "labels": torch.from_numpy(set_data["labels"][puzzle_start:puzzle_end]),
            "puzzle_identifiers": torch.from_numpy(np.array([puzzle_identifier]))
        }
        console.print(f"Loaded puzzle with input shape: {puzzle_data['inputs'].shape}")

        dummy_metadata = PuzzleDatasetMetadata(
            pad_id=0, ignore_label_id=-1, blank_identifier_id=0,
            vocab_size=32, seq_len=puzzle_data['inputs'].shape[1],
            num_puzzle_identifiers=1, total_groups=1,
            mean_puzzle_examples=1.0, sets=['all']
        )
        return puzzle_data, dummy_metadata

    def _initialize_trainer_for_puzzle(self, algo_config: AlgorithmConfig, metadata: PuzzleDatasetMetadata) -> Trainer:
        """Initializes a trainer instance for a given algorithm and puzzle metadata."""
        run_config = {"study_name": f"interactive_{algo_config.name}", "output_dir": "experiments"}
        trainer = Trainer(self.default_training_config, algo_config.config, self.challenge_dataset_config, run_config)

        trainer.train_metadata = metadata
        trainer.build_model()
        return trainer
