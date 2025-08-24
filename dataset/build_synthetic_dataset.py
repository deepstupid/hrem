from typing import Optional
import os
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/synthetic-smoke"
    num_aug: int = 0
    task_type: str = "copy"
    seq_len: int = 10
    vocab_size: int = 10
    num_samples: int = 100


def generate_synthetic_data(config: DataProcessConfig):
    """Generate synthetic data for a given task."""
    inputs = []
    labels = []

    for _ in range(config.num_samples):
        input_seq = np.random.randint(1, config.vocab_size + 1, size=(config.seq_len,))
        if config.task_type == "copy":
            output_seq = input_seq.copy()
        elif config.task_type == "reverse":
            output_seq = input_seq[::-1].copy()
        else:
            raise ValueError(f"Unknown task type: {config.task_type}")

        inputs.append(input_seq)
        labels.append(output_seq)

    return inputs, labels


def convert_subset(set_name: str, config: DataProcessConfig):
    # Generate dataset
    inputs, labels = generate_synthetic_data(config)

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out in zip(tqdm(inputs), labels):
        # Push puzzle (only single example)
        results["inputs"].append(inp)
        results["labels"].append(out)
        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.array(seq)
        return arr

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),

        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.seq_len,
        vocab_size=config.vocab_size + 1,  # PAD + "0" ... "9"

        pad_id=0,
        ignore_label_id=0,

        blank_identifier_id=0,
        num_puzzle_identifiers=1,

        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
