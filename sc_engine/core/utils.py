import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Any, Optional, Sequence, List

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from typing import Dict, Any
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from sc_engine.utils.functions import get_model_source_path, load_model_class


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class LocalLogger:
    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path
        self.log_data = []

    def log(self, data: dict, step: int):
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self.log_data.append({"step": step, **data})
            with open(self.log_path, "w") as f:
                json.dump(self.log_data, f, indent=4, cls=NumpyEncoder)
                f.flush()

    def finish(self):
        pass


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int
    scaler: Optional[torch.cuda.amp.GradScaler] = None


def create_dataloader(training_config: Dict[str, Any], data_path: str, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=training_config['seed'],
        dataset_path=data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=training_config['num_workers'],
        prefetch_factor=training_config['prefetch_factor'],
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))

def linear_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0
):
    """Linear decay schedule which can be more stable than cosine for some tasks."""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (1.0 - progress * (1.0 - min_ratio))


def compute_lr(base_lr: float, training_config: Dict[str, Any], train_state: TrainState):
    if training_config['lr_schedule'] == "linear":
        return linear_schedule_with_warmup_lr_lambda(
            current_step=train_state.step,
            base_lr=base_lr,
            num_warmup_steps=round(training_config['lr_warmup_steps']),
            num_training_steps=train_state.total_steps,
            min_ratio=training_config['lr_min_ratio']
        )
    else:  # default to cosine
        return cosine_schedule_with_warmup_lr_lambda(
            current_step=train_state.step,
            base_lr=base_lr,
            num_warmup_steps=round(training_config['lr_warmup_steps']),
            num_training_steps=train_state.total_steps,
            min_ratio=training_config['lr_min_ratio']
        )


def save_train_state(checkpoint_path: Optional[str], train_state: TrainState):
    if checkpoint_path is None:
        return

    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(checkpoint_path, f"step_{train_state.step}.pth"))


def save_code_and_config(checkpoint_path: Optional[str], model_config: Dict[str, Any], training_config: Dict[str, Any], logger: LocalLogger):
    if checkpoint_path is None:
        return

    os.makedirs(checkpoint_path, exist_ok=True)

    # NOTE: This part is simplified. In a real scenario, you might want to version control your code
    # and store the commit hash instead of copying the files.
    # For now, we just save the configs.
    config_to_save = {
        "model_config": model_config,
        "training_config": training_config
    }

    config_file = os.path.join(checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config_to_save, f)
