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

from hrm_system.config import ModelConfig, TrainingConfig
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import get_model_source_path, load_model_class


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


def create_dataloader(training_config: TrainingConfig, data_path: str, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=training_config.seed,
        dataset_path=data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=training_config.num_workers,
        prefetch_factor=training_config.prefetch_factor,
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


def compute_lr(base_lr: float, training_config: TrainingConfig, train_state: TrainState):
    if training_config.lr_schedule == "linear":
        return linear_schedule_with_warmup_lr_lambda(
            current_step=train_state.step,
            base_lr=base_lr,
            num_warmup_steps=round(training_config.lr_warmup_steps),
            num_training_steps=train_state.total_steps,
            min_ratio=training_config.lr_min_ratio
        )
    else:  # default to cosine
        return cosine_schedule_with_warmup_lr_lambda(
            current_step=train_state.step,
            base_lr=base_lr,
            num_warmup_steps=round(training_config.lr_warmup_steps),
            num_training_steps=train_state.total_steps,
            min_ratio=training_config.lr_min_ratio
        )


def train_batch(training_config: TrainingConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    torch._functorch.config.donated_buffer = False
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {k: v.to(device) for k, v in batch.items()}

    if train_state.carry is None:
        with torch.device(device):
            train_state.carry = train_state.model.initial_carry(batch)

    # Enable gradient scaling for mixed precision training
    use_amp = training_config.use_amp and device.type == "cuda"
    if use_amp and train_state.scaler is None:
        train_state.scaler = torch.cuda.amp.GradScaler()

    scaler = train_state.scaler

    # Forward pass with optional AMP
    if use_amp:
        with torch.cuda.amp.autocast():
            new_carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])
    else:
        new_carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    train_state.carry = new_carry

    # Backward pass with optional AMP
    if use_amp and scaler is not None:
        scaled_loss = scaler.scale((1 / global_batch_size) * loss)
        scaled_loss.backward()
    else:
        scaled_loss = (1 / global_batch_size) * loss
        scaled_loss.backward()

    # Clean up to prevent memory issues
    del scaled_loss
    del loss

    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, training_config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step

        # Optimizer step with optional AMP
        if use_amp and scaler is not None:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad()

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            count = max(reduced_metrics["count"], 1)
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics
    return None


def evaluate(training_config: TrainingConfig, checkpoint_path: Optional[str], train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        all_preds = {}

        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]

        carry = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.device(device):
                carry = train_state.model.initial_carry(batch)

            while True:
                carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=training_config.eval_save_outputs)

                if all_finish:
                    break

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in training_config.eval_save_outputs:
                        all_preds.setdefault(k, [])
                        all_preds[k].append(v.cpu())

            del carry, preds, batch, all_finish

            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device=device)

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        if len(all_preds) and checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                                   for set_id, set_name in enumerate(set_ids)}

                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                return reduced_metrics
    return None


def save_train_state(checkpoint_path: Optional[str], train_state: TrainState):
    if checkpoint_path is None:
        return

    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(checkpoint_path, f"step_{train_state.step}.pth"))


def save_code_and_config(checkpoint_path: Optional[str], model_config: ModelConfig, training_config: TrainingConfig, logger: LocalLogger):
    if checkpoint_path is None:
        return

    os.makedirs(checkpoint_path, exist_ok=True)

    # NOTE: This part is simplified. In a real scenario, you might want to version control your code
    # and store the commit hash instead of copying the files.
    # For now, we just save the configs.
    config_to_save = {
        "model_config": model_config.model_dump(),
        "training_config": training_config.model_dump()
    }

    config_file = os.path.join(checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config_to_save, f)
