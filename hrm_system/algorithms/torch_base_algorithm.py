import os
import yaml
import torch
from omegaconf import OmegaConf
import torch.distributed as dist
from torch import nn
from torch.optim import Adam
import tqdm
from typing import Dict, Any, Callable, Optional

from hrm_system.algorithms.base import Algorithm
from hrm_system.config import ModelConfig, TrainingConfig
from hrm_system.algorithms.utils import (
    LocalLogger,
    TrainState,
    create_dataloader,
    train_batch,
    evaluate,
    save_train_state,
    save_code_and_config,
)
from utils.functions import load_model_class
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from puzzle_dataset import PuzzleDatasetMetadata

class TorchBaseAlgorithm(Algorithm):
    """
    A base class for PyTorch-based training algorithms.
    """

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.train_state: Optional[TrainState] = None

    def _init_train_state(self, train_metadata: PuzzleDatasetMetadata, world_size: int, rank: int):
        # Load base architecture config using OmegaConf to handle interpolations
        arch_config_path = f"config/arch/{self.model_config.base_arch_config}.yaml"
        with open(arch_config_path, 'r') as f:
            # Use OmegaConf to load, which supports interpolation
            arch_config = OmegaConf.load(f)

        # Create a new OmegaConf object for the model configuration
        model_cfg = arch_config.copy()

        # Merge HREM params or other overrides
        if self.model_config.hrem_params:
            hrem_params = OmegaConf.create(self.model_config.hrem_params.model_dump())
            model_cfg = OmegaConf.merge(model_cfg, hrem_params)

        if self.model_config.arch_overrides:
            overrides = OmegaConf.create(self.model_config.arch_overrides)
            model_cfg = OmegaConf.merge(model_cfg, overrides)

        # Add/override with other dynamic parameters
        dynamic_params = OmegaConf.create({
            "batch_size": self.training_config.global_batch_size // world_size,
            "vocab_size": train_metadata.vocab_size,
            "seq_len": train_metadata.seq_len,
            "num_puzzle_identifiers": train_metadata.num_puzzle_identifiers,
            "causal": False,
        })
        model_cfg = OmegaConf.merge(model_cfg, dynamic_params)

        if self.training_config.smoke_test:
            smoke_overrides = OmegaConf.create({
                "puzzle_emb_ndim": 16,
                "num_heads": 1,
                "expansion": 1.0,
            })
            model_cfg = OmegaConf.merge(model_cfg, smoke_overrides)

        # Resolve all interpolations and convert to a plain python dict
        model_cfg_resolved = OmegaConf.to_container(model_cfg, resolve=True)

        # Instantiate model with loss head
        model_cls = load_model_class(model_cfg_resolved['name'])
        loss_config = model_cfg_resolved['loss'].copy()
        loss_head_cls = load_model_class(loss_config.pop('name'))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.device(device):
            model: nn.Module = model_cls(model_cfg)
            model = loss_head_cls(model, **loss_config)
            if "DISABLE_COMPILE" not in os.environ and device.type == "cuda":
                model = torch.compile(model, dynamic=False)

            if world_size > 1:
                # Remove torch.no_grad() to ensure gradients are properly enabled
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

        # Optimizers
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=0,
                weight_decay=self.training_config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            Adam(
                model.parameters(),
                lr=0,
                weight_decay=self.training_config.weight_decay,
                betas=(self.training_config.beta1, self.training_config.beta2),
            ),
        ]
        optimizer_lrs = [self.training_config.puzzle_emb_lr, self.training_config.lr]

        # Estimated total training steps
        total_steps = int(
            self.training_config.epochs
            * train_metadata.total_groups
            * train_metadata.mean_puzzle_examples
            / self.training_config.global_batch_size
        )

        self.train_state = TrainState(
            step=0,
            total_steps=total_steps,
            model=model,
            optimizers=optimizers,
            optimizer_lrs=optimizer_lrs,
            carry=None,
        )

    def train(self, data_path: str, logger_callback: Callable[[str], None], checkpoint_path: Optional[str] = None, run_name: Optional[str] = None) -> Dict[str, Any]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        RANK = 0
        WORLD_SIZE = 1

        if "LOCAL_RANK" in os.environ:
            dist.init_process_group(backend="nccl" if device == "cuda" else "gloo")
            RANK = dist.get_rank()
            WORLD_SIZE = dist.get_world_size()
            if device == "cuda":
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        torch.random.manual_seed(self.training_config.seed + RANK)

        train_epochs_per_iter = self.training_config.eval_interval if self.training_config.eval_interval is not None else self.training_config.epochs
        total_iters = self.training_config.epochs // train_epochs_per_iter

        train_loader, train_metadata = create_dataloader(self.training_config, data_path, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=self.training_config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
        eval_loader, eval_metadata = create_dataloader(self.training_config, data_path, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=self.training_config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

        self._init_train_state(train_metadata, world_size=WORLD_SIZE, rank=RANK)

        if self.training_config.smoke_test:
            self.train_state.total_steps = 1

        progress_bar = None
        logger = None
        if RANK == 0:
            progress_bar = tqdm.tqdm(total=self.train_state.total_steps)
            if checkpoint_path and run_name:
                log_path = os.path.join(checkpoint_path, f"tmp_results_{run_name}.json")
                logger = LocalLogger(log_path=log_path)
                logger.log({"num_params": sum(x.numel() for x in self.train_state.model.parameters())}, step=0)
                # save_code_and_config(checkpoint_path, self.model_config, self.training_config, logger)


        final_metrics = {}
        for _iter_id in range(total_iters):
            logger_callback(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

            self.train_state.model.train()
            for set_name, batch, global_batch_size in train_loader:
                metrics = train_batch(self.training_config, self.train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
                if RANK == 0 and metrics is not None:
                    if logger:
                        logger.log(metrics, self.train_state.step)
                    progress_bar.update(self.train_state.step - progress_bar.n)

            self.train_state.model.eval()
            metrics = evaluate(self.training_config, checkpoint_path, self.train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
            if RANK == 0 and metrics is not None:
                if logger:
                    logger.log(metrics, self.train_state.step)
                final_metrics = metrics

            if RANK == 0 and (self.training_config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                if checkpoint_path:
                    self.save_checkpoint(os.path.join(checkpoint_path, f"step_{self.train_state.step}.pth"))

        if logger:
            logger.finish()

        if dist.is_initialized():
            dist.destroy_process_group()

        return final_metrics

    def evaluate(self, data_path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def load_checkpoint(self, path: str):
        if self.train_state is None or self.train_state.model is None:
            raise RuntimeError("Model must be initialized before loading a checkpoint.")
        self.train_state.model.load_state_dict(torch.load(path))

    def save_checkpoint(self, path: str):
        if self.train_state is None or self.train_state.model is None:
            raise RuntimeError("Model must be trained before saving a checkpoint.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.train_state.model.state_dict(), path)
