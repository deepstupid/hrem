from sc_engine.core.utils import TrainState
from sc_engine.utils.functions import load_model_class
import torch
import os
from omegaconf import OmegaConf

class HREMAlgorithm:
    """
    The HREM algorithm implementation.
    This class is responsible for initializing the model, optimizers, and training state.
    """
    def __init__(self, model_config: dict, training_config: dict):
        self.model_config = model_config
        self.training_config = training_config
        self.train_state = None

    def initialize_train_state(self, train_metadata, world_size: int, rank: int):
        # Load base architecture config
        arch_config_name = self.model_config.get('base_arch_config')
        if not arch_config_name:
            raise ValueError("base_arch_config not specified in model config")

        arch_config_path = os.path.join('config', 'arch', f"{arch_config_name}.yaml")
        try:
            model_cfg_omega = OmegaConf.load(arch_config_path)
        except FileNotFoundError:
            raise ValueError(f"Architecture config file not found: {arch_config_path}")

        # Create a dictionary of overrides from the model-specific config
        overrides = {k: v for k, v in self.model_config.items() if k not in ['name', 'base_arch_config']}

        # Explicitly enable memory for HREM
        overrides['use_memory'] = True

        model_cfg_omega = OmegaConf.merge(model_cfg_omega, OmegaConf.create(overrides))

        # Create a dictionary for dynamic parameters and overrides
        dynamic_params = {
            "batch_size": self.training_config['global_batch_size'] // world_size,
            "vocab_size": train_metadata.vocab_size,
            "seq_len": train_metadata.seq_len,
            "num_puzzle_identifiers": train_metadata.num_puzzle_identifiers,
            "causal": False,
        }

        # Apply smoke test overrides if applicable
        if self.training_config.get('smoke_test'):
            dynamic_params.update({
                "puzzle_emb_ndim": 16,
                "num_heads": 1,
                "expansion": 1.0,
            })

        # Merge dynamic params and overrides
        model_cfg_omega = OmegaConf.merge(model_cfg_omega, OmegaConf.create(dynamic_params))

        # Resolve all interpolations and get the final config dictionary
        model_cfg = OmegaConf.to_container(model_cfg_omega, resolve=True)

        # Instantiate model with loss head
        model_cls = load_model_class(model_cfg['name'])
        loss_config = model_cfg['loss'].copy()
        loss_head_cls = load_model_class(loss_config.pop('name'))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.device(device):
            model = model_cls(model_cfg)
            model = loss_head_cls(model, **loss_config)
            if "DISABLE_COMPILE" not in os.environ and device.type == "cuda":
                model = torch.compile(model, dynamic=False)

            if world_size > 1:
                for param in list(model.parameters()) + list(model.buffers()):
                    torch.distributed.broadcast(param, src=0)

        # Optimizers
        from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
        from torch.optim import Adam, AdamW

        optimizer_class = Adam if self.training_config['optimizer'] == "Adam" else AdamW

        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.inner.puzzle_emb.buffers(),
                lr=0,
                weight_decay=self.training_config['puzzle_emb_weight_decay'],
                world_size=world_size,
            ),
            optimizer_class(
                model.parameters(),
                lr=0,
                weight_decay=self.training_config['weight_decay'],
                betas=(self.training_config['beta1'], self.training_config['beta2']),
                eps=self.training_config['optimizer_eps'],
            ),
        ]
        optimizer_lrs = [self.training_config['puzzle_emb_lr'], self.training_config['lr']]

        # Estimated total training steps
        total_steps = int(
            self.training_config['epochs']
            * train_metadata.total_groups
            * train_metadata.mean_puzzle_examples
            / self.training_config['global_batch_size']
        )

        self.train_state = TrainState(
            step=0,
            total_steps=total_steps,
            model=model,
            optimizers=optimizers,
            optimizer_lrs=optimizer_lrs,
            carry=None,
        )

    def train(self, data_path: str, logger_callback: callable, checkpoint_path: str, run_name: str):
        logger_callback("[bold yellow]Warning: Training loop not yet implemented in the refactored algorithm.[/bold yellow]")
        pass
