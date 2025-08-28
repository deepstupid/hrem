from hrm_system.algorithms.torch_base_algorithm import TorchBaseAlgorithm
from utils.functions import load_model_class
import torch
import yaml
import os


class HREMAlgorithm(TorchBaseAlgorithm):
    """
    The HREM algorithm implementation. It inherits the common training logic
    from TorchBaseAlgorithm but handles the specific initialization required
    for HREM with external memory.
    """
    
    def _init_train_state(self, train_metadata, world_size: int, rank: int):
        # Load base architecture config
        arch_config_path = f"config/arch/{self.model_config.base_arch_config}.yaml"
        with open(arch_config_path, 'r') as f:
            arch_config = yaml.safe_load(f)

        # Model config
        hrem_params = self.model_config.hrem_params.model_dump() if self.model_config.hrem_params else {}

        model_cfg = arch_config.copy()
        model_cfg.update(hrem_params)

        # Add/override with other dynamic parameters
        model_cfg.update({
            "batch_size": self.training_config.global_batch_size // world_size,
            "vocab_size": train_metadata.vocab_size,
            "seq_len": train_metadata.seq_len,
            "num_puzzle_identifiers": train_metadata.num_puzzle_identifiers,
            "causal": False,
        })

        # Apply overrides from model_config
        if self.model_config.arch_overrides:
            model_cfg.update(self.model_config.arch_overrides)

        if self.training_config.smoke_test:
            model_cfg.update({
                "puzzle_emb_ndim": 16,
                "num_heads": 1,
                "expansion": 1.0,
                "use_memory": True,  # Ensure memory is used even in smoke test
                "m_loc": 8,
                "d_mem": 8,
                "top_k": 2,
            })

        # Instantiate model with loss head
        model_cls = load_model_class(arch_config['name'])
        loss_config = arch_config['loss'].copy()
        loss_head_cls = load_model_class(loss_config.pop('name'))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.device(device):
            model = model_cls(model_cfg)
            model = loss_head_cls(model, **loss_config)
            if "DISABLE_COMPILE" not in os.environ and device.type == "cuda":
                model = torch.compile(model, dynamic=False)

            if world_size > 1:
                # Remove torch.no_grad() to ensure gradients are properly enabled
                for param in list(model.parameters()) + list(model.buffers()):
                    torch.distributed.broadcast(param, src=0)

        # Optimizers
        from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
        from torch.optim import Adam, AdamW

        optimizer_class = Adam if self.training_config.optimizer == "Adam" else AdamW
        
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),
                lr=0,
                weight_decay=self.training_config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            optimizer_class(
                model.parameters(),
                lr=0,
                weight_decay=self.training_config.weight_decay,
                betas=(self.training_config.beta1, self.training_config.beta2),
                eps=self.training_config.optimizer_eps,
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

        if self.training_config.smoke_test:
            total_steps = 1

        # Use the existing TrainState class
        from hrm_system.algorithms.utils import TrainState
        
        self.train_state = TrainState(
            step=0,
            total_steps=total_steps,
            model=model,
            optimizers=optimizers,
            optimizer_lrs=optimizer_lrs,
            carry=None,
        )
