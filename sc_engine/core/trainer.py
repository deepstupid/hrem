import time
import os
import torch
import torch.distributed as dist
import tqdm
from typing import Dict, Any, Optional

from .utils import (
    LocalLogger,
    create_dataloader,
    compute_lr,
    TrainState,
)
from .progress_handler import ProgressHandler
from puzzle_dataset import PuzzleDatasetMetadata
import importlib
import threading

class Trainer:
    """
    The Trainer class encapsulates the logic for training and evaluating a model.
    It is responsible for setting up the environment, building the model,
    and running the training and evaluation loops.
    """
    def __init__(self, training_config: Dict[str, Any], model_config: Dict[str, Any], data_config: Dict[str, Any], run_config: Dict[str, Any], progress_handler: Optional[ProgressHandler] = None, cancel_event: Optional[threading.Event] = None):
        """
        Initializes the Trainer.

        Args:
            training_config: The configuration for training.
            model_config: The configuration for the model.
            data_config: The configuration for the data.
            run_config: The configuration for the run.
            progress_handler: An optional handler for reporting progress.
            cancel_event: An optional event to signal cancellation.
        """
        self.training_config = training_config
        self.model_config = model_config
        self.data_config = data_config
        self.run_config = run_config
        self.progress_handler = progress_handler
        self.cancel_event = cancel_event or threading.Event()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rank = 0
        self.world_size = 1
        self.train_state: Optional[TrainState] = None

    def setup_distributed_training(self):
        """
        Sets up distributed training if the environment variables are set.
        """
        if "LOCAL_RANK" in os.environ:
            dist.init_process_group(backend="nccl" if self.device == "cuda" else "gloo")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            if self.device == "cuda":
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        torch.random.manual_seed(self.training_config['seed'] + self.rank)

    def prepare_dataloaders(self):
        """
        Prepares the training and evaluation dataloaders based on the data config.
        """
        # The data_config['dataset'] is now expected to be a full path to the dataset directory.
        data_dir = self.data_config['dataset']

        train_epochs_per_iter = self.training_config.get('eval_interval', self.training_config['epochs'])

        self.train_loader, self.train_metadata = create_dataloader(
            self.training_config, data_dir, "train", test_set_mode=False,
            epochs_per_iter=train_epochs_per_iter,
            global_batch_size=self.training_config['global_batch_size'],
            rank=self.rank,
            world_size=self.world_size
        )
        self.eval_loader, self.eval_metadata = create_dataloader(
            self.training_config, data_dir, "test", test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=self.training_config['global_batch_size'],
            rank=self.rank,
            world_size=self.world_size
        )

    def build_model(self):
        """
        Builds the model and initializes the training state.
        It uses the algorithm class specified in the model config to instantiate the algorithm.
        """
        module_path, class_name = self.model_config['algorithm_class'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        algorithm_class = getattr(module, class_name)
        algorithm = algorithm_class(self.model_config, self.training_config)

        algorithm.initialize_train_state(self.train_metadata, world_size=self.world_size, rank=self.rank)
        self.train_state = algorithm.train_state

        if self.training_config['smoke_test']:
            self.train_state.total_steps = 1

    def initialize(self):
        """Initializes the trainer, including data loaders and model."""
        # self.setup_distributed_training() # This can hang in non-distributed environments
        self.prepare_dataloaders()
        self.build_model()
        self.train_loader_iter = iter(self.train_loader)

    def train_batch(self):
        """Trains the model on a single batch of data."""
        try:
            _, batch, global_batch_size = next(self.train_loader_iter)
        except StopIteration:
            return None, True # Indicates epoch is finished

        # torch._functorch.config.donated_buffer = False # This is deprecated
        self.train_state.step += 1
        if self.train_state.step > self.train_state.total_steps:
            return None, True

        device = torch.device(self.device)
        batch = {k: v.to(device) for k, v in batch.items()}

        if self.train_state.carry is None:
            with torch.device(device):
                self.train_state.carry = self.train_state.model.initial_carry(batch)

        use_amp = self.training_config['use_amp'] and device.type == "cuda"
        if use_amp and self.train_state.scaler is None:
            self.train_state.scaler = torch.cuda.amp.GradScaler()

        scaler = self.train_state.scaler

        if use_amp:
            with torch.cuda.amp.autocast():
                new_carry, loss, metrics, _, _ = self.train_state.model(carry=self.train_state.carry, batch=batch, return_keys=[])
        else:
            new_carry, loss, metrics, _, _ = self.train_state.model(carry=self.train_state.carry, batch=batch, return_keys=[])

        self.train_state.carry = new_carry

        # --- Backpropagation and Optimizer Step ---
        # Only perform backpropagation if there are optimizers (i.e., the model is trainable)
        if self.train_state.optimizers:
            if use_amp and scaler is not None:
                scaled_loss = scaler.scale((1 / global_batch_size) * loss)
                scaled_loss.backward()
            else:
                scaled_loss = (1 / global_batch_size) * loss
                scaled_loss.backward()

            del scaled_loss, loss

            if self.world_size > 1:
                for param in self.train_state.model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad)

            lr_this_step = None
            for optim, base_lr in zip(self.train_state.optimizers, self.train_state.optimizer_lrs):
                lr_this_step = compute_lr(base_lr, self.training_config, self.train_state)

                for param_group in optim.param_groups:
                    param_group['lr'] = lr_this_step

                if use_amp and scaler is not None:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad()
        else:
            # If not trainable, we still need to set lr for logging
            lr_this_step = 0

        if len(metrics):
            if 'loss' in metrics and isinstance(metrics['loss'], torch.Tensor):
                assert not metrics['loss'].requires_grad

            metric_keys = list(sorted([k for k, v in metrics.items() if isinstance(v, torch.Tensor)]))

            if not metric_keys:
                # Handle case where no tensor metrics are present
                reduced_metrics = {k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)}
                reduced_metrics["train/lr"] = lr_this_step
                self._send_progress('train_batch', {'metrics': reduced_metrics, 'step': self.train_state.step, 'total_steps': self.train_state.total_steps})
                return reduced_metrics, False

            metric_values = torch.stack([metrics[k] for k in metric_keys])
            if self.world_size > 1:
                dist.reduce(metric_values, dst=0)

            if self.rank == 0:
                metric_values = metric_values.cpu().numpy()
                reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

                count = max(reduced_metrics.get("count", 1), 1)
                reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

                reduced_metrics["train/lr"] = lr_this_step
                self._send_progress('train_batch', {'metrics': reduced_metrics, 'step': self.train_state.step, 'total_steps': self.train_state.total_steps})
                return reduced_metrics, False
        return None, False

    def evaluate(self, checkpoint_path: Optional[str]):
        """Runs evaluation on the test set."""
        with torch.inference_mode():
            set_ids = {k: idx for idx, k in enumerate(self.eval_metadata.sets)}
            all_preds = {}
            metric_keys = []
            tensor_metric_values = None
            non_tensor_metrics = [{} for _ in range(len(set_ids))]
            device = torch.device(self.device)

            self._send_progress('start_evaluation_phase')
            for set_name, batch, global_batch_size in self.eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.device(device):
                    carry = self.train_state.model.initial_carry(batch)

                while True:
                    carry, _, metrics, preds, all_finish = self.train_state.model(
                        carry=carry, batch=batch, return_keys=self.training_config['eval_save_outputs']
                    )
                    if all_finish:
                        break

                for collection in (batch, preds):
                    for k, v in collection.items():
                        if k in self.training_config['eval_save_outputs']:
                            all_preds.setdefault(k, [])
                            all_preds[k].append(v.cpu())

                del carry, preds, batch, all_finish

                set_id = set_ids[set_name]
                if tensor_metric_values is None:
                    metric_keys = list(sorted([k for k, v in metrics.items() if isinstance(v, torch.Tensor)]))
                    tensor_metric_values = torch.zeros((len(set_ids), len(metric_keys)), dtype=torch.float32, device=device)

                tensor_metrics_stacked = torch.stack([metrics[k] for k in metric_keys])
                tensor_metric_values[set_id] += tensor_metrics_stacked

                for k, v in metrics.items():
                    if not isinstance(v, torch.Tensor):
                        non_tensor_metrics[set_id][k] = non_tensor_metrics[set_id].get(k, 0) + v

            if len(all_preds) and checkpoint_path is not None:
                all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(all_preds, os.path.join(checkpoint_path, f"step_{self.train_state.step}_all_preds.{self.rank}"))

            if tensor_metric_values is not None:
                if self.world_size > 1:
                    dist.reduce(tensor_metric_values, dst=0)
                if self.rank == 0:
                    reduced_metrics = tensor_metric_values.cpu().numpy()
                    final_metrics = {}
                    for set_id, set_name in enumerate(set_ids):
                        final_metrics[set_name] = {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
                        final_metrics[set_name].update(non_tensor_metrics[set_id])

                    for set_name, metrics in final_metrics.items():
                        count = metrics.pop("count", 1)
                        final_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                    self._send_progress('end_evaluation_phase', {'metrics': final_metrics})
                    return final_metrics
        return None

    def run_sequential_training(self) -> Dict[str, Any]:
        """
        Runs the full training and evaluation loop sequentially.
        This is the original train_and_evaluate functionality.
        """
        self.initialize()

        logger = None
        if self.rank == 0:
            log_path = os.path.join(self.run_config['output_dir'], f"tmp_results_{self.run_config['study_name']}.json")
            logger = LocalLogger(log_path=log_path)
            logger.log({"num_params": sum(x.numel() for x in self.train_state.model.parameters())}, step=0)
            self._send_progress('start_training', {'total_steps': self.train_state.total_steps})

        final_metrics = {}

        # The main training loop, broken down by steps
        for step in range(self.train_state.total_steps):
            # --- Check for pause/cancel signals ---
            if self.cancel_event.is_set():
                self._send_progress('training_cancelled', {'step': step})
                break

            metrics, is_finished = self.train_batch()
            if is_finished:
                break
            if self.rank == 0 and metrics is not None and logger:
                logger.log(metrics, self.train_state.step)

        self.train_state.model.eval()
        checkpoint_path = self.run_config['output_dir']
        metrics = self.evaluate(checkpoint_path)
        if self.rank == 0 and metrics is not None:
            if logger:
                logger.log(metrics, self.train_state.step)
            final_metrics = metrics

        log_history = []
        if logger:
            log_history = logger.get_log_history()
            logger.finish()
        if dist.is_initialized():
            dist.destroy_process_group()

        self._send_progress('end_training', {'final_metrics': final_metrics})
        return final_metrics, log_history

    def _send_progress(self, event_type: str, data: Dict = None):
        """Send progress update via handler if available."""
        if self.progress_handler:
            self.progress_handler.on_progress(f'trainer:{event_type}', data if data is not None else {})
