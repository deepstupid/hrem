import time
import os
import torch
import torch.distributed as dist
import tqdm
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .utils import (
    LocalLogger,
    create_dataloader,
    train_batch,
    evaluate,
    TrainState,
)
from puzzle_dataset import PuzzleDatasetMetadata
import importlib

class Trainer:
    """
    The Trainer class encapsulates the logic for training and evaluating a model.
    It is responsible for setting up the environment, building the model,
    and running the training and evaluation loops.
    """
    def __init__(self, training_config: Dict[str, Any], model_config: Dict[str, Any], data_config: Dict[str, Any], run_config: Dict[str, Any]):
        """
        Initializes the Trainer.

        Args:
            training_config: The configuration for training.
            model_config: The configuration for the model.
            data_config: The configuration for the data.
            run_config: The configuration for the run.
        """
        self.training_config = training_config
        self.model_config = model_config
        self.data_config = data_config
        self.run_config = run_config

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
        dataset_name = self.data_config['dataset']
        task_name = self.data_config.get('synthetic_task', 'default')
        if dataset_name.startswith("synthetic-"):
            parts = dataset_name.split('-', 1)
            dataset_name = parts[0]
            task_name = parts[1]

        smoke_test = self.run_config.get('smoke_test', False)
        if smoke_test:
            data_dir = f"data/{dataset_name}-{task_name}-smoke"
        else:
            data_dir = f"data/{dataset_name}-{task_name}-full"

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

    def train_and_evaluate(self) -> Dict[str, Any]:
        """
        Runs the full training and evaluation loop.

        Returns:
            A dictionary containing the final metrics.
        """
        self.setup_distributed_training()
        self.prepare_dataloaders()
        self.build_model()

        progress_bar = None
        logger = None
        if self.rank == 0:
            progress_bar = tqdm.tqdm(total=self.train_state.total_steps)
            log_path = os.path.join(self.run_config['output_dir'], f"tmp_results_{self.run_config['study_name']}.json")
            logger = LocalLogger(log_path=log_path)
            logger.log({"num_params": sum(x.numel() for x in self.train_state.model.parameters())}, step=0)

        final_metrics = {}
        train_epochs_per_iter = self.training_config.get('eval_interval', self.training_config['epochs'])
        total_iters = self.training_config['epochs'] // train_epochs_per_iter

        for _iter_id in range(total_iters):
            self.train_state.model.train()
            iter_start_time = time.time()

            for set_name, batch, global_batch_size in self.train_loader:
                metrics = train_batch(self.training_config, self.train_state, batch, global_batch_size, rank=self.rank, world_size=self.world_size)
                if self.rank == 0 and metrics is not None:
                    if logger:
                        logger.log(metrics, self.train_state.step)
                    progress_bar.update(self.train_state.step - progress_bar.n)

            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            avg_epoch_time = iter_duration / train_epochs_per_iter if train_epochs_per_iter > 0 else 0

            self.train_state.model.eval()
            checkpoint_path = self.run_config['output_dir']
            metrics = evaluate(self.training_config, checkpoint_path, self.train_state, self.eval_loader, self.eval_metadata, rank=self.rank, world_size=self.world_size)
            if self.rank == 0 and metrics is not None:
                if logger:
                    logger.log(metrics, self.train_state.step)
                final_metrics = metrics
                final_metrics['avg_epoch_time'] = avg_epoch_time

        if logger:
            logger.finish()
        if dist.is_initialized():
            dist.destroy_process_group()

        return final_metrics
