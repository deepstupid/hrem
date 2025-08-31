import torch
from sc_engine.core.utils import TrainState

class RandomModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, carry, batch, return_keys=[]):
        # The trainer expects a loss value, even if it's not used for optimization.
        loss = torch.tensor(0.0, device=batch['inputs'].device)

        # The trainer also expects metrics.
        metrics = {'loss': loss.item(), 'count': batch['inputs'].size(0)}

        # No predictions are generated, and we are always "finished".
        preds = {}
        all_finish = torch.tensor(True)

        return carry, loss, metrics, preds, all_finish

    def initial_carry(self, batch):
        return None

class RandomAlgorithm:
    """
    A simple algorithm that returns random predictions.
    """
    def __init__(self, model_config: dict, training_config: dict):
        self.model_config = model_config
        self.training_config = training_config
        self.train_state = None

    def initialize_train_state(self, train_metadata, world_size: int, rank: int):
        model = RandomModel(train_metadata.vocab_size)

        self.train_state = TrainState(
            step=0,
            total_steps=1,
            model=model,
            optimizers=[],
            optimizer_lrs=[],
            carry=None,
        )

    def train(self, data_path: str, logger_callback: callable, checkpoint_path: str, run_name: str):
        logger_callback("[bold yellow]RandomAlgorithm does not require training.[/bold yellow]")
        pass
