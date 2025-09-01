import torch
from typing import Dict, Any, Tuple
from models.hrm.hrem import HREM

class InteractiveRunner:
    """
    Handles the step-by-step execution of a model on a single puzzle instance for interactive visualization.
    """

    def __init__(self, model, is_hrem: bool):
        self.model = model
        self.is_hrem = is_hrem
        self.model.eval()
        self.carry = None
        self.mem_states = {}

    def reset(self, batch: Dict[str, torch.Tensor]):
        """Initializes the carry and memory states for a new puzzle."""
        with torch.inference_mode():
            if self.is_hrem:
                self.carry, self.mem_states = self.model.initial_carry(batch)
            else:
                self.carry = self.model.initial_carry(batch)
                self.mem_states = {}

    def run_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs a single step of the model.

        Args:
            batch: A dictionary containing the input data for a single step.

        Returns:
            A tuple containing the predictions and metrics for the step.
        """
        with torch.inference_mode():
            model_input_carry = (self.carry, self.mem_states) if self.is_hrem else self.carry

            new_carry, _, metrics, preds, _ = self.model(
                return_keys=['logits'],
                carry=model_input_carry,
                batch=batch
            )

            if self.is_hrem:
                self.carry, self.mem_states = new_carry
            else:
                self.carry = new_carry

            # Bug fix: use index 0 for the sequence dimension, as we process one step at a time.
            prediction = preds['logits'][:, 0, :].argmax(dim=-1)
            correct = (prediction == batch['labels'].squeeze()).item()

            step_result = {
                "prediction": prediction.item(),
                "correct": correct,
            }

            step_metrics = {k: v.item() for k, v in metrics.items() if k != 'count'}

            return step_result, step_metrics
