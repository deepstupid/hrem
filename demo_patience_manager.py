from typing import Dict

class PatienceManager:
    """Manages the demo's time budget based on user patience and baseline metrics."""

    def __init__(self, patience_in_seconds: int, timing_metrics: Dict[str, float]):
        """
        Initialize the PatienceManager.

        Args:
            patience_in_seconds: The total time budget for the demo.
            timing_metrics: A dictionary mapping model names to their average epoch time.
        """
        self.total_budget = patience_in_seconds
        self.timing_metrics = timing_metrics

    def get_optimization_trial_budget(self, model_name: str, epochs_per_trial: int, time_spent_so_far: float) -> int:
        """
        Calculate the number of optimization trials that can be run.

        Args:
            model_name: The name of the model being optimized.
            epochs_per_trial: The number of epochs for each trial.
            time_spent_so_far: The time already spent in the demo.

        Returns:
            The number of trials that can be run.
        """
        if model_name not in self.timing_metrics:
            # If we have no timing info, fall back to a small number of trials
            return 3

        avg_epoch_time = self.timing_metrics[model_name]
        estimated_trial_time = avg_epoch_time * epochs_per_trial

        if estimated_trial_time <= 0:
            return 3 # Avoid division by zero

        remaining_time = self.total_budget - time_spent_so_far

        # Allocate 70% of the *remaining* time to optimization
        time_for_optimization = remaining_time * 0.7

        num_trials = int(time_for_optimization / estimated_trial_time)

        # Ensure at least 1 trial is run, and not more than a reasonable max
        return max(1, min(num_trials, 50))
