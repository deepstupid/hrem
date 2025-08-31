import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any

def generate_performance_plot(log_histories: Dict[str, List[Dict[str, Any]]], output_dir: str = "plots") -> str:
    """
    Generates a plot of performance metrics vs. training steps for multiple algorithms.

    Args:
        log_histories: A dictionary where keys are algorithm names and values are lists of log entries.
        output_dir: The directory to save the plot in.

    Returns:
        The path to the saved plot image, or an empty string if no plot was generated.
    """
    if not log_histories:
        return ""

    fig, ax = plt.subplots(figsize=(12, 8))

    has_data = False
    for alg_name, history in log_histories.items():
        # Filter out entries that don't have the required keys
        filtered_history = [entry for entry in history if 'step' in entry and 'train/lm_loss' in entry]

        if not filtered_history:
            continue

        steps = [entry['step'] for entry in filtered_history]
        losses = [entry['train/lm_loss'] for entry in filtered_history]

        if steps and losses:
            ax.plot(steps, losses, label=alg_name, marker='o', linestyle='-')
            has_data = True

    if not has_data:
        plt.close(fig)
        return ""

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Train LM Loss")
    ax.set_title("Training Performance Comparison")
    ax.legend()
    ax.grid(True)

    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename to avoid overwriting
    plot_filename = f"performance_plot_{len(os.listdir(output_dir)) + 1}.png"
    output_path = os.path.join(output_dir, plot_filename)

    fig.savefig(output_path)
    plt.close(fig)

    return output_path
