import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import optuna
from typing import Optional

def plot_hyperparameter_pca(study: optuna.study.Study, model_name: str) -> Optional[str]:
    """
    Generates a PCA plot of the hyperparameter search space.

    Args:
        study: The Optuna study object.
        model_name: The name of the model being optimized.

    Returns:
        The path to the saved plot image, or None if the plot could not be generated.
    """
    try:
        # Extract trial data
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(trials) < 2:
            return None

        # Create a DataFrame
        data = []
        for trial in trials:
            row = trial.params
            row['value'] = trial.value
            data.append(row)

        df = pd.DataFrame(data)

        # Separate features (params) and target (value)
        params = df.drop('value', axis=1)
        values = df['value']

        # Handle categorical parameters by one-hot encoding
        params_processed = pd.get_dummies(params, drop_first=True)

        # Perform PCA
        pca = PCA(n_components=2)
        params_pca = pca.fit_transform(params_processed)

        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(params_pca[:, 0], params_pca[:, 1], c=values, cmap='viridis_r', alpha=0.8)
        plt.colorbar(scatter, label='Objective Value (Loss)')
        plt.title(f'PCA of Hyperparameter Search Space for {model_name}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)

        # Save plot
        filepath = f"hparam_pca_{model_name}.png"
        plt.savefig(filepath)
        plt.close()

        return filepath
    except Exception as e:
        print(f"Error generating PCA plot: {e}")
        return None
