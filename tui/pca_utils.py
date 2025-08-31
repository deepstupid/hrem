import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

def get_pca_2d(hparams: List[Dict[str, Any]]) -> np.ndarray:
    """
    Performs PCA on a list of hyperparameter configurations to get 2D coordinates.
    """
    if not hparams:
        return np.array([])

    # For now, assume all hyperparameters are numerical.
    # A more robust implementation would handle categorical features.
    try:
        data = np.array([[v for v in d.values()] for d in hparams])
    except (ValueError, TypeError):
        # Fallback for non-numerical data: create a simple grid for now
        num_points = len(hparams)
        side_len = int(np.ceil(np.sqrt(num_points)))
        x = np.linspace(0, 1, side_len)
        y = np.linspace(0, 1, side_len)
        xv, yv = np.meshgrid(x, y)
        return np.vstack([xv.ravel(), yv.ravel()]).T[:num_points]

    if data.shape[1] < 2:
        # Not enough dimensions for PCA, pad with zeros
        padded_data = np.zeros((data.shape[0], 2))
        padded_data[:, :data.shape[1]] = data
        return padded_data

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    return pca_result
