# Hierarchical Reasoning and Memory (HRM)

![](./assets/hrm.png)

## Overview

This repository contains the PyTorch implementation for a family of models focused on complex reasoning tasks. These models execute sequential reasoning in a single forward pass without explicit supervision of the intermediate steps.

The primary models are:
- **HRM (Hierarchical Reasoning Model)**: A recurrent architecture with two interdependent modules: a high-level module for abstract planning and a low-level module for detailed computations.
- **HREM (Hierarchical Recurrent Execution Model)**: An extension of HRM with a multi-layer memory architecture and sparse memory addressing for more efficient information retrieval.
- **EnhancedHREM**: An improved version of HREM with a better training algorithm, optimized architecture, and faster convergence.

## Quick Start

This project provides a unified command-line interface (CLI) through `run.py` for all major functionalities.

### Evaluation

To run a side-by-side comparison of specified models:
```bash
# Compare HRM and HREM on the synthetic dataset
python run.py evaluate --dataset synthetic --models HRM HREM

# Evaluate the EnhancedHREM model on the ARC dataset
python run.py evaluate --dataset arc --models EnhancedHREM
```

### Hyperparameter Optimization

To run hyperparameter optimization for a model using [Optuna](https://optuna.org/):
```bash
python run.py optimize --dataset arc --model HREM --n-trials 50
```

### Interactive TUI

To launch the Textual User Interface (TUI) for a more interactive experience:
```bash
python run.py tui
```
The TUI allows you to manage datasets, run evaluations, and visualize results.

### Scripted Demo

For a non-interactive, scripted demonstration of a full workflow (baseline evaluation, optimization, and final comparison):
```bash
python run.py demo
```

## Models

### HRM (Hierarchical Reasoning Model)
HRM is a recurrent architecture with two interdependent modules: a high-level module for abstract planning and a low-level module for detailed computations.

### HREM (Hierarchical Recurrent Execution Model)
HREM is an extension of HRM with a multi-layer memory architecture and sparse memory addressing for more efficient information retrieval.

### EnhancedHREM
This is an improved version of HREM designed to be more competitive with HRM. The key enhancements include:

- **Improved Model Architecture**: Increased memory capacity, improved memory retrieval with more top-k values, additional processing layers, and a larger model capacity overall.
- **Better Training Algorithm**: Uses the AdamW optimizer for superior weight decay handling and better numerical stability.
- **Enhanced Training Utilities**: Faster data loading, support for more workers, linear learning rate scheduling, and mixed-precision (AMP) training.
- **Dynamic Reporting System**: The reporting system can dynamically handle any number of models, providing flexible comparisons and enhanced metrics.

These enhancements lead to several performance improvements, including faster training, better convergence, and enhanced stability.

## Optimization and Evaluation Framework

The repository includes a comprehensive framework for model evaluation and hyperparameter tuning.

- **Unified Interface**: All modes (evaluation, optimization, demo, TUI) are accessible through `run.py`.
- **Hyperparameter Tuning**: Utilizes Optuna for efficient hyperparameter searches.
- **Detailed Reporting**: Generates Markdown reports with detailed metrics and comparisons.
- **Scientific Analysis**: Provides statistical significance testing (t-tests) to compare model performance.

## Prerequisites

- Python 3.8+
- PyTorch
- CUDA (for GPU acceleration)

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Citation

If you use this work, please cite:
```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```