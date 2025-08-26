# Hierarchical Reasoning Model (HRM) and Hierarchical Recurrent Execution Model (HREM)

![](./assets/hrm.png)

## Overview

This repository contains the PyTorch implementation for the Hierarchical Reasoning Model (HRM) and the Hierarchical Recurrent Execution Model (HREM). These models are designed for complex reasoning tasks, executing sequential reasoning in a single forward pass without explicit supervision of the intermediate steps.

- **HRM**: A recurrent architecture with two interdependent modules: a high-level module for abstract planning and a low-level module for detailed computations.
- **HREM**: An extension of HRM with a multi-layer memory architecture and sparse memory addressing for more efficient information retrieval.
- **Enhanced HREM**: An improved version of HREM with a better training algorithm, optimized architecture, and faster convergence.

## Quick Start

This project provides a unified command-line interface (CLI) for running evaluations, hyperparameter optimization, and interactive demonstrations.

### Evaluation

To run a side-by-side comparison of specified models:
```bash
python run.py evaluate --dataset synthetic --models HRM HREM
```

### Hyperparameter Optimization

To run hyperparameter optimization for a model:
```bash
python run.py optimize --dataset arc --model HREM --n-trials 50
```

### Interactive Demo

For a guided experience through baseline evaluation, optimization, and final comparison, run the interactive demo:
```bash
python run.py demo
```

## Optimization and Evaluation Framework

The repository includes a comprehensive framework for model evaluation and hyperparameter tuning.

- **Unified Interface**: All modes (evaluation, optimization, demo) are accessible through `run.py`.
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

For detailed setup instructions, including specific CUDA and PyTorch versions, please refer to the original `README.md` in the project's commit history.

## Citation

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