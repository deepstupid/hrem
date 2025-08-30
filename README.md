# Real-Time Scientific Algorithm Comparison Engine

## Overview

This repository contains the implementation of a scientifically-rigorous framework for real-time comparison of algorithms (e.g., HRM vs HREM). The system balances computational exploration with user patience constraints to maximize the potential for scientific discovery within practical time limits.

The primary models compared are:
- **HRM (Hierarchical Reasoning Model)**: A recurrent architecture with two interdependent modules: a high-level module for abstract planning and a low-level module for detailed computations.
- **HREM (Hierarchical Recurrent Execution Model)**: An extension of HRM with a multi-layer memory architecture and sparse memory addressing for more efficient information retrieval.

The core of the repository is the **Scientific Discovery Engine**, a flexible and extensible system for running, comparing, and analyzing complex machine learning models.

## Quick Start

This project provides a unified command-line interface (CLI) through `run.py` for all major functionalities.

### Scientific Comparison

To run a side-by-side comparison of specified models for a given scientific challenge:
```bash
# Compare HRM and HREM on the synthetic sorting challenge
python run.py compare --challenge synthetic_sort
```

### Hyperparameter Optimization

To run hyperparameter optimization for a model:
```bash
# Optimize HREM on the synthetic sorting challenge for 50 trials
python run.py optimize --challenge synthetic_sort --model-to-optimize HREM --n-trials 50
```

### Scripted Demo

For a non-interactive, scripted demonstration of a full workflow (baseline evaluation, optimization, and final comparison):
```bash
python run.py demo --challenge synthetic_sort
```

### Interactive TUI

To launch the Textual User Interface (TUI) for a more interactive experience:
```bash
python run.py tui
```
The TUI allows you to manage datasets, run evaluations, and visualize results.

## Core Philosophy

Scientific discovery in algorithm comparison requires:
1. **Challenge-Driven Focus**: Each comparison centers around a specific scientific problem.
2. **Algorithm-First Design**: Algorithms define the exploration methodology.
3. **Adaptive Exploration**: Resource allocation responds dynamically to performance signals.
4. **Discovery Maximization**: Optimizing for insight generation rather than just performance metrics.

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