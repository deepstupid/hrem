# Hierarchical Reasoning and Memory (HRM)

![](./assets/hrm.png)

## Overview

This repository contains the PyTorch implementation for a family of models focused on complex reasoning tasks. These models execute sequential reasoning in a single forward pass without explicit supervision of the intermediate steps.

The primary models are:
- **HRM (Hierarchical Reasoning Model)**: A recurrent architecture with two interdependent modules: a high-level module for abstract planning and a low-level module for detailed computations.
- **HREM (Hierarchical Recurrent Execution Model)**: An extension of HRM with a multi-layer memory architecture and sparse memory addressing for more efficient information retrieval.

## Quick Start

This project is run through a unified command-line interface (CLI) in `run.py`. The primary entry point is the `compare` command, which uses a Scientific Discovery Engine to run a rigorous, patience-aware comparison between algorithms.

### Scientific Comparison

To run a full comparison for a pre-defined scientific challenge:
```bash
# Run the default comparison challenge for sorting algorithms
python run.py compare
```

You can specify a different challenge and patience level:
```bash
# Run the long-range dependency challenge with a high patience budget
python run.py compare --challenge long_range_dependencies --patience high
```

The engine performs baseline evaluation, hyperparameter optimization, and a final comparison, then generates scientific insights about the algorithms' performance. For more details on the architecture of this engine, see `ARCHITECTURE.md`.

### Other Commands

While `compare` is the recommended workflow, the following direct commands are also available:
- `evaluate`: For a simple side-by-side model evaluation.
- `optimize`: To run hyperparameter optimization for a single model.
- `tui`: To launch the Textual User Interface for managing runs.

## Models

### HRM (Hierarchical Reasoning Model)
HRM is a recurrent architecture with two interdependent modules: a high-level module for abstract planning and a low-level module for detailed computations.

### HREM (Hierarchical Recurrent Execution Model)
HREM is an extension of HRM with a multi-layer memory architecture and sparse memory addressing for more efficient information retrieval.

## Scientific Comparison Framework

The repository includes a Scientific Discovery Engine for rigorous algorithm comparison.

- **Unified Workflow**: The `compare` command provides an integrated workflow for baseline evaluation, hyperparameter optimization, and final comparison.
- **Patience-Aware Execution**: The engine manages a time budget ("patience") to ensure that comparisons complete within a reasonable timeframe.
- **Insight Generation**: The framework analyzes results to produce scientific insights into algorithm performance.
- **Extensible**: The engine is designed to be extensible with new algorithms and scientific challenges. See `ARCHITECTURE.md` for details.

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