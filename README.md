# The Scientific Discovery Engine

## Overview

This repository contains the implementation of the **Scientific Discovery Engine**, a framework for modeling and automating algorithmic discovery. It provides a testbed for a core theory: that scientific progress can be viewed as a resource allocation problem. Researchers invest finite resources—such as time, computational energy, and attention—to conduct experiments. The goal of this engine is to maximize the scientific insights gained from a given **patience budget**.

The primary models currently being explored are:
- **HRM (Hierarchical Reasoning Model)**: A recurrent architecture with two interdependent modules: a high-level module for abstract planning and a low-level module for detailed computations.
- **HREM (Hierarchical Recurrent Execution Model)**: An extension of HRM with a multi-layer memory architecture and sparse memory addressing for more efficient information retrieval.

## Core Philosophy: Discovery as Investment

The engine is built on a theory of automated science where **discovery is the return on an investment of patience**.

1.  **Investment**: A researcher's primary investment is their **patience**. This is a finite resource, representing the time and computational energy they are willing to spend on a line of inquiry. In the engine, this is formalized as a `PatienceBudget`.
2.  **Automation**: The engine automates the process of scientific inquiry. It runs experiments, compares algorithms, and tunes parameters, all while staying within the allocated patience budget.
3.  **Discovery**: The return on investment is **scientific insight**. The engine continuously analyzes experimental results to generate novel, actionable conclusions about the algorithms being studied.
4.  **Adaptive Exploration**: The engine doesn't explore blindly. It uses performance signals to dynamically allocate resources, focusing the search on the most promising areas of the solution space to maximize the rate of discovery.

## Graphical User Interface

The primary way to interact with the Scientific Discovery Engine is through its graphical user interface (GUI).

```bash
python run.py gui
```

The GUI is designed to directly reflect the core philosophy of investment vs. discovery:

1.  **The Investment Screen**: This is the "Create Experiment" view where you, the researcher, make your initial investment. You define the scientific question by selecting a challenge, choosing algorithms, and—most importantly—setting the **Patience Level** for the experiment.
2.  **The Discovery Screen**: This is the "Run Experiment" view where you observe the return on your investment in real-time. You can monitor the consumption of your patience budget, see live performance metrics, and watch as the engine generates new scientific insights.

## Quick Start (CLI)

While the GUI is the recommended interface, the engine can also be run from the command line for scripting and batch processing.

### Run an Evaluation
```bash
# Compare HRM and HREM on the synthetic sorting challenge
python run.py evaluate --challenge synthetic_sort
```

### Run Hyperparameter Optimization
```bash
# Optimize HREM for 50 trials
python run.py optimize --challenge synthetic_sort --model-to-optimize HREM --n-trials 50
```

## Prerequisites

- Python 3.8+
- PyQt6 (for the GUI)
- PyTorch
- CUDA (for GPU acceleration)

Install the required Python packages:
```bash
pip install -r requirements.txt
pip install PyQt6
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