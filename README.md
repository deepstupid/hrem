# Hierarchical Reasoning Model (HRM) and Hierarchical Recurrent Execution Model (HREM)

![](./assets/hrm.png)

## Overview

Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI.
Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency.

HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes.

Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities.

Building upon HRM, we introduce the Hierarchical Recurrent Execution Model (HREM), which extends the hierarchical approach with a multi-layer memory architecture. HREM introduces sparse memory addressing mechanisms that enable more efficient information retrieval and storage across multiple abstraction levels.

These results underscore HRM/HREM's potential as a transformative advancement toward universal computation and general-purpose reasoning systems.

## Quick Start with CLI Demo 🚀

The easiest way to explore the capabilities of HRM and HREM is through our interactive command-line interface (CLI) demo:

```bash
python run_demo_cli.py
```

This will present you with a menu of pre-configured challenges ranging from simple synthetic tasks to complex real-world problems:

- **Beginner**: Simple copy/reverse tasks for testing basic functionality
- **Intermediate**: More complex synthetic tasks with data augmentation
- **Advanced**: Real-world challenges like ARC, Sudoku, and Maze navigation
- **Research**: Full-scale datasets for breakthrough research

Each challenge is designed to work with different computational resources:
- **Beginner**: Runs on any hardware (CPU or GPU)
- **Intermediate**: Requires GPU for reasonable training time
- **Advanced**: Requires significant GPU memory and compute
- **Research**: Requires powerful multi-GPU setup

### Fast Mode and Interactive Mode

Add `--fast` flag for quick testing:
```bash
python run_demo_cli.py --fast
```

Add `--interactive` flag for step-by-step execution:
```bash
python run_demo_cli.py --interactive
```

## HRM vs HREM: Architectural Differences

### Hierarchical Reasoning Model (HRM)
- Two interdependent recurrent modules (high-level planning, low-level execution)
- External memory mechanism for information storage and retrieval
- Adaptive computation time (ACT) for variable-depth reasoning
- Configured with a fixed hierarchy of processing cycles

### Hierarchical Recurrent Execution Model (HREM)
- Multi-layer memory architecture with sparse addressing
- Hierarchical memory locations (m_loc) and dimensions (d_mem)
- Top-K sparse memory access for computational efficiency
- Configurable hierarchy levels (H_layers, L_layers) and cycles (H_cycles, L_cycles)
- Enhanced memory management with location-based addressing

## Evaluation and Optimization Framework

Our repository includes a comprehensive evaluation and optimization framework that enables scientific methodology for model comparison and hyperparameter tuning.

### Scientific Methodology

1. **Baseline Evaluation**: Establish performance baselines for HRM and HREM with default parameters
2. **Guided Hyperparameter Optimization**: Use Optuna to search for optimal hyperparameters for both models
3. **Statistical Comparison**: Run multiple evaluation runs to ensure statistically significant results
4. **Multi-objective Optimization**: Balance accuracy and parameter efficiency for fair comparisons

### Evaluation Features

- **Real-time Results**: Generate actionable insights after each iteration
- **Side-by-side Comparison**: Direct performance comparison between HRM and HREM
- **Statistical Significance**: Multiple runs with mean and standard deviation reporting
- **Parameter Efficiency**: Cost-benefit analysis considering both accuracy and model size
- **Detailed Metrics**: Accuracy, loss, steps to solve, and parameter counts

### Optimization Features

- **Smart Search Space**: Configurable hyperparameter ranges in `config/hparam_search_space.yaml`
- **Parallel Execution**: Run multiple optimization trials simultaneously
- **Persistent Studies**: SQLite backend allows stopping and resuming optimization
- **Live Monitoring**: Use optuna-dashboard to monitor progress in real-time
- **Comprehensive Reporting**: Detailed markdown reports with best parameters and comparisons

## Prerequisites ⚙️

Ensure PyTorch and CUDA are installed. The repo needs CUDA extensions to be built. If not present, run the following commands:

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Additional packages for building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

Then install FlashAttention. For Hopper GPUs, install FlashAttention 3

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

For Ampere or earlier GPUs, install FlashAttention 2

```bash
pip3 install flash-attn
```

## Install Python Dependencies 🐍

```bash
pip install -r requirements.txt
```

## Dataset Requirements 📁

Some challenges require specific datasets to be downloaded and prepared:

- **ARC Challenges**: Require the ARC-AGI and ConceptARC datasets
- **Sudoku Challenges**: Require the Sudoku dataset
- **Maze Challenges**: Require the Maze dataset

To prepare datasets, follow the instructions in the "Dataset Preparation" section below.

For quick testing, use the beginner challenges which work with synthetic data that is generated on-the-fly.

## W&B Integration 📈

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Text-based User Interface (TUI) 🖥️

For a more visual experience, this repository includes a Text-based User Interface built with Textual. The TUI provides a terminal-based graphical interface to:

- Run model evaluations
- Perform hyperparameter optimization
- Execute unit tests
- Manage datasets
- Visualize datasets

### Installation
Make sure you have Textual installed:
```bash
pip install textual
```

### Running the TUI
To start the TUI, run:
```bash
python run_tui.py
```

## Advanced Usage

### Standard Evaluation
To run a side-by-side comparison of HRM and HREM models:

```bash
python run_demo_cli.py
```

Select a challenge from the menu to begin the evaluation process.

### Hyperparameter Optimization with Optuna

For more advanced analysis, our framework includes hyperparameter optimization:

**Key Features:**

- **Flexible Search Space:** The hyperparameter search space is defined in `config/hparam_search_space.yaml`, making it easy to customize.
- **Parallel Execution:** Run multiple trials in parallel to speed up the optimization process.
- **Robust Comparison:** Runs the final comparison multiple times to provide mean and standard deviation of the performance metrics, ensuring statistical significance.
- **Detailed Reporting:** Generates a markdown report with the best hyperparameters found and the final performance comparison.
- **Persistent & Collaborative:** Uses an SQLite backend to save study progress, allowing you to stop and resume optimization.

**Usage:**

1.  **Start the optimization:**
    ```bash
    python run_demo_cli.py
    ```
    Select a challenge and let the system guide you through the optimization process.

2.  **(Optional) Monitor with Optuna Dashboard:** While the optimization is running, you can launch the Optuna dashboard in a separate terminal:
    ```bash
    optuna-dashboard sqlite:///experiments/optuna_demo_cli.db
    ```

## Notes

 - Small-sample learning typically exhibits accuracy variance of around ±2 points.
 - For Sudoku-Extreme (1,000-example dataset), late-stage overfitting may cause numerical instability during training and Q-learning. It is advisable to use early stopping once the training accuracy approaches 100%.
 - The Q-learning mechanism has been optimized to eliminate a redundant forward pass during training, significantly improving training performance.

## Citation 📜

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