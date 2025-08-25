# HRM System TUI

The HRM System Text-based User Interface (TUI) provides an intuitive way to access all functionality of the codebase through a terminal-based graphical interface.

## Features

The TUI includes the following tabs:

1. **Demo**: Turnkey demonstration of HRM vs HREM with hyperparameter optimization
2. **Evaluation**: Run side-by-side comparisons of HRM and HREM models
3. **Optimization**: Run hyperparameter optimization for the HREM model using Optuna
4. **Testing**: Run unit tests for the HRM System
5. **Dataset Management**: Build and manage datasets for training and evaluation
6. **Dataset Visualizer**: Visualize datasets (work in progress)

## Installation

Make sure you have Textual installed:

```bash
pip install textual
```

## Running the TUI

To start the TUI, run:

```bash
python -m tui.app
```

## Usage

Navigate between tabs using the tab bar at the top or using Ctrl+Left/Right arrows.
Within each tab, you can configure options and run the corresponding functionality.

### Demo Tab
- Click "Start Demo" to begin a turnkey demonstration
- The demo will show:
  1. A baseline evaluation of HRM vs HREM
  2. Hyperparameter optimization for HREM with real-time feedback
  3. A final comparison with the optimized HREM
- Use the Pause/Resume button to control the demo execution
- No configuration parameters needed - it's all automatic!
- **Unique Feature**: Real-time results generation after each iteration

### Evaluation Tab
- Select dataset type (Synthetic, ARC, Sudoku, Maze)
- Set number of runs for statistical significance
- Enable smoke test mode for quick testing
- Click "Run Evaluation" to start

### Optimization Tab
- Select dataset type
- Set number of trials for hyperparameter optimization
- Set number of parallel jobs
- Enable smoke test mode for quick testing
- Click "Run Optimization" to start

### Testing Tab
- Select which tests to run (all, specific file, or smoke tests only)
- Click "Run Selected Tests" or "Run All Tests"

### Dataset Management Tab
- Select dataset type to build
- Set output directory and number of samples
- Click the corresponding dataset button to start building