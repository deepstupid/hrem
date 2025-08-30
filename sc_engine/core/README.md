# Scientific Algorithm Comparison Engine

This directory contains the implementation of a scientifically-rigorous framework for real-time comparison of algorithms (HRM vs HREM) that balances computational exploration with user patience constraints to maximize potential for scientific discovery within practical time limits.

## Overview

The Scientific Comparison Engine provides a challenge-centric approach to algorithm evaluation, focusing on generating actionable scientific insights rather than just performance metrics. It implements adaptive exploration that responds dynamically to performance signals and optimizes for discovery potential within user patience constraints.

## Key Components

### 1. ScientificDiscoveryEngine
The central orchestrator that manages the scientific exploration process, executing baseline evaluation, hyperparameter optimization, and final evaluation phases.

### 2. AdaptivePatienceManager
Intelligently allocates user patience across exploration activities, dynamically adjusting time allocation based on discovery potential.

### 3. DiscoveryAwareScheduler
Prioritizes exploration activities based on scientific discovery potential and adjusts exploration depth based on performance signals.

### 4. ScientificInsightGenerator
Extracts meaningful scientific insights from algorithm comparisons, classifying them by type (efficiency, robustness, scalability, etc.) and discovery potential.

### 5. ScientificTimingManager
Extends the existing timing utilities with scientific discovery metrics, tracking insights generated per unit time.

## Usage

### CLI Command
```bash
# Run a scientific comparison with default settings
python run.py compare

# Run a specific challenge with custom patience level
python run.py compare --challenge long_range_dependencies --patience high

# Run in smoke test mode for quick validation
python run.py compare --smoke-test --patience low
```

### Python API
```python
from scientific_comparison.model_runner import ScientificModelRunner

runner = ScientificModelRunner()
results = runner.run_comparison_from_config(
    challenge_id="long_range_dependencies",
    patience_level="medium"
)
```

## Scientific Challenges

The engine comes with several predefined scientific challenges:

1. **Long-range Dependency Modeling** - Evaluates how HRM and HREM handle long-range dependencies
2. **Sequence Duplication Challenge** - Tests pattern duplication capabilities
3. **Parity Computation Challenge** - Examines global information processing

## Features

- **Challenge-Centric Exploration**: Focuses exploration on challenge-specific characteristics
- **Algorithm-Driven Methodology**: Leverages algorithm-specific strengths in exploration
- **Patience-Aware Optimization**: Continuously monitors patience consumption and predicts exploration value
- **Scientific Discovery Pipeline**: Generates hypotheses, validates with statistical tests, and classifies insights by potential

## Implementation Details

The engine follows the specification in `README.new.md` and integrates with the existing HRM/HREM infrastructure while adding scientific rigor to the comparison process.