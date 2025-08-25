## Parameterized CLI Demo

The parameterized CLI demo (`run_demo_cli_parameterized.py`) allows you to run the HRM vs HREM demonstration with customizable parameters:

### Usage

```bash
# Run with interactive challenge selection and default models (HRM, HREM)
python run_demo_cli_parameterized.py

# Run with specific challenge and models
python run_demo_cli_parameterized.py --challenge-key "ARC Challenge" --models HRM HREM EnhancedHREM

# Run in fast mode for quicker testing
python run_demo_cli_parameterized.py --fast --challenge-key "ARC Challenge"

# Run with interactive prompts
python run_demo_cli_parameterized.py --interactive --challenge-key "ARC Challenge"
```

### Supported Models

- `HRM`: Traditional recurrent model with external memory
- `HREM`: Hierarchical approach with multiple memory layers
- `EnhancedHREM`: Advanced version of HREM with additional features

### Parameters

- `--challenge-key`: Specify the challenge to run directly (skips menu)
- `--models`: List of models to compare (default: HRM HREM)
- `--fast`: Run in fast mode for quicker testing
- `--interactive`: Enable interactive mode with user prompts

The script will optimize all specified models and display comprehensive results including:
- Baseline performance metrics
- Optimized parameters for each model
- Performance improvements
- Cost-benefit analysis
- Final comparison of all models