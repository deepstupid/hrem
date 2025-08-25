#!/usr/bin/env python3
"""
Script to run enhanced HREM training with improved performance.
"""

import click
from rich.console import Console

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    OptimizationConfig,
    HREMParams,
    run_evaluation,
)

console = Console()

def logger_callback(message: str):
    """A simple logger callback that prints to the console."""
    console.print(message)

@click.command()
@click.option("--dataset", default="arc", type=click.Choice(["arc", "sudoku", "maze", "synthetic"]), help="Dataset to use.")
@click.option("--num-aug", default=100, type=int, help="Number of augmentations.")
@click.option("--n-runs", default=3, type=int, help="Number of runs for statistical significance.")
@click.option("--smoke-test", is_flag=True, default=False, help="Run in smoke test mode.")
@click.option("--study-name", default="enhanced_hrem_evaluation", type=str, help="Name for the study.")
@click.option("--compare", is_flag=True, default=False, help="Compare HRM, HREM, and EnhancedHREM.")
def evaluate(dataset, num_aug, n_runs, smoke_test, study_name, compare):
    """Run an evaluation of the enhanced HREM model."""
    console.print(f"[bold blue]Starting Enhanced HREM Evaluation: {study_name}[/bold blue]")

    # Create evaluation config
    if compare:
        # Compare all three models
        eval_config = EvaluationConfig(
            n_runs=n_runs,
            model_a=ModelConfig(
                name="HRM", 
                algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm", 
                base_arch_config="hrm_v1"
            ),
            model_b=ModelConfig(
                name="HREM", 
                algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm", 
                base_arch_config="hrem_v1"
            ),
            model_c=ModelConfig(
                name="EnhancedHREM", 
                algorithm_class="hrm_system.algorithms.enhanced_hrem.EnhancedHREMAlgorithm", 
                base_arch_config="enhanced_hrem_v1"
            )
        )
    else:
        # Just run enhanced HREM
        eval_config = EvaluationConfig(
            n_runs=n_runs,
            model_a=ModelConfig(
                name="HREM", 
                algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm", 
                base_arch_config="hrem_v1"
            ),
            model_b=ModelConfig(
                name="EnhancedHREM", 
                algorithm_class="hrm_system.algorithms.enhanced_hrem.EnhancedHREMAlgorithm", 
                base_arch_config="enhanced_hrem_v1"
            )
        )

    config = ExperimentConfig(
        mode="evaluate",
        run_config=RunConfig(
            smoke_test=smoke_test,
            study_name=study_name,
            logger_callback=logger_callback
        ),
        data_config=DataConfig(
            dataset=dataset,
            num_aug=num_aug
        ),
        evaluation_config=eval_config
    )

    run_evaluation(config)
    console.print(f"[bold green]Enhanced HREM Evaluation '{study_name}' finished.[/bold green]")

if __name__ == "__main__":
    evaluate()