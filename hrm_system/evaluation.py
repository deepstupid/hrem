from typing import Dict, Any, List

from .config import ExperimentConfig
from .runner import run_single_model
from .reporting import aggregate_metrics, generate_evaluation_report

def run_evaluation(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Runs a side-by-side comparison of two models.
    """
    if config.mode != "evaluate":
        raise ValueError("ExperimentConfig must be in 'evaluate' mode.")

    run_config = config.run_config
    data_config = config.data_config
    training_config = config.training_config
    eval_config = config.evaluation_config

    logger = run_config.logger_callback or print

    logger(f"--- Starting Evaluation: {run_config.study_name} ---")

    all_metrics = {}
    # Check if we have a third model to evaluate
    models_to_run = [eval_config.model_a, eval_config.model_b]
    if hasattr(eval_config, 'model_c') and eval_config.model_c:
        models_to_run.append(eval_config.model_c)

    for model_config in models_to_run:
        logger(f"\n[bold blue]--- Running Model: {model_config.name} ---[/bold blue]")

        if eval_config.n_runs > 1:
            logger(f"Running {eval_config.n_runs} times for statistical significance...")

        metrics_list: List[Dict[str, Any]] = []
        for i in range(eval_config.n_runs):
            run_identifier = f"run_{i}"

            # Create a deep copy of configs for each run to avoid side effects
            run_config_copy = run_config.model_copy(deep=True)
            data_config_copy = data_config.model_copy(deep=True)
            model_config_copy = model_config.model_copy(deep=True)
            training_config_copy = training_config.model_copy(deep=True)

            metrics = run_single_model(
                run_config=run_config_copy,
                data_config=data_config_copy,
                model_config=model_config_copy,
                training_config=training_config_copy,
                run_identifier=run_identifier
            )
            metrics_list.append(metrics)

        # Aggregate the metrics from all runs for this model
        aggregated = aggregate_metrics(metrics_list)
        all_metrics[model_config.name] = aggregated
        logger(f"Aggregated metrics for {model_config.name}: {aggregated}")

    # Generate the final report
    logger("\n[bold blue]--- Generating Comparison Report ---[/bold blue]")
    report_path = generate_evaluation_report(
        all_metrics=all_metrics,
        eval_config=eval_config,
        run_config=run_config
    )
    logger(f"Generated comparison report: {report_path}")

    return {
        "results": all_metrics,
        "report_path": report_path
    }
