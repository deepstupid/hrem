import pytest
from pathlib import Path
from hrm_system import ExperimentConfig, run_evaluation

@pytest.fixture
def smoke_test_config():
    """Provides a smoke test configuration for evaluation mode."""
    return ExperimentConfig(
        mode="evaluate",
        run_config={
            "smoke_test": True,
            "study_name": "test_evaluation_smoke"
        },
        data_config={
            "dataset": "synthetic"
        },
        evaluation_config={
            "model_a": {
                "name": "HRM",
                "algorithm_class": "hrm_system.algorithms.hrm.HRMAlgorithm",
                "base_arch_config": "hrm_v1"
            },
            "model_b": {
                "name": "HREM",
                "algorithm_class": "hrm_system.algorithms.hrem.HREMAlgorithm",
                "base_arch_config": "hrem_v1"
            }
        }
    )

def test_run_evaluation_smoke(smoke_test_config):
    """
    An integration test that runs the full evaluation pipeline in smoke_test mode.
    This will build a small dataset and train two models for a single epoch.
    """
    logs = []
    def logger_callback(message):
        logs.append(message)

    smoke_test_config.run_config.logger_callback = logger_callback

    # Run the evaluation
    results = run_evaluation(smoke_test_config)

    # 1. Check the results dictionary
    assert "results" in results
    assert "report_path" in results

    # Check that we have results for both models
    assert "HRM" in results["results"]
    assert "HREM" in results["results"]

    # Check that the metrics look reasonable (not empty)
    assert "all/lm_loss" in results["results"]["HRM"]
    assert results["results"]["HRM"]["all/lm_loss"] != "N/A"

    # 2. Check that the report file was created
    report_path = Path(results["report_path"])
    assert report_path.exists()
    assert report_path.is_file()

    # 3. Check that the logs were produced
    assert len(logs) > 0
    assert any("--- Starting Evaluation" in log for log in logs)
    assert any("Running Model: HRM" in log for log in logs)
    assert any("Running Model: HREM" in log for log in logs)
    assert any("Generated comparison report" in log for log in logs)

    # Clean up the created report
    report_path.unlink()
    if report_path.parent.exists():
        # Attempt to remove the directory if it's empty
        try:
            report_path.parent.rmdir()
        except OSError:
            # It's not empty, which is fine. The tmp results files are there.
            pass
