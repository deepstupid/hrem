import pytest
from pathlib import Path
from hrm_system import ExperimentConfig, run_optimization

@pytest.fixture
def smoke_test_config():
    """Provides a smoke test configuration for optimization mode."""
    # Using an in-memory sqlite DB for the test to avoid file creation
    storage_url = "sqlite:///:memory:"

    return ExperimentConfig(
        mode="optimize",
        run_config={
            "smoke_test": True,
            "study_name": "test_optimization_smoke"
        },
        data_config={
            "dataset": "synthetic"
        },
        optimization_config={
            "n_trials": 2, # Keep it short for a test
            "n_final_runs": 1,
            "storage": storage_url
        }
    )

def test_run_optimization_smoke(smoke_test_config):
    """
    An integration test that runs the full optimization pipeline in smoke_test mode.
    """
    logs = []
    def logger_callback(message):
        logs.append(message)

    smoke_test_config.run_config.logger_callback = logger_callback

    # Run the optimization
    results = run_optimization(smoke_test_config)

    # 1. Check the results dictionary
    assert "best_trial" in results
    assert "best_params" in results
    assert "best_value" in results
    assert "results" in results
    assert "report_path" in results

    # Check that we have results for both models in the final comparison
    assert "HRM" in results["results"]
    assert "HREM_best" in results["results"]
    assert results["results"]["HRM"]["all/lm_loss"] != "N/A"

    # 2. Check that the report file was created
    report_path = Path(results["report_path"])
    assert report_path.exists()
    assert report_path.is_file()

    # 3. Check that the results CSV was created
    study_name = smoke_test_config.run_config.study_name
    output_dir = smoke_test_config.run_config.output_dir
    csv_path = Path(output_dir) / study_name / "optimization_results.csv"
    assert csv_path.exists()
    assert csv_path.is_file()

    # 4. Check logs
    assert len(logs) > 0
    assert any("--- Starting Optimization" in log for log in logs)
    assert any("Optimization finished" in log for log in logs)
    assert any("--- Running Final Comparison" in log for log in logs)
    assert any("Generated optimization report" in log for log in logs)

    # Clean up created files
    report_path.unlink()
    csv_path.unlink()
    if report_path.parent.exists():
        try:
            report_path.parent.rmdir()
        except OSError:
            pass
