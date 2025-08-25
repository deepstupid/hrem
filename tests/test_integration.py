import pytest
from hrm_system.config import ExperimentConfig, RunConfig, DataConfig, EvaluationConfig
from hrm_system.evaluation import run_evaluation

def test_hrm_vs_hrem_integration():
    """
    An integration test that runs a side-by-side comparison of HRM and HREM
    for a few runs on a synthetic dataset.
    """
    config = ExperimentConfig(
        mode="evaluate",
        run_config=RunConfig(
            smoke_test=True,
            study_name="integration_test",
            output_dir="tests/test_output",
        ),
        data_config=DataConfig(
            dataset="synthetic",
        ),
        evaluation_config=EvaluationConfig(
            n_runs=1,
        ),
    )

    # The run_evaluation function will raise an exception if anything goes wrong.
    # The test will pass if the function completes successfully.
    run_evaluation(config)
