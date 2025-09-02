import pytest
from cui.progress import CUIProgressHandler
from sc_engine.core.insights import ScientificInsight, InsightType

@pytest.fixture
def handler():
    return CUIProgressHandler()

def test_cui_smoke_test(handler, capsys):
    events = [
        ('start_session', {'challenge': 'Test Challenge'}),
        ('start_phase', {'phase': 'baseline_evaluation'}),
        ('start_evaluation', {'title': 'Running Baseline Evaluation'}),
        ('start_algorithm', {'algorithm': 'Test_Algorithm_1', 'progress': 0.5}),
        ('trainer:train_batch', {'metrics': {'loss': 1.234, 'acc': 0.567}, 'step': 1, 'total_steps': 10}),
        ('end_algorithm', {'algorithm': 'Test_Algorithm_1', 'metrics': {'final_loss': 0.987, 'final_acc': 0.654}}),
        ('end_evaluation', {'title': 'Running Baseline Evaluation'}),
        ('end_phase', {'phase': 'baseline_evaluation'}),
        ('start_phase', {'phase': 'optimization'}),
        ('start_optimization_alg', {'algorithm': 'Test_Algorithm_1', 'n_trials': 3}),
        ('start_trial', {'params': {'lr': 0.01, 'dropout': 0.2}}),
        ('end_trial', {'loss': 0.876}),
        ('start_trial', {'params': {'lr': 0.001, 'dropout': 0.3}}),
        ('end_trial', {'loss': 0.765}),
        ('end_optimization_alg', {'algorithm': 'Test_Algorithm_1', 'best_params': {'lr': 0.001, 'dropout': 0.3}}),
        ('skip_optimization', {'algorithm': 'Test_Algorithm_2', 'reason': 'No search space provided'}),
        ('end_phase', {'phase': 'optimization'}),
        ('start_phase', {'phase': 'insight_generation'}),
        ('insights_generated', {
            'insights': [
                ScientificInsight(
                    type=InsightType.EFFICIENCY,
                    summary="Algorithm A is better than B",
                    confidence=0.9,
                    implications=["Use A in production"],
                    discovery_potential=0.8
                )
            ],
            'report_path': '/tmp/report.md'
        }),
        ('no_insights', {}),
        ('end_phase', {'phase': 'insight_generation'}),
    ]

    for event_type, data in events:
        handler.on_progress(event_type, data)

    captured = capsys.readouterr()
    assert "Starting Scientific Discovery Session" in captured.out
    assert "Phase: Baseline Evaluation" in captured.out
    assert "Running algorithm: Test_Algorithm_1" in captured.out
    assert "Step 1/10" in captured.out
    assert "Finished algorithm: Test_Algorithm_1" in captured.out
    assert "Phase: Optimization" in captured.out
    assert "Optimizing: Test_Algorithm_1" in captured.out
    assert "Trial 1 starting" in captured.out
    assert "Trial 1 finished" in captured.out
    assert "Finished Optimizing: Test_Algorithm_1" in captured.out
    assert "Skipping optimization for Test_Algorithm_2" in captured.out
    assert "Scientific Insights Generated" in captured.out
    assert "No significant insights" in captured.out
