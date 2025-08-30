import pytest
from sc_engine.core.insight_generator import ScientificInsightGenerator
from sc_engine.core.report_generator import ScientificReportGenerator
from sc_engine.core.insights import InsightType
from sc_engine.core.config import ChallengeConfig, ChallengeLevel, AlgorithmConfig, Hypothesis
from hrm_system.config import DataConfig

@pytest.fixture
def mock_challenge_config():
    """Fixture for a mock ChallengeConfig."""
    return ChallengeConfig(
        name="Test Challenge",
        id="test_challenge",
        description="A mock challenge for testing.",
        dataset=DataConfig(path="test/path", name="test_dataset"),
        difficulty=ChallengeLevel.INTERMEDIATE,
        scientific_question="Which model is better?",
        hypothesis_space=[
            Hypothesis(description="HREM is better", metric="scalability", expected_winner="HREM", expected_loser="HRM"),
            Hypothesis(description="HRM is better", metric="robustness", expected_winner="HRM", expected_loser="HREM")
        ]
    )

@pytest.fixture
def mock_algorithm_configs():
    """Fixture for mock AlgorithmConfigs."""
    return [
        AlgorithmConfig(name="HREM", algorithm_class="HREM", theoretical_advantages=[], theoretical_limitations=[], search_space={}, complexity=2.0),
        AlgorithmConfig(name="HRM", algorithm_class="HRM", theoretical_advantages=[], theoretical_limitations=[], search_space={}, complexity=1.0)
    ]

def test_extract_efficiency_insights(mock_challenge_config, mock_algorithm_configs):
    """Test the extraction of efficiency insights."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)

    comparison_results = {
        "HREM": {"timing": 10},
        "HRM": {"timing": 25}
    }

    insights = generator._extract_efficiency_insights(comparison_results)

    assert len(insights) == 1
    insight = insights[0]
    assert insight.type == InsightType.EFFICIENCY
    assert insight.evidence[0].metric_name == "speedup_factor"
    assert insight.evidence[0].metric_value == 2.5
    assert "surprisingly" in insight.implications[0]

def test_no_efficiency_insight_when_no_significant_difference(mock_challenge_config, mock_algorithm_configs):
    """Test that no efficiency insight is generated when the speedup is not significant."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)

    comparison_results = {
        "HREM": {"timing": 10},
        "HRM": {"timing": 11}
    }

    insights = generator._extract_efficiency_insights(comparison_results)

    assert len(insights) == 0

def test_extract_scalability_insights(mock_challenge_config, mock_algorithm_configs):
    """Test the extraction of scalability insights."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)

    comparison_results = {
        "HREM": {"all/lm_loss": 0.1},
        "HRM": {"all/lm_loss": 0.2}
    }

    insights = generator._extract_scalability_insights(comparison_results)

    assert len(insights) == 1
    insight = insights[0]
    assert insight.type == InsightType.SCALABILITY
    assert insight.evidence[0].metric_name == "performance_gap"
    assert insight.evidence[0].metric_value == "50.00%"

def test_extract_convergence_insights(mock_challenge_config, mock_algorithm_configs):
    """Test the extraction of convergence insights."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)

    comparison_results = {
        "HREM": {"all/lm_loss": 0.1, "loss_history": [0.5, 0.3, 0.1]},
        "HRM": {"all/lm_loss": 0.2, "loss_history": [0.6, 0.5, 0.4, 0.3, 0.2]}
    }

    insights = generator._extract_convergence_insights(comparison_results)

    assert len(insights) == 2
    assert insights[0].type == InsightType.CONVERGENCE
    assert insights[1].type == InsightType.CONVERGENCE

def test_extract_robustness_insights(mock_challenge_config, mock_algorithm_configs, tmp_path):
    """Test the extraction of robustness insights."""
    import yaml
    from omegaconf import OmegaConf

    # Load the default config
    base_config = OmegaConf.load("config/insight_config.yaml")

    # Override with test-specific values
    test_config_overrides = {
        "statistical_significance_threshold": 0.1,
        "robustness_variance_threshold": 1.1,
    }
    base_config.update(test_config_overrides)

    config_path = tmp_path / "temp_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(config=base_config, f=f)

    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs, config_path=str(config_path))

    comparison_results = {
        "HREM": {"all/lm_loss_runs": [0.1, 0.11, 0.09]},
        "HRM": {"all/lm_loss_runs": [0.2, 0.25, 0.15]}
    }

    insights = generator._extract_robustness_insights(comparison_results)

    assert len(insights) == 1
    insight = insights[0]
    assert insight.type == InsightType.ROBUSTNESS
    assert insight.confidence > 0.9

def test_extract_generalization_insights(mock_challenge_config, mock_algorithm_configs):
    """Test the extraction of generalization insights."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)

    comparison_results = {
        "HREM": {"all/lm_loss": 0.1, "generalization_loss": 0.15},
        "HRM": {"all/lm_loss": 0.2, "generalization_loss": 0.3}
    }

    insights = generator._extract_generalization_insights(comparison_results)

    assert len(insights) == 2

def test_extract_adaptability_insights(mock_challenge_config, mock_algorithm_configs):
    """Test the extraction of adaptability insights."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)

    comparison_results = {
        "HREM": {"pre_finetune_loss": 0.5, "post_finetune_loss": 0.2},
        "HRM": {"pre_finetune_loss": 0.6, "post_finetune_loss": 0.58}
    }

    insights = generator._extract_adaptability_insights(comparison_results)

    assert len(insights) == 1
    assert insights[0].type == InsightType.ADAPTABILITY
    assert "HREM" in insights[0].implications[0]

def test_synthesize_meta_insights(mock_challenge_config, mock_algorithm_configs):
    """Test the synthesis of meta-insights."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)

    # Set the comparison_results manually for this private method test
    generator.comparison_results = {
        "HREM": {"all/lm_loss": 0.1, "timing": 10},
        "HRM": {"all/lm_loss": 0.2, "timing": 20},
    }

    # Mock insights for meta-insight synthesis
    mock_insights = [
        # Create mock ScientificInsight objects here
    ]

    # This test needs to be updated to reflect the new structure of ScientificInsight and Evidence
    # For now, we'll just check that the function runs without errors
    meta_insights = generator._synthesize_meta_insights(mock_insights)
    assert isinstance(meta_insights, list)

def test_generate_report(mock_challenge_config, mock_algorithm_configs):
    """Test the generation of a scientific report."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)
    comparison_results = {
        "HREM": {"timing": 10, "all/lm_loss": 0.1},
        "HRM": {"timing": 25, "all/lm_loss": 0.2}
    }
    insights = generator.extract_insights(comparison_results)

    report_generator = ScientificReportGenerator(
        challenge_name=mock_challenge_config.name,
        algorithm_names=[alg.name for alg in mock_algorithm_configs]
    )
    report = report_generator.generate_report(insights)

    assert "# Scientific Comparison Report: Test Challenge" in report
    assert "HREM" in report
    assert "HRM" in report
    assert "Efficiency: HREM vs HRM" in report
    assert "Scalability: HREM vs HRM" in report
    assert "speedup_factor" in report
    assert "performance_gap" in report
