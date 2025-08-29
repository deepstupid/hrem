import pytest
from scientific_comparison.insight_generator import ScientificInsightGenerator, InsightType
from scientific_comparison.config import ChallengeConfig, ChallengeLevel, AlgorithmConfig
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
        hypothesis_space=["HREM is better", "HRM is better"]
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
    assert insight.evidence[0]['faster_algorithm'] == "HREM"
    assert insight.evidence[0]['speedup_factor'] == 2.5
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
    assert insight.evidence[0]['winning_algorithm'] == "HREM"
    assert insight.evidence[0]['performance_gap'] == "50.00%"

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

def test_extract_robustness_insights(mock_challenge_config, mock_algorithm_configs):
    """Test the extraction of robustness insights."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)

    comparison_results = {
        "HREM": {"all/lm_loss_runs": [0.1, 0.11, 0.09]},
        "HRM": {"all/lm_loss_runs": [0.2, 0.25, 0.15]}
    }

    insights = generator._extract_robustness_insights(comparison_results)

    assert len(insights) == 1
    insight = insights[0]
    assert insight.type == InsightType.ROBUSTNESS
    assert insight.evidence[0]['most_robust_algorithm'] == "HREM"
    assert insight.confidence < 1.0

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
    assert insights[0].evidence[0]['algorithm'] == "HREM"

def test_synthesize_meta_insights(mock_challenge_config, mock_algorithm_configs):
    """Test the synthesis of meta-insights."""
    generator = ScientificInsightGenerator(challenge=mock_challenge_config, algorithms=mock_algorithm_configs)

    comparison_results = {
        "HREM": {
            "timing": 10,
            "all/lm_loss": 0.1,
            "all/lm_loss_runs": [0.1, 0.11, 0.09],
            "loss_history": [0.5, 0.3, 0.1]
        },
        "HRM": {
            "timing": 25,
            "all/lm_loss": 0.2,
            "all/lm_loss_runs": [0.2, 0.25, 0.15],
            "loss_history": [0.6, 0.5, 0.4, 0.3, 0.2]
        }
    }

    insights = generator.extract_insights(comparison_results)

    meta_insights = [i for i in insights if i.type == InsightType.META]
    assert len(meta_insights) == 1
    assert meta_insights[0].evidence[0]['algorithm'] == "HREM"
    assert meta_insights[0].evidence[0]['number_of_wins'] >= 3
