import pytest
from sc_engine.core.insight_generator import ScientificInsightGenerator
from sc_engine.core.report_generator import ScientificReportGenerator
from sc_engine.core.config import ChallengeConfig, AlgorithmConfig, ChallengeLevel
from sc_engine.core.insights import InsightType

from hrm_system.config import DataConfig

from sc_engine.core.config import Hypothesis

@pytest.fixture
def mock_challenge_config():
    """Fixture for a mock ChallengeConfig."""
    return ChallengeConfig(
        name="Test Challenge",
        id="test_challenge",
        description="A test challenge.",
        difficulty=ChallengeLevel.INTERMEDIATE,
        hypothesis_space=[
            Hypothesis(description="HREM will be more efficient than HRM", metric="efficiency", expected_winner="HREM", expected_loser="HRM"),
            Hypothesis(description="HRM will be more robust than HREM", metric="robustness", expected_winner="HRM", expected_loser="HREM"),
        ],
        dataset=DataConfig(),
        scientific_question="Which model is better?"
    )

@pytest.fixture
def mock_algorithm_configs():
    """Fixture for mock AlgorithmConfigs."""
    return [
        AlgorithmConfig(name="HREM", complexity=10, theoretical_advantages=["memory"], theoretical_limitations=["overhead"], algorithm_class="HREM_class", search_space={}),
        AlgorithmConfig(name="HRM", complexity=5, theoretical_advantages=["speed"], theoretical_limitations=["memory_capacity"], algorithm_class="HRM_class", search_space={}),
    ]

class TestScientificInsightGenerator:
    def test_initialization(self, mock_challenge_config, mock_algorithm_configs):
        """Test that the insight generator initializes correctly."""
        generator = ScientificInsightGenerator(
            challenge=mock_challenge_config,
            algorithms=mock_algorithm_configs,
            config_path="config/insight_config.yaml"
        )
        assert generator is not None
        assert generator.challenge.name == "Test Challenge"
        assert "HREM" in generator.algorithms
        assert "HRM" in generator.algorithms
        assert "efficiency_speedup_threshold" in generator.config

    def test_no_insights_on_empty_results(self, mock_challenge_config, mock_algorithm_configs):
        """Test that no insights are generated from empty results."""
        generator = ScientificInsightGenerator(
            challenge=mock_challenge_config,
            algorithms=mock_algorithm_configs,
        )
        insights = generator.extract_insights({})
        # We expect hypothesis and failure insights even with empty results
        assert len(insights) > 0
        assert all(i.type in [InsightType.HYPOTHESIS, InsightType.FAILURE] for i in insights)

    def test_efficiency_insight(self, mock_challenge_config, mock_algorithm_configs):
        """Test the generation of an efficiency insight."""
        generator = ScientificInsightGenerator(
            challenge=mock_challenge_config,
            algorithms=mock_algorithm_configs,
        )
        results = {
            "HREM": {"timing": 100},
            "HRM": {"timing": 50},
        }
        insights = generator.extract_insights(results)
        efficiency_insights = [i for i in insights if i.type == InsightType.EFFICIENCY]
        assert len(efficiency_insights) == 1
        insight = efficiency_insights[0]
        assert "HRM" in insight.summary
        assert "2.00x speedup" in insight.implications[1] # The speedup text is in the second implication
        assert insight.causal_attribution is not None

    def test_scalability_insight(self, mock_challenge_config, mock_algorithm_configs):
        """Test the generation of a scalability insight."""
        generator = ScientificInsightGenerator(
            challenge=mock_challenge_config,
            algorithms=mock_algorithm_configs,
        )
        results = {
            "HREM": {"all/lm_loss": 0.1},
            "HRM": {"all/lm_loss": 0.2},
        }
        insights = generator.extract_insights(results)
        scalability_insights = [i for i in insights if i.type == InsightType.SCALABILITY]
        assert len(scalability_insights) == 1
        insight = scalability_insights[0]
        assert "HREM outperforms HRM" in insight.summary
        assert "50.00%" in insight.evidence[0].metric_value
        assert insight.causal_attribution is not None
        assert insight.recommendations is not None

    def test_failure_insight_nan_loss(self, mock_challenge_config, mock_algorithm_configs):
        """Test the generation of a failure insight for NaN loss."""
        generator = ScientificInsightGenerator(
            challenge=mock_challenge_config,
            algorithms=mock_algorithm_configs,
        )
        results = {
            "HREM": {"all/lm_loss": float('nan')},
            "HRM": {"all/lm_loss": 0.2},
        }
        insights = generator.extract_insights(results)
        failure_insights = [i for i in insights if i.type == InsightType.FAILURE]
        assert len(failure_insights) == 1
        assert "Training Failure: HREM" in failure_insights[0].title

    def test_hypothesis_testing(self, mock_challenge_config, mock_algorithm_configs):
        """Test the hypothesis testing functionality."""
        generator = ScientificInsightGenerator(
            challenge=mock_challenge_config,
            algorithms=mock_algorithm_configs,
        )
        results = {
            "HREM": {"timing": 100},
            "HRM": {"timing": 50},
        }
        insights = generator.extract_insights(results)
        hypothesis_insights = [i for i in insights if i.type == InsightType.HYPOTHESIS]
        assert len(hypothesis_insights) == 2

        # Note: This test is brittle due to the simple implementation of hypothesis testing.
        h1 = next(i for i in hypothesis_insights if "efficient" in i.hypothesis)
        assert "Refuted" in h1.title # HRM was more efficient, so HREM hypothesis is refuted

class TestScientificReportGenerator:
    def test_report_generation(self, mock_challenge_config, mock_algorithm_configs):
        """Test that the report generator runs without errors."""
        insight_generator = ScientificInsightGenerator(
            challenge=mock_challenge_config,
            algorithms=mock_algorithm_configs,
        )
        results = {
            "HREM": {"timing": 100, "all/lm_loss": 0.1},
            "HRM": {"timing": 50, "all/lm_loss": 0.2},
        }
        insights = insight_generator.extract_insights(results)

        report_generator = ScientificReportGenerator(
            challenge_name=mock_challenge_config.name,
            algorithm_names=[alg.name for alg in mock_algorithm_configs],
        )
        report = report_generator.generate_report(insights)

        assert "Scientific Comparison Report" in report
        assert "Executive Summary" in report
        assert "Hypothesis Testing" in report
        assert "Detailed Findings" in report
        assert "HREM" in report
        assert "HRM" in report
        assert "🚀" in report # Check for emoji
