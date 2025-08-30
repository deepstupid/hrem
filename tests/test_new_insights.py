import pytest
from sc_engine.core.insight_generator import ScientificInsightGenerator
from sc_engine.core.config import ChallengeConfig, AlgorithmConfig, ChallengeLevel
from sc_engine.core.insights import InsightType

@pytest.fixture
def three_alg_challenge_config():
    """Fixture for a mock ChallengeConfig with three algorithms."""
    return ChallengeConfig(
        name="3-way Test Challenge",
        id="3_way_test",
        description="A test challenge for three algorithms.",
        difficulty=ChallengeLevel.ADVANCED,
        hypothesis_space=[],
        dataset={},
        scientific_question="How do three algorithms compare?"
    )

@pytest.fixture
def three_alg_configs():
    """Fixture for three mock AlgorithmConfigs."""
    return [
        AlgorithmConfig(
            name="Algo-Fast",
            complexity=5,
            algorithm_class="Fast_class",
            search_space={},
            theoretical_advantages=[],
            theoretical_limitations=[],
            complexity_profile={"params": 3e6, "flops": 1e9},
            config={}
        ),
        AlgorithmConfig(
            name="Algo-Accurate",
            complexity=15,
            algorithm_class="Accurate_class",
            search_space={},
            theoretical_advantages=[],
            theoretical_limitations=[],
            complexity_profile={"params": 5e6, "flops": 10e9},
            config={}
        ),
        AlgorithmConfig(
            name="Algo-Balanced",
            complexity=10,
            algorithm_class="Balanced_class",
            search_space={},
            theoretical_advantages=[],
            theoretical_limitations=[],
            complexity_profile={"params": 3e6, "flops": 5e9},
            config={}
        ),
    ]

class TestNewInsightFeatures:
    def test_pareto_front_analysis(self, three_alg_challenge_config, three_alg_configs):
        """Test the Pareto front analysis for meta-insights."""
        generator = ScientificInsightGenerator(
            challenge=three_alg_challenge_config,
            algorithms=three_alg_configs,
        )

        # Algo-Fast is fastest, Algo-Accurate has lowest loss, Algo-Balanced is in between.
        # All three should be on the Pareto front.
        results = {
            "Algo-Fast":     {"all/lm_loss": 0.3, "timing": 50},
            "Algo-Accurate": {"all/lm_loss": 0.1, "timing": 150},
            "Algo-Balanced": {"all/lm_loss": 0.2, "timing": 100},
        }

        insights = generator.extract_insights(results)

        meta_insights = [i for i in insights if i.type == InsightType.META and "Pareto Front" in i.title]
        assert len(meta_insights) == 1

        pareto_insight = meta_insights[0]
        assert "revealing a trade-off" in pareto_insight.summary

        pareto_points = [e.metric_value for e in pareto_insight.evidence if e.metric_name == 'pareto_optimal_point']
        assert len(pareto_points) == 3
        assert "Algo-Fast" in pareto_points
        assert "Algo-Accurate" in pareto_points
        assert "Algo-Balanced" in pareto_points

    def test_complexity_profile_attribution(self, three_alg_challenge_config, three_alg_configs):
        """Test the causal attribution from complexity profiles."""
        generator = ScientificInsightGenerator(
            challenge=three_alg_challenge_config,
            algorithms=three_alg_configs,
        )

        results = {
            "Algo-Fast":     {"timing": 50},
            "Algo-Balanced": {"timing": 100},
        }

        insights = generator.extract_insights(results)

        efficiency_insights = [i for i in insights if i.type == InsightType.EFFICIENCY]
        assert len(efficiency_insights) == 1

        insight = efficiency_insights[0]
        assert insight.causal_attribution is not None
        assert "lower flops" in insight.causal_attribution
        assert "1000000000.0 vs. 5000000000.0" in insight.causal_attribution

    def test_automated_hypothesis_generation(self, three_alg_challenge_config, three_alg_configs):
        """Test the automated generation of new hypotheses."""
        # Make Algo-Fast (less complex) outperform Algo-Accurate (more complex)
        three_alg_configs[0].complexity = 5
        three_alg_configs[1].complexity = 10

        generator = ScientificInsightGenerator(
            challenge=three_alg_challenge_config,
            algorithms=three_alg_configs,
        )

        # A surprising result: the MORE complex model is more efficient.
        results = {
            "Algo-Fast":     {"timing": 150},
            "Algo-Accurate": {"timing": 50},
        }

        insights = generator.extract_insights(results)

        generated_hypotheses = [i for i in insights if i.type == InsightType.GENERATED_HYPOTHESIS]
        assert len(generated_hypotheses) > 0

        # Check for the specific hypothesis related to surprising performance
        surprising_insight = next((h for h in generated_hypotheses if "Surprising Performance" in h.title), None)
        assert surprising_insight is not None
        assert "may not accurately reflect its practical performance" in surprising_insight.summary
