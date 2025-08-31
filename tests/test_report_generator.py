import unittest
import os
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from sc_engine.core.report_generator import ScientificReportGenerator
from sc_engine.core.insights import ScientificInsight, InsightType, Evidence

class TestScientificReportGenerator(unittest.TestCase):

    def setUp(self):
        """Set up the test environment."""
        self.challenge_name = "Test Report Challenge"
        self.algorithm_names = ["Model_A", "Model_B"]
        self.final_results = {
            "Model_A_optimized": {
                "all": {"accuracy": 0.95, "lm_loss": 0.15, "steps": 100}
            },
            "Model_B_optimized": {
                "all": {"accuracy": 0.92, "lm_loss": 0.18, "steps": 120}
            }
        }

        # Create a dummy plot file
        self.plot_dir = "plots_test"
        os.makedirs(self.plot_dir, exist_ok=True)
        self.plot_path = os.path.join(self.plot_dir, "test_plot.png")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Test Plot")
        fig.savefig(self.plot_path)
        plt.close(fig)

        self.insights = [
            ScientificInsight(
                type=InsightType.SCALABILITY,
                title="Scalability Insight",
                summary="Model_A scales better than Model_B.",
                implications=["Model_A is a better choice for large-scale tasks."],
                confidence=0.9,
                discovery_potential=0.8,
                evidence=[Evidence(metric_name="performance_gap", metric_value="15%")]
            )
        ]

    def tearDown(self):
        """Clean up the test environment."""
        if os.path.exists(self.plot_path):
            os.remove(self.plot_path)
        if os.path.exists(self.plot_dir):
            os.rmdir(self.plot_dir)

    def test_generate_report_with_all_sections(self):
        """
        Test that the report generator correctly includes the header, summary table, plot, and insights.
        """
        # Arrange
        report_generator = ScientificReportGenerator(
            challenge_name=self.challenge_name,
            algorithm_names=self.algorithm_names,
            final_results=self.final_results,
            plot_path=self.plot_path
        )

        # Act
        report = report_generator.generate_report(self.insights)

        # Assert
        # Check for header
        self.assertIn(f"# Scientific Comparison Report: {self.challenge_name}", report)
        self.assertIn("`Model_A`, `Model_B`", report)

        # Check for summary table
        self.assertIn("## 📈 Final Metrics Summary", report)
        self.assertIn("Model_A (Optimized)", report)
        self.assertIn("0.9500", report) # accuracy for Model_A
        self.assertIn("0.1800", report) # lm_loss for Model_B

        # Check for plot
        self.assertIn("## 📊 Performance Plot", report)
        # Use os.path.normpath to handle different OS path separators
        self.assertIn(f"![Performance Plot]({os.path.normpath(self.plot_path)})", report)

        # Check for insights
        self.assertIn("## 📊 Detailed Findings", report)
        self.assertIn("Scalability Insight", report)

    def test_generate_report_without_plot_or_results(self):
        """
        Test that the report generates gracefully when no plot or final results are provided.
        """
        # Arrange
        report_generator = ScientificReportGenerator(
            challenge_name=self.challenge_name,
            algorithm_names=self.algorithm_names
        )

        # Act
        report = report_generator.generate_report(self.insights)

        # Assert
        self.assertIn(f"# Scientific Comparison Report: {self.challenge_name}", report)
        self.assertNotIn("## 📈 Final Metrics Summary", report)
        self.assertNotIn("## 📊 Performance Plot", report)
        self.assertIn("Scalability Insight", report)

if __name__ == "__main__":
    unittest.main()
