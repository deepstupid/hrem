"""Generate a scientific report from a list of insights."""

from typing import List, Dict, Any
from .insights import ScientificInsight, Evidence

class ScientificReportGenerator:
    """Generates a Markdown report from a list of scientific insights."""

    def __init__(self, challenge_name: str, algorithm_names: List[str]):
        """
        Initialize the report generator.

        Args:
            challenge_name: The name of the scientific challenge.
            algorithm_names: A list of the algorithm names that were compared.
        """
        self.challenge_name = challenge_name
        self.algorithm_names = algorithm_names

    def generate_report(self, insights: List[ScientificInsight]) -> str:
        """
        Generate a Markdown report from a list of insights.

        Args:
            insights: A list of ScientificInsight objects.

        Returns:
            A string containing the Markdown report.
        """
        if not insights:
            return "## No significant insights were generated in this run."

        report_parts = [self._generate_header()]
        report_parts.append(self._generate_summary(insights))
        report_parts.append("---")

        for insight in sorted(insights, key=lambda i: i.discovery_potential, reverse=True):
            report_parts.append(self._format_insight(insight))
            report_parts.append("\n---\n")

        return "\n".join(report_parts)

    def _generate_header(self) -> str:
        """Generate the report header."""
        header = f"# Scientific Comparison Report: {self.challenge_name}\n\n"
        header += f"**Algorithms Compared:** `{'`, `'.join(self.algorithm_names)}`\n\n"
        return header

    def _generate_summary(self, insights: List[ScientificInsight]) -> str:
        """Generate a summary of the key findings."""
        summary = "## Key Findings\n\n"
        # For now, a simple summary. This could be made more sophisticated.
        for insight in sorted(insights, key=lambda i: i.discovery_potential, reverse=True)[:3]:
             summary += f"- **{insight.title}:** {insight.summary}\n"
        return summary

    def _format_insight(self, insight: ScientificInsight) -> str:
        """Format a single insight into a Markdown section."""
        formatted = f"### {insight.title}\n\n"
        formatted += f"**Insight Type:** `{insight.type.value.upper()}`\n"
        formatted += f"**Confidence:** `{insight.confidence:.2%}`\n"
        formatted += f"**Discovery Potential:** `{insight.discovery_potential:.2%}`\n\n"

        formatted += "**Implications:**\n"
        for implication in insight.implications:
            formatted += f"- {implication}\n"

        if insight.evidence:
            formatted += "\n**Evidence:**\n\n"
            formatted += "| Metric | Value | Description |\n"
            formatted += "|---|---|---|\n"
            for ev in insight.evidence:
                value_str = f"{ev.metric_value:.4f}" if isinstance(ev.metric_value, float) else str(ev.metric_value)
                formatted += f"| {ev.metric_name} | `{value_str}` | {ev.description or ''} |\n"

        return formatted
