"""Generate a scientific report from a list of insights."""

from typing import List, Dict, Any
from collections import defaultdict
from .insights import ScientificInsight, InsightType, Evidence

INSIGHT_EMOJIS = {
    InsightType.META: "🌟",
    InsightType.HYPOTHESIS: "🎯",
    InsightType.EFFICIENCY: "🚀",
    InsightType.SCALABILITY: "📈",
    InsightType.ROBUSTNESS: "🛡️",
    InsightType.CONVERGENCE: "📉",
    InsightType.GENERALIZATION: "🌍",
    InsightType.ADAPTABILITY: "🔧",
    InsightType.FAILURE: "💥",
}

class ScientificReportGenerator:
    """Generates a Markdown report from a list of scientific insights."""

    def __init__(self, challenge_name: str, algorithm_names: List[str]):
        """Initialize the report generator."""
        self.challenge_name = challenge_name
        self.algorithm_names = algorithm_names

    def generate_report(self, insights: List[ScientificInsight]) -> str:
        """Generate a Markdown report from a list of insights."""
        if not insights:
            return "## No significant insights were generated in this run."

        report_parts = [self._generate_header()]

        # Separate insights for structured reporting
        meta_insights = [i for i in insights if i.type == InsightType.META]
        hypothesis_insights = [i for i in insights if i.type == InsightType.HYPOTHESIS]
        failure_insights = [i for i in insights if i.type == InsightType.FAILURE]
        other_insights = [i for i in insights if i.type not in [InsightType.META, InsightType.HYPOTHESIS, InsightType.FAILURE]]

        report_parts.append(self._generate_executive_summary(meta_insights, failure_insights))
        report_parts.append("---")

        if hypothesis_insights:
            report_parts.append(self._generate_hypothesis_section(hypothesis_insights))
            report_parts.append("---")

        if other_insights:
            report_parts.append(self._generate_detailed_findings(other_insights))
            report_parts.append("---")

        if failure_insights:
            report_parts.append(self._generate_failure_analysis_section(failure_insights))
            report_parts.append("---")

        return "\n".join(report_parts)

    def _generate_header(self) -> str:
        """Generate the report header."""
        return f"# Scientific Comparison Report: {self.challenge_name}\n\n**Algorithms Compared:** `{'`, `'.join(self.algorithm_names)}`\n"

    def _generate_executive_summary(self, meta_insights: List[ScientificInsight], failure_insights: List[ScientificInsight]) -> str:
        """Generate a high-level summary of the most important findings and recommendations."""
        summary = "## 🌟 Executive Summary\n\n"
        if failure_insights:
            summary += "**Warning:** One or more models encountered failures during the run. See the Failure Analysis section for details.\n\n"

        if not meta_insights:
            summary += "No overarching meta-insights were identified. See detailed findings below.\n"
            return summary

        for insight in meta_insights:
            summary += f"### {insight.title}\n"
            summary += f"{insight.summary}\n\n"
            if insight.recommendations:
                summary += "**Top Recommendation:**\n"
                for rec in insight.recommendations:
                    summary += f"> {rec}\n"
        return summary

    def _generate_hypothesis_section(self, insights: List[ScientificInsight]) -> str:
        """Generate the hypothesis testing section."""
        section = "## 🎯 Hypothesis Testing\n\n"
        for insight in sorted(insights, key=lambda i: i.title):
            section += self._format_insight(insight)
            section += "\n"
        return section

    def _generate_failure_analysis_section(self, insights: List[ScientificInsight]) -> str:
        """Generate the failure analysis section."""
        section = "## 💥 Failure Analysis\n\n"
        for insight in insights:
            section += self._format_insight(insight)
            section += "\n"
        return section

    def _generate_detailed_findings(self, insights: List[ScientificInsight]) -> str:
        """Generate the detailed findings section, grouped by insight type."""
        section = "## 📊 Detailed Findings\n\n"

        grouped_insights = defaultdict(list)
        for insight in insights:
            grouped_insights[insight.type].append(insight)

        for insight_type, insight_list in sorted(grouped_insights.items(), key=lambda item: item[0].value):
            emoji = INSIGHT_EMOJIS.get(insight_type, "🔹")
            section += f"### {emoji} {insight_type.value.replace('_', ' ').title()}\n\n"
            for insight in sorted(insight_list, key=lambda i: i.discovery_potential, reverse=True):
                section += self._format_insight(insight)
                section += "\n"
        return section

    def _format_insight(self, insight: ScientificInsight) -> str:
        """Format a single insight into a Markdown section."""
        emoji = INSIGHT_EMOJIS.get(insight.type, "🔹")
        formatted = f"#### {emoji} {insight.title}\n\n"

        if insight.type not in [InsightType.HYPOTHESIS, InsightType.META]:
             formatted += f"**Confidence:** `{insight.confidence:.1%}` | **Discovery Potential:** `{insight.discovery_potential:.1%}`\n\n"

        if insight.summary:
            formatted += f"_{insight.summary}_\n\n"

        if insight.implications:
            formatted += "**Implications:**\n"
            for implication in insight.implications:
                formatted += f"- {implication}\n"
            formatted += "\n"

        if insight.causal_attribution:
            formatted += f"**Causal Attribution:** {insight.causal_attribution}\n\n"

        if insight.recommendations:
            formatted += "**Recommendations:**\n"
            for rec in insight.recommendations:
                formatted += f"- {rec}\n"
            formatted += "\n"

        if insight.evidence:
            formatted += "**Evidence:**\n"
            formatted += "| Metric | Value | Effect Size | p-value | Description |\n"
            formatted += "|---|---|---|---|---|\n"
            for ev in insight.evidence:
                value_str = f"{ev.metric_value:.4f}" if isinstance(ev.metric_value, float) else str(ev.metric_value)
                effect_str = f"{ev.effect_size:.3f}" if ev.effect_size is not None else "N/A"
                pval_str = f"{ev.p_value:.3f}" if ev.p_value is not None else "N/A"
                formatted += f"| {ev.metric_name} | `{value_str}` | `{effect_str}` | `{pval_str}` | {ev.description or ''} |\n"

        return formatted
