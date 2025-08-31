import os
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

    def __init__(self, challenge_name: str, algorithm_names: List[str], final_results: Dict[str, Any] = None, plot_path: str = None):
        """Initialize the report generator."""
        self.challenge_name = challenge_name
        self.algorithm_names = algorithm_names
        self.final_results = final_results
        self.plot_path = plot_path

    def generate_report(self, insights: List[ScientificInsight]) -> str:
        """Generate a Markdown report from a list of insights."""
        report_parts = [self._generate_header()]

        if self.final_results:
            report_parts.append(self._generate_summary_table())
            report_parts.append("---")

        if self.plot_path:
            report_parts.append(self._generate_plot_section())
            report_parts.append("---")

        if not insights:
            report_parts.append("## No significant insights were generated in this run.")
            return "\n".join(report_parts)

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

    def _generate_summary_table(self) -> str:
        """Generates a markdown table of the final results."""
        section = "## 📈 Final Metrics Summary\n\n"
        if not self.final_results:
            return ""

        headers = ["Algorithm"]
        all_keys = set()
        for result in self.final_results.values():
            # The result for each algorithm is now a dictionary of metrics for each dataset split (e.g., 'all')
            for split_result in result.values():
                if isinstance(split_result, dict):
                    all_keys.update(split_result.keys())

        sorted_keys = sorted(list(all_keys), key=lambda x: ('loss' not in x, 'accuracy' not in x, x))
        headers.extend([key.replace('_', ' ').title() for key in sorted_keys])

        section += f"| {' | '.join(headers)} |\n"
        section += f"|{'|'.join(['---'] * len(headers))}|\n"

        for alg_name, results in self.final_results.items():
            row = [alg_name.replace('_optimized', ' (Optimized)')]
            # We will display the 'all' split results in the summary table
            summary_metrics = results.get('all', {})
            for key in sorted_keys:
                value = summary_metrics.get(key)
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value) if value is not None else "N/A")
            section += f"| {' | '.join(row)} |\n"

        return section + "\n"

    def _generate_plot_section(self) -> str:
        """Generates the markdown section for the performance plot."""
        section = "## 📊 Performance Plot\n\n"
        if not self.plot_path or not os.path.exists(self.plot_path):
            return section + "No performance plot was generated for this run.\n"

        relative_plot_path = os.path.relpath(self.plot_path, start=os.getcwd())
        section += f"![Performance Plot]({relative_plot_path})\n"
        return section + "\n"

    def _generate_executive_summary(self, meta_insights: List[ScientificInsight], failure_insights: List[ScientificInsight]) -> str:
        summary = "## 🌟 Executive Summary\n\n"
        if failure_insights:
            summary += "**Warning:** One or more models encountered failures during the run. See the Failure Analysis section for details.\n\n"
        if not meta_insights:
            summary += "No overarching meta-insights were identified. See detailed findings below.\n"
            return summary
        for insight in meta_insights:
            summary += f"### {insight.title}\n{insight.summary}\n\n"
            if insight.recommendations:
                summary += "**Top Recommendation:**\n"
                for rec in insight.recommendations:
                    summary += f"> {rec}\n"
        return summary

    def _generate_hypothesis_section(self, insights: List[ScientificInsight]) -> str:
        section = "## 🎯 Hypothesis Testing\n\n"
        for insight in sorted(insights, key=lambda i: i.title):
            section += self._format_insight(insight)
            section += "\n"
        return section

    def _generate_failure_analysis_section(self, insights: List[ScientificInsight]) -> str:
        section = "## 💥 Failure Analysis\n\n"
        for insight in insights:
            section += self._format_insight(insight)
            section += "\n"
        return section

    def _generate_detailed_findings(self, insights: List[ScientificInsight]) -> str:
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
            formatted += "**Evidence:**\n| Metric | Value | Effect Size | p-value | Description |\n|---|---|---|---|---|\n"
            for ev in insight.evidence:
                value_str = f"{ev.metric_value:.4f}" if isinstance(ev.metric_value, float) else str(ev.metric_value)
                effect_str = f"{ev.effect_size:.3f}" if ev.effect_size is not None else "N/A"
                pval_str = f"{ev.p_value:.3f}" if ev.p_value is not None else "N/A"
                formatted += f"| {ev.metric_name} | `{value_str}` | `{effect_str}` | `{pval_str}` | {ev.description or ''} |\n"
        return formatted
