"""The main demo screen for the HRM System TUI."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal
from textual.widgets import Static, Button, Log, ProgressBar, DataTable, Markdown, Select, Checkbox
from textual.reactive import reactive
from textual._work_decorator import work

from sc_engine.core.model_runner import ScientificModelRunner
from sc_engine.core.challenge_registry import ChallengeRegistry

class DemoScreen(Static):
    """The main screen for running pre-configured demo challenges."""

    log_messages = reactive("")
    hrm_metrics = reactive({})
    hrem_metrics = reactive({})
    phase_progress = reactive(0.0)
    current_phase = reactive("Idle")
    current_algorithm = reactive("")
    insights = reactive("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.challenge_registry = ChallengeRegistry()

    def compose(self) -> ComposeResult:
        """Create child widgets for the demo screen."""
        challenges = self.challenge_registry.get_all_challenges()
        challenge_options = [(c['name'], c['id']) for c in challenges]

        yield VerticalScroll(
            Static("🚀 HRM vs HREM Demonstration", classes="header"),
            Horizontal(
                Select(challenge_options, id="challenge_select", prompt="Select a Challenge"),
                Button("Start Demo", id="start_demo", variant="success"),
                classes="button_container"
            ),
            VerticalScroll(id="algorithm_selection_container"),
            Static("---", classes="divider"),
            Log(id="log", classes="log_view", auto_scroll=True),
            Static("---", classes="divider"),
            Static("Current Phase:", classes="label"),
            Static(self.current_phase, id="current_phase", classes="phase_indicator"),
            ProgressBar(id="phase_progress", total=1.0, show_eta=False),
            Static("---", classes="divider"),
            Horizontal(
                VerticalScroll(
                    Static("HRM Metrics", classes="table_header"),
                    DataTable(id="hrm_metrics_table", classes="metrics_table")
                ),
                VerticalScroll(
                    Static("HREM Metrics", classes="table_header"),
                    DataTable(id="hrem_metrics_table", classes="metrics_table")
                ),
            ),
            Static("---", classes="divider"),
            Static("Scientific Insights", classes="header"),
            Markdown(self.insights, id="insights_markdown")
        )

    def on_mount(self) -> None:
        """Set up the initial state of the widgets."""
        hrm_table = self.query_one("#hrm_metrics_table", DataTable)
        hrem_table = self.query_one("#hrem_metrics_table", DataTable)
        hrm_table.add_columns("Metric", "Value")
        hrem_table.add_columns("Metric", "Value")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "start_demo":
            self.query_one("#start_demo", Button).disabled = True
            self.run_demo()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle the challenge selection change."""
        if event.select.id == "challenge_select":
            challenge_id = event.value
            challenge = self.challenge_registry.get_challenge_by_id(challenge_id)
            container = self.query_one("#algorithm_selection_container")
            container.remove_children()
            if challenge:
                for model in challenge.get('models', []):
                    container.mount(Checkbox(model, id=f"alg_{model}", value=True))

    @work(exclusive=True, thread=True)
    def run_demo(self) -> None:
        """Run the demo in a background thread."""
        challenge_select = self.query_one("#challenge_select", Select)
        challenge_id = challenge_select.value

        if not challenge_id:
            self.call_from_thread(self.query_one("#log", Log).write_line, "Please select a challenge first.")
            self.query_one("#start_demo", Button).disabled = False
            return

        selected_models = []
        for checkbox in self.query(Checkbox):
            if checkbox.value:
                selected_models.append(str(checkbox.label))

        if not selected_models:
            self.call_from_thread(self.query_one("#log", Log).write_line, "Please select at least one algorithm.")
            self.query_one("#start_demo", Button).disabled = False
            return

        runner = ScientificModelRunner()
        runner.run(run_type='demo', challenge_id=challenge_id, models=selected_models, progress_callback=self._handle_progress)

    def _handle_progress(self, payload: dict) -> None:
        """Handle progress updates from the scientific engine."""
        event = payload.get('event')
        data = payload.get('data', {})

        self.call_from_thread(self.query_one("#log", Log).write_line, f"Event: {event}, Data: {data}")

        if event == 'start_session':
            self.current_phase = f"Starting session for challenge: {data.get('challenge')}"
            self.phase_progress = 0.0
        elif event == 'start_phase':
            self.current_phase = f"Phase: {data.get('phase')}"
            self.phase_progress = 0.0
        elif event == 'end_phase':
            self.phase_progress = 1.0
        elif event == 'start_algorithm':
            self.current_algorithm = data.get('algorithm', '')
            self.log_messages = f"Running algorithm: {self.current_algorithm}"
        elif event in ['end_algorithm', 'trainer:train_batch']:
            self._update_metrics(data)
        elif event == 'insights_generated':
            self._update_insights(data)

    def _update_metrics(self, data: dict) -> None:
        """Update model-specific metrics based on the current algorithm."""
        metrics = data.get('metrics', {})
        if not self.current_algorithm:
            return

        # Determine which reactive property to update
        if 'hrm' in self.current_algorithm.lower():
            self.hrm_metrics = metrics
        elif 'hrem' in self.current_algorithm.lower():
            self.hrem_metrics = metrics

        # Update progress bar for training batches
        if data.get('event') == 'trainer:train_batch':
            step = data.get('step', 0)
            total_steps = data.get('total_steps', 1)
            self.phase_progress = step / total_steps

    def _update_insights(self, data: dict) -> None:
        """Format and display scientific insights."""
        insights_data = data.get('insights', [])
        markdown_text = ""
        for insight in insights_data:
            markdown_text += f"**{insight.type.upper()}** (Confidence: {insight.confidence:.2f})\n"
            for implication in insight.implications:
                markdown_text += f"- {implication}\n"
            markdown_text += "\n"
        self.insights = markdown_text

    def _update_metrics_table(self, table_id: str, metrics: dict) -> None:
        """Helper to update a DataTable with new metrics."""
        if self.is_mounted:
            table = self.query_one(f"#{table_id}", DataTable)
            table.clear()
            for key, value in metrics.items():
                # Ensure value is a string for display
                display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                table.add_row(key, display_value)

    def watch_log_messages(self, messages: str) -> None:
        """Update the log widget when log_messages changes."""
        self.query_one("#log", Log).write(messages)

    def watch_current_phase(self, phase: str) -> None:
        """Update the current phase indicator."""
        if self.is_mounted:
            self.query_one("#current_phase", Static).update(phase)

    def watch_phase_progress(self, progress: float) -> None:
        """Update the phase progress bar."""
        if self.is_mounted:
            self.query_one("#phase_progress", ProgressBar).progress = progress

    def watch_hrm_metrics(self, metrics: dict) -> None:
        """Update the HRM metrics table."""
        self._update_metrics_table("hrm_metrics_table", metrics)

    def watch_hrem_metrics(self, metrics: dict) -> None:
        """Update the HREM metrics table."""
        self._update_metrics_table("hrem_metrics_table", metrics)

    def watch_insights(self, insights: str) -> None:
        """Update the insights markdown widget."""
        if self.is_mounted:
            self.query_one("#insights_markdown", Markdown).update(insights)
