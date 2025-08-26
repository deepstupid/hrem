"""The main demo screen for the HRM System TUI."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical, Horizontal
from textual.widgets import Static, Button, Select
from rich.text import Text

from demo_config import load_challenge_config, get_configs_for_challenge, load_ui_config
from demo_utils import ResultsDisplay
from adaptive_demo_runner import AdaptiveDemoRunner
from hrm_system.config import ExperimentConfig, RunConfig, EvaluationConfig

class DemoScreen(Static):
    """The main screen for running pre-configured demo challenges."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.challenges = load_challenge_config()
        self.ui_config = load_ui_config()
        self.results_displayer = ResultsDisplay(self.ui_config)
        self.selected_challenge = self.challenges[0] if self.challenges else None

    def compose(self) -> ComposeResult:
        """Create child widgets for the demo screen."""
        challenge_options = [(c['name'], c['id']) for c in self.challenges]

        yield VerticalScroll(
            Static("🚀 HRM vs HREM Demonstration", classes="header"),
            Static("Select a pre-configured challenge to run the end-to-end demo.", classes="description"),

            Horizontal(
                Select(challenge_options, value=self.selected_challenge['id'] if self.selected_challenge else None, id="challenge_select"),
                classes="select_container"
            ),

            Static(id="challenge_details", classes="details_box"),

            Button("Run Challenge", variant="primary", id="run_challenge_button", classes="run_button"),

            Static(id="results_output")
        )

    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.update_challenge_details()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "challenge_select":
            challenge_id = event.value
            self.selected_challenge = next((c for c in self.challenges if c['id'] == challenge_id), None)
            self.update_challenge_details()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "run_challenge_button" and self.selected_challenge:
            self.query_one("#results_output", Static).update("") # Clear previous results
            self.run_challenge()

    def update_challenge_details(self):
        """Update the challenge details box."""
        if not self.selected_challenge:
            self.query_one("#challenge_details").update("[dim]No challenge selected.[/dim]")
            return

        difficulty_map = self.ui_config.get('challenge_selector', {}).get('difficulty_display', {})
        difficulty = difficulty_map.get(self.selected_challenge['difficulty'], self.selected_challenge['difficulty'])

        details = Text.assemble(
            ("Challenge: ", "bold"), f"{self.selected_challenge['name']}\n",
            ("Description: ", "bold"), f"{self.selected_challenge['description']}\n",
            ("Difficulty: ", "bold"), f"{difficulty}\n",
            ("Models: ", "bold"), f"{', '.join(self.selected_challenge['models'])}\n",
            ("Duration: ", "bold"), f"{self.selected_challenge['duration']}"
        )
        self.query_one("#challenge_details").update(details)

    def run_challenge(self):
        """Configure and run the selected challenge."""
        if not self.selected_challenge:
            return

        # 1. Get the Pydantic configs for the selected challenge
        data_config, training_config, opt_config, model_configs = get_configs_for_challenge(self.selected_challenge)

        # 2. Create the full ExperimentConfig
        run_config = RunConfig(study_name=self.selected_challenge['id'], output_dir=f"experiments/{self.selected_challenge['id']}")
        eval_config = EvaluationConfig()
        eval_config.set_models(model_configs)

        config = ExperimentConfig(
            mode="evaluate", # Start in evaluate mode for baseline
            run_config=run_config,
            data_config=data_config,
            training_config=training_config,
            evaluation_config=eval_config,
            optimization_config=opt_config
        )

        # 3. Instantiate the runner and run the demo flow
        runner = AdaptiveDemoRunner(config, self.results_displayer)
        
        # --- Baseline ---
        baseline_results = runner.run_baseline_evaluation()
        self.results_displayer.display_model_detailed_stats("📊 Baseline Results", baseline_results)

        # --- Optimization (Optional) ---
        optimized_results = {}
        if self.selected_challenge.get("optimization"):
            config.mode = "optimize"
            optimized_results = runner.run_hyperparameter_optimization()
            self.results_displayer.display_model_detailed_stats("📈 Optimization Results", optimized_results, model_names=[opt_config.model_to_optimize.name])

        # --- Final Evaluation ---
        final_results = runner.run_final_evaluation(optimized_results)
        self.results_displayer.display_final_comparison("🏆 Final Comparison", final_results)
        self.results_displayer.display_final_leader(final_results)
        runner.display_timing_summary()
