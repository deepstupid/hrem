import os
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Button, Select, Checkbox

from hrm_system import (
    ExperimentConfig,
    RunConfig,
    DataConfig,
    TrainingConfig,
    EvaluationConfig,
    OptimizationConfig,
    ModelConfig,
)
from .progress import ProgressScreen


class DemoScreen(Static):
    """The configuration screen for the HRM System demo."""

    DATASET_TASKS = {
        "synthetic": ["copy", "reverse"],
        "arc": ["arc"],
        "maze": ["maze"],
        "sudoku": ["sudoku"],
    }

    def compose(self) -> ComposeResult:
        yield Static("Experiment Configuration", classes="header")

        with Vertical(id="config-form"):
            yield Static("Dataset:")
            yield Select(
                options=[(d, d) for d in self.DATASET_TASKS.keys()],
                value="synthetic",
                id="dataset_select",
            )

            yield Static("Task:")
            yield Select(
                options=[(t, t) for t in self.DATASET_TASKS["synthetic"]],
                value="copy",
                id="task_select",
            )

            yield Static("Algorithms:")
            yield Checkbox("HRM", value=True, id="hrm_checkbox")
            yield Checkbox("HREM", value=True, id="hrem_checkbox")

            yield Static("Options:")
            yield Checkbox("Run Hyperparameter Optimization", value=False, id="optimize_checkbox")

            yield Button("Run Experiment", variant="primary", id="run_experiment_button")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "dataset_select":
            if event.value is None:
                return
            tasks = self.DATASET_TASKS.get(event.value, [])
            task_select = self.query_one("#task_select", Select)
            task_select.set_options([(t, t) for t in tasks])
            if tasks:
                task_select.value = tasks[0]
            else:
                task_select.value = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "run_experiment_button":
            self.run_experiment()

    def run_experiment(self) -> None:
        """Configure and run the experiment."""
        dataset_value = self.query_one("#dataset_select", Select).value
        task_value = self.query_one("#task_select", Select).value

        if not dataset_value or not task_value:
            return

        use_hrm = self.query_one("#hrm_checkbox", Checkbox).value
        use_hrem = self.query_one("#hrem_checkbox", Checkbox).value
        optimize = self.query_one("#optimize_checkbox", Checkbox).value

        model_a = None
        if use_hrm:
            model_a = ModelConfig(
                name="HRM",
                algorithm_class="hrm_system.algorithms.hrm.HRMAlgorithm",
                base_arch_config="hrm_v1",
            )

        model_b = None
        if use_hrem:
            model_b = ModelConfig(
                name="HREM",
                algorithm_class="hrm_system.algorithms.hrem.HREMAlgorithm",
                base_arch_config="hrem_v1",
            )

        run_config = RunConfig(
            smoke_test=True,
            study_name="tui_experiment",
        )
        data_config = DataConfig(dataset=dataset_value, synthetic_task=task_value)
        training_config = TrainingConfig(epochs=50, eval_interval=25)

        if optimize:
            mode = "optimize"
            db_path = "experiments/optuna_tui.db"
            if os.path.exists(db_path):
                os.remove(db_path)

            optimization_config = OptimizationConfig(
                n_trials=3,
                n_jobs=1,
                storage=f"sqlite:///{db_path}",
                n_final_runs=1,
                model_to_optimize=model_b,
            )
            evaluation_config = None
        else:
            mode = "evaluate"
            evaluation_config = EvaluationConfig(
                n_runs=1,
                model_a=model_a,
                model_b=model_b,
            )
            optimization_config = None
        
        config = ExperimentConfig(
            mode=mode,
            run_config=run_config,
            data_config=data_config,
            training_config=training_config,
            evaluation_config=evaluation_config,
            optimization_config=optimization_config,
        )
        
        progress_screen = ProgressScreen(experiment_config=config)
        if config.run_config:
            config.run_config.logger_callback = progress_screen.log_message
        
        self.app.push_screen(progress_screen)
