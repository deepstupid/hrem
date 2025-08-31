import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, ListView, ListItem, Button, DataTable
from rich.panel import Panel

from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.engine import ScientificDiscoveryEngine
from dataset_manager import dataset_manager

class InteractiveDiscoveryDashboard(Static):
    """An interactive dashboard for discovering and comparing model behavior."""

    def compose(self) -> ComposeResult:
        """Create the layout for the dashboard."""
        with Horizontal(id="dashboard-container"):
            # Left panel for dataset and puzzle selection
            with Vertical(id="selection-panel", classes="panel"):
                yield Static("Dataset Explorer", classes="header")
                yield ListView(
                    ListItem(Static("synthetic-sort")),
                    ListItem(Static("synthetic-copy")),
                    id="dataset-list",
                )
                yield Static("Puzzles", classes="header")
                yield ListView(id="puzzle-list")
                yield Button("Load Puzzle", variant="primary", id="load-puzzle")

            # Right panel for model comparison and insights
            with Vertical(id="main-panel", classes="panel"):
                yield Static("Model Comparison", classes="header")
                with Horizontal(id="comparison-container"):
                    with Vertical(id="hrm-panel", classes="model-panel"):
                        yield Static("HRM Model", classes="model-header")
                        yield DataTable(id="hrm-output")
                    with Vertical(id="hrem-panel", classes="model-panel"):
                        yield Static("HREM Model", classes="model-header")
                        yield DataTable(id="hrem-output")
                with Container(id="insights-panel-container", classes="panel"):
                    yield Static("Insights", classes="header")
                    yield DataTable(id="insights-panel")

    def on_mount(self) -> None:
        """Initial setup."""
        puzzle_list = self.query_one("#puzzle-list", ListView)
        for i in range(10):
            puzzle_list.append(ListItem(Static(f"Puzzle {i+1}")))

        for table_id in ["#hrm-output", "#hrem-output"]:
            table = self.query_one(table_id, DataTable)
            table.add_columns("Step", "Prediction", "Correct")

        insights_table = self.query_one("#insights-panel", DataTable)
        insights_table.add_columns("Insight", "Source")


    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "load-puzzle":
            self.run_puzzle()

    def run_puzzle(self):
        """Load and run the selected puzzle."""
        dataset_list = self.query_one("#dataset-list", ListView)
        puzzle_list = self.query_one("#puzzle-list", ListView)

        hrm_table = self.query_one("#hrm-output", DataTable)
        hrem_table = self.query_one("#hrem-output", DataTable)

        if dataset_list.highlighted is None or puzzle_list.highlighted is None:
            hrm_table.clear()
            hrm_table.add_row("Please select a dataset and a puzzle.")
            return

        dataset_name = dataset_list.children[dataset_list.highlighted].children[0].renderable
        puzzle_index = puzzle_list.highlighted

        hrm_table.clear()
        hrem_table.clear()
        self.query_one("#insights-panel", DataTable).clear()

        hrm_table.add_row(f"Loading {dataset_name}, puzzle {puzzle_index}...")

        try:
            config_manager = ConfigManager()
            challenge_configs = config_manager.load_challenge_configs()
            model_configs = config_manager.load_model_configs()
            search_spaces = config_manager.load_search_spaces()

            challenge_config = challenge_configs.challenges[0]

            algorithm_names = ["HRM", "HREM"]
            algorithm_configs = []
            for name in algorithm_names:
                model_config = model_configs.get(name)
                from types import SimpleNamespace
                algorithm_configs.append(SimpleNamespace(
                    name=model_config.name,
                    config=model_config.model_dump(),
                    search_space=search_spaces.get(name)
                ))

            with open("config/demo_config.yaml", 'r') as f:
                demo_config = yaml.safe_load(f)

            engine = ScientificDiscoveryEngine(
                challenge=challenge_config,
                algorithms=algorithm_configs,
                config=demo_config
            )

            dataset_path = dataset_manager.get_dataset_path(dataset_name, smoke_test=True)

            results = engine.run_interactive_puzzle(
                dataset_path=dataset_path,
                puzzle_index=puzzle_index,
                algorithm_configs=algorithm_configs
            )

            hrm_table.clear()
            hrem_table.clear()

            for i, step in enumerate(results.get("HRM", [])):
                hrm_table.add_row(str(i), str(step["prediction"]), "✅" if step["correct"] else "❌")

            for i, step in enumerate(results.get("HREM", [])):
                hrem_table.add_row(str(i), str(step["prediction"]), "✅" if step["correct"] else "❌")

        except Exception as e:
            hrm_table.clear()
            hrm_table.add_row(f"An error occurred: {e}")
