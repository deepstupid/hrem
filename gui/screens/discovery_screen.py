from PyQt6.QtCore import QThread, pyqtSlot, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QComboBox,
    QCheckBox, QPushButton, QLabel, QSlider, QSplitter, QTabWidget,
    QTextEdit, QTableWidget, QHeaderView, QTableWidgetItem, QMessageBox
)
from PyQt6.QtCore import Qt
from functools import partial

from ..worker import ExperimentWorker
from ..plot_widget import PlotWidget
from ..styles import DANGER_COLOR, WARNING_COLOR
from .optimization_window import OptimizationWindow
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry

class DiscoveryScreen(QWidget):
    """
    The main dashboard for running and visualizing scientific experiments.
    This screen is now the central hub of the application.
    """
    def __init__(self):
        super().__init__()
        self.worker = None
        self.thread = None
        self.metric_tables = {}
        self.algorithm_steps = {}
        self.current_algorithm = ""
        self.optimization_windows = []

        # --- Main Layout ---
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Side: Controls ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setMinimumWidth(350)
        controls_widget.setMaximumWidth(450)
        self._create_controls(controls_layout)

        # --- Right Side: Results ---
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        self._create_results_display(results_layout)

        splitter.addWidget(controls_widget)
        splitter.addWidget(results_widget)
        splitter.setSizes([400, 1000])

        # --- Populate and Connect ---
        self._populate_controls()
        self._connect_signals()
        self._setup_event_handlers()

        # --- Instant-On ---
        self.challenge_combo.currentIndexChanged.connect(self._on_start_button_clicked)
        self._start_default_experiment()

    def _create_controls(self, layout: QVBoxLayout):
        """Creates the UI controls for configuring an experiment."""
        challenge_group = QGroupBox("1. Select Challenge")
        challenge_form = QFormLayout(challenge_group)
        self.challenge_combo = QComboBox()
        self.challenge_combo.setToolTip("Select the scientific problem to investigate.")
        challenge_form.addRow("Challenge:", self.challenge_combo)

        self.models_group = QGroupBox("2. Select Models to Compare")
        self.models_layout = QVBoxLayout(self.models_group)

        patience_group = QGroupBox("3. Set Patience Level")
        patience_form = QFormLayout(patience_group)
        self.patience_slider = QSlider(Qt.Orientation.Horizontal)
        self.patience_slider.setRange(0, 2)
        self.patience_slider.setPageStep(1)
        self.patience_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.patience_slider.setTickInterval(1)
        self.patience_label = QLabel("Medium")
        patience_form.addRow(self.patience_label, self.patience_slider)

        # Action Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("🚀 Start/Restart Experiment")
        self.start_button.setObjectName("startButton")
        self.pause_button = QPushButton("⏸️ Pause")
        self.pause_button.setObjectName("pauseButton")
        self.pause_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)

        layout.addWidget(challenge_group)
        layout.addWidget(self.models_group)
        layout.addWidget(patience_group)
        layout.addStretch()
        layout.addLayout(button_layout)

    def _create_results_display(self, layout: QVBoxLayout):
        """Creates the UI elements for displaying experiment results."""
        self.results_tabs = QTabWidget()

        self.plot_widget = PlotWidget()
        self.plot_widget.initialize_plot()

        self.metrics_tabs = QTabWidget()
        self.metrics_tabs.setToolTip("Displays live metrics for each running algorithm.")

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        self.results_tabs.addTab(self.plot_widget, "📈 Performance Plot")
        self.results_tabs.addTab(self.metrics_tabs, "📊 Live Metrics")
        self.results_tabs.addTab(self.log_view, "📜 Live Log")

        layout.addWidget(self.results_tabs)

    def _populate_controls(self):
        """Populates the control widgets with data from the config files."""
        self.config_manager = ConfigManager()
        self.challenge_registry = ChallengeRegistry(self.config_manager)
        challenges = self.challenge_registry.get_all_challenges()
        for challenge in challenges:
            self.challenge_combo.addItem(challenge.name, userData=challenge.id)

        model_configs = self.config_manager.load_model_configs()
        self.model_checkboxes = {}
        for model_name in sorted(model_configs.keys()):
            checkbox = QCheckBox(model_name)
            checkbox.setChecked(True)
            self.models_layout.addWidget(checkbox)
            self.model_checkboxes[model_name] = checkbox
            checkbox.stateChanged.connect(self._on_start_button_clicked)

    def _connect_signals(self):
        """Connects UI element signals to handler slots."""
        self.patience_slider.valueChanged.connect(self._update_patience_label)
        self.patience_slider.valueChanged.connect(self._on_start_button_clicked)
        self.start_button.clicked.connect(self._on_start_button_clicked)
        self.pause_button.clicked.connect(self._on_pause_button_clicked)

    def _update_patience_label(self, value):
        levels = {0: "Low", 1: "Medium", 2: "High"}
        self.patience_label.setText(levels.get(value, "Medium"))

    def _start_default_experiment(self):
        """Kicks off a default experiment on application start."""
        self.challenge_combo.setCurrentIndex(0)

        for name, checkbox in self.model_checkboxes.items():
            checkbox.setChecked(name in ["HRM", "HREM"])

        self.patience_slider.setValue(0)
        self._on_start_button_clicked()

    def _on_start_button_clicked(self):
        """Gathers the configuration from the UI and starts the experiment."""
        if self.thread and self.thread.isRunning():
            self._cancel_experiment()

        selected_models = [name for name, checkbox in self.model_checkboxes.items() if checkbox.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, "Warning", "Please select at least one model to compare.")
            return

        patience_map = {0: "low", 1: "medium", 2: "high"}

        config = {
            "run_type": "comparison",
            "challenge_id": self.challenge_combo.currentData(),
            "patience_level": patience_map.get(self.patience_slider.value(), "medium"),
            "models": selected_models,
            "smoke_test": False
        }

        self.start_experiment_run(config)

    def _on_pause_button_clicked(self):
        """Handles the pause/resume button click."""
        if not self.worker:
            return

        if self.worker.is_paused():
            self.worker.resume()
            self.pause_button.setText("⏸️ Pause")
            self.log_view.append("<i>--- Experiment Resumed ---</i>")
        else:
            self.worker.pause()
            self.pause_button.setText("▶️ Resume")
            self.log_view.append("<i>--- Experiment Paused ---</i>")

    def start_experiment_run(self, config: dict):
        """Sets up and starts the ExperimentWorker thread."""
        self._reset_ui()
        self.log_view.append(f"<b>--- Starting Experiment ---</b>")
        self.log_view.append(f"Config: {config}")

        models_to_run = config.get("models", [])
        for model_name in models_to_run:
            if model_name:
                metric_widget, table = self._create_metric_table_widget(model_name)
                self.metric_tables[model_name] = table
                self.metrics_tabs.addTab(metric_widget, model_name)

        self.thread = QThread()
        self.worker = ExperimentWorker(config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_thread_finished)
        self.worker.progress_updated.connect(self.handle_progress_update)
        self.worker.experiment_finished.connect(self.handle_experiment_finished)
        self.worker.error_occurred.connect(self.handle_error)

        self.thread.start()
        self.start_button.setText("🛑 Stop Experiment")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(True)

    def _reset_ui(self):
        self.log_view.clear()
        self.metrics_tabs.clear()
        self.metric_tables.clear()
        self.algorithm_steps.clear()
        self.current_algorithm = ""
        self.start_button.setText("🚀 Start/Restart Experiment")
        self.pause_button.setText("⏸️ Pause")
        self.pause_button.setEnabled(False)
        if self.plot_widget:
            self.plot_widget.clear_plot()

    def _cancel_experiment(self):
        if self.worker:
            self.log_view.append("\n<b>--- User requested cancellation ---</b>")
            self.start_button.setEnabled(False)
            self.start_button.setText("🛑 Stopping...")
            self.worker.stop()

    def _handle_thread_finished(self):
        self.log_view.append("<b>--- Worker thread finished ---</b>")
        self.start_button.setText("🚀 Start/Restart Experiment")
        self.start_button.setEnabled(True)
        self.pause_button.setText("⏸️ Pause")
        self.pause_button.setEnabled(False)

        if self.worker:
            self.worker.deleteLater()
            self.worker = None

        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
            self.thread = None

    def _setup_event_handlers(self):
        self.event_handlers = {
            'start_phase': self._handle_start_phase,
            'start_algorithm': self._handle_start_algorithm,
            'trainer:train_batch': self._handle_train_batch,
            'end_algorithm': self._handle_end_algorithm,
        }

    @pyqtSlot(dict)
    def handle_progress_update(self, payload: dict):
        event = payload.get('event')
        data = payload.get('data', {})
        handler = self.event_handlers.get(event)
        if handler:
            handler(data)
        else:
            self.log_view.append(f"<font color='gray'><i>Unhandled event: {event}</i></font>")

    def _handle_start_phase(self, data: dict):
        phase = data.get('phase', '...').replace('_', ' ').title()
        self.log_view.append(f"<b>--- Starting Phase: {phase} ---</b>")

    def _handle_start_algorithm(self, data: dict):
        self.current_algorithm = data.get('algorithm', '')
        self.algorithm_steps[self.current_algorithm] = 0
        self.log_view.append(f"<font color='#007BFF'><b>--- Running Algorithm: {self.current_algorithm} ---</b></font>")
        for i in range(self.metrics_tabs.count()):
            if self.metrics_tabs.tabText(i) == self.current_algorithm:
                self.metrics_tabs.setCurrentIndex(i)
                break

    def _handle_train_batch(self, data: dict):
        metrics = data.get('metrics', {})
        self._update_metrics(metrics)
        if self.plot_widget and self.current_algorithm:
            step = self.algorithm_steps.get(self.current_algorithm, 0)
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    line_name = f"{self.current_algorithm} - {metric_name}"
                    self.plot_widget.add_point(line_name, step, value)
            self.algorithm_steps[self.current_algorithm] = step + 1

    def _handle_end_algorithm(self, data: dict):
        self.log_view.append(f"<b>--- Finished Algorithm: {self.current_algorithm} ---</b>")
        self._update_metrics(data.get('final_metrics', {}))

    @pyqtSlot(object)
    def handle_experiment_finished(self, results):
        self.log_view.append("\n<h2><font color='#28A745'>🎉 Experiment Finished! 🎉</font></h2>")

    @pyqtSlot(str)
    def handle_error(self, error_message: str):
        if "cancelled by user" in error_message:
            self.log_view.append(f"<h3><font color='{WARNING_COLOR}'>🛑 {error_message}</font></h3>")
        else:
            full_error_message = f"An unexpected error occurred:\n\n{error_message}"
            self.log_view.append(f"<h3><font color='{DANGER_COLOR}'>❌ {full_error_message}</font></h3>")
            QMessageBox.critical(self, "Experiment Error", full_error_message)

    def _create_metric_table_widget(self, model_name: str) -> (QWidget, QTableWidget):
        """Creates a container widget with a metric table and an Optimize button."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.setWordWrap(True)

        optimize_button = QPushButton(f"🧬 Launch Optimization for {model_name}")
        optimize_button.clicked.connect(partial(self._launch_optimization_window, model_name))

        layout.addWidget(table)
        layout.addWidget(optimize_button)

        return container, table

    def _launch_optimization_window(self, model_name: str):
        """Creates and shows a new window for an optimization run."""
        self.log_view.append(f"--- Launching optimization for <b>{model_name}</b> ---")

        opt_config = {
            "run_type": "optimization",
            "model_to_optimize": model_name,
            "challenge_id": self.challenge_combo.currentData(),
            "patience_level": "high", # Optimization should be thorough
            "n_trials": 50 # A reasonable default
        }

        # Create and show the new window
        opt_window = OptimizationWindow(config=opt_config)
        opt_window.show()

        # Keep a reference to it to prevent garbage collection
        self.optimization_windows.append(opt_window)

    def _update_metrics(self, metrics: dict):
        if not self.current_algorithm or not metrics: return
        table = self.metric_tables.get(self.current_algorithm)
        if not table: return
        current_metrics = {table.item(r, 0).text(): r for r in range(table.rowCount())}
        for key, value in metrics.items():
            display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            if key in current_metrics:
                # Update existing row
                table.item(current_metrics[key], 1).setText(display_value)
            else:
                # Add new row
                row_pos = table.rowCount()
                table.insertRow(row_pos)
                table.setItem(row_pos, 0, QTableWidgetItem(str(key)))
                table.setItem(row_pos, 1, QTableWidgetItem(display_value))
                # Add the new metric to our lookup
                current_metrics[key] = row_pos

    def closeEvent(self, event):
        """Ensure worker is stopped when the window closes."""
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        event.accept()
