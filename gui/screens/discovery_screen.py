from PyQt6.QtCore import QThread, pyqtSlot, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QProgressBar,
    QTableWidget,
    QHeaderView,
    QTableWidgetItem,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QTabWidget,
)
from ..worker import ExperimentWorker, ExperimentCancelledError

class DiscoveryScreen(QWidget):
    """The 'Run Experiment' screen (Discovery Phase)."""
    experiment_concluded = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._apply_styles()
        self.config = None
        self.thread = None
        self.worker = None
        self.current_algorithm = ""
        self.metric_tables = {}
        self._setup_event_handlers()

        # --- Layout ---
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # --- Top Section (Title & Progress) ---
        self.title_label = QLabel("🔬 Discovery in Progress...")
        self.title_label.setObjectName("titleLabel")
        layout.addWidget(self.title_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setToolTip("Shows the progress of the current training batch.")
        layout.addWidget(self.progress_bar)

        # --- Log & Metrics ---
        log_group = QGroupBox("📜 Live Log")
        log_group.setToolTip("Displays real-time events from the experiment.")
        log_layout = QVBoxLayout()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # --- Metrics Tab ---
        metrics_group = QGroupBox("📊 Live Metrics")
        metrics_group.setToolTip("Displays live metrics for each running algorithm.")
        metrics_layout = QVBoxLayout()
        self.metrics_tabs = QTabWidget()
        metrics_layout.addWidget(self.metrics_tabs)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # --- Insights ---
        insights_group = QGroupBox("💡 Scientific Insights")
        insights_group.setToolTip("Shows high-level insights and final results after the experiment.")
        insights_layout = QVBoxLayout()
        self.insights_view = QTextEdit()
        self.insights_view.setReadOnly(True)
        insights_layout.addWidget(self.insights_view)
        insights_group.setLayout(insights_layout)
        layout.addWidget(insights_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("🛑 Cancel Experiment")
        self.cancel_button.setToolTip("Stops the currently running experiment.")
        self.cancel_button.clicked.connect(self._cancel_experiment)
        self.conclusion_button = QPushButton("⬅️ Back to Configuration")
        self.conclusion_button.setToolTip("Return to the setup screen.")
        self.conclusion_button.clicked.connect(self.experiment_concluded.emit)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        button_layout.addWidget(self.conclusion_button)
        layout.addLayout(button_layout)


    def _apply_styles(self):
        self.setStyleSheet("""
            /* ... (styles remain the same) ... */
        """)

    def _create_metric_table_widget(self) -> QTableWidget:
        """Creates a standard QTableWidget for displaying metrics."""
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def start_experiment_run(self, config: dict):
        self.config = config
        self._reset_ui()
        self.log_view.append(f"Starting experiment with config: {config}")

        # --- Create metric tabs for each model ---
        models_to_run = []
        if config.get("run_type") == "optimization":
            models_to_run.append(config.get("model_to_optimize"))
        else:
            models_to_run = config.get("models", [])

        for model_name in models_to_run:
            if model_name:
                table = self._create_metric_table_widget()
                self.metric_tables[model_name] = table
                self.metrics_tabs.addTab(table, model_name)

        self.thread = QThread()
        self.worker = ExperimentWorker(config)
        self.worker.moveToThread(self.thread)

        # --- Connect signals ---
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_thread_finished)
        self.worker.progress_updated.connect(self.handle_progress_update)
        self.worker.experiment_finished.connect(self.handle_experiment_finished)
        self.worker.error_occurred.connect(self.handle_error)

        self.thread.start()

    def _reset_ui(self):
        self.log_view.clear()
        self.insights_view.clear()
        self.metrics_tabs.clear()
        self.metric_tables.clear()
        self.progress_bar.setValue(0)
        self.title_label.setText("🔬 Discovery in Progress...")
        self.current_algorithm = ""
        self.conclusion_button.hide()
        self.cancel_button.show()

    def _cancel_experiment(self):
        if self.worker:
            self.log_view.append("\n🛑 User requested cancellation...")
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("🛑 Cancelling...")
            self.worker.stop()

    def _handle_thread_finished(self):
        """Cleans up the thread and worker objects."""
        self.log_view.append("Cleaning up worker thread.")
        self.cancel_button.hide()
        self.conclusion_button.show()
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
            self.thread = None
        if self.worker:
            self.worker.deleteLater()
            self.worker = None


    def _setup_event_handlers(self):
        """Initializes the event handler mapping."""
        self.event_handlers = {
            'start_phase': self._handle_start_phase,
            'start_algorithm': self._handle_start_algorithm,
            'trainer:train_batch': self._handle_train_batch,
            'end_algorithm': self._handle_end_algorithm,
            'insights_generated': self._handle_insights_generated,
        }

    @pyqtSlot(dict)
    def handle_progress_update(self, payload: dict):
        """Dispatches progress updates to the appropriate handler."""
        event = payload.get('event')
        data = payload.get('data', {})

        self.log_view.append(f"EVENT: {event}")

        handler = self.event_handlers.get(event)
        if handler:
            handler(data)
        else:
            self.log_view.append(f"WARNING: No handler for event '{event}'")

    def _handle_start_phase(self, data: dict):
        self.title_label.setText(f"Phase: {data.get('phase', '...')}")
        self.progress_bar.setValue(0)

    def _handle_start_algorithm(self, data: dict):
        self.current_algorithm = data.get('algorithm', '')
        self.log_view.append(f"--- Running Algorithm: {self.current_algorithm} ---")

    def _handle_train_batch(self, data: dict):
        step = data.get('step', 0)
        total_steps = data.get('total_steps', 1)
        if total_steps > 0:
            self.progress_bar.setValue(int((step / total_steps) * 100))
        self._update_metrics(data.get('metrics', {}))

    def _handle_end_algorithm(self, data: dict):
        self._update_metrics(data.get('metrics', {}))
        self.log_view.append(f"--- Finished Algorithm: {self.current_algorithm} ---")

    def _handle_insights_generated(self, data: dict):
        self._display_insights(data.get('insights', []))

    @pyqtSlot(object)
    def handle_experiment_finished(self, results):
        """
        Handles the final results of the experiment, parsing and displaying them.
        """
        self.log_view.append("\n🎉 Experiment Finished! 🎉")
        self.title_label.setText("✅ Discovery Complete!")
        self.progress_bar.setValue(100)
        self.conclusion_button.setText("🎉 Finish")

        # Clear previous results and display new ones
        self.insights_view.clear()
        self.insights_view.append("<h2>Final Results</h2>")

        # 1. Display final metrics in tables
        if hasattr(results, 'algorithm_results'):
            for alg_name, metrics in results.algorithm_results.items():
                # Fake an algorithm name in the context to reuse the update function
                self.current_algorithm = alg_name
                self._update_metrics(metrics)
            self.current_algorithm = "" # Reset context

        # 2. Display scientific insights
        if hasattr(results, 'insights') and results.insights:
            self._display_insights(results.insights)
        else:
            self.insights_view.append("<i>No significant insights were generated.</i>")

        # 3. Display timing and metadata
        self.insights_view.append("<br><h3>Execution Summary</h3>")
        if hasattr(results, 'metadata') and 'total_elapsed_time' in results.metadata:
            total_time = results.metadata['total_elapsed_time']
            self.insights_view.append(f"<b>Total Duration:</b> {total_time:.2f} seconds")
        if hasattr(results, 'timing_data'):
            self.insights_view.append("<b>Timing Breakdown:</b><ul>")
            for phase, data in results.timing_data.items():
                self.insights_view.append(f"<li><b>{phase.replace('_', ' ').title()}:</b> {data['duration']:.2f}s</li>")
            self.insights_view.append("</ul>")

        # Fallback for unexpected results format
        if not hasattr(results, 'algorithm_results'):
            self.insights_view.append("<br><i>Could not parse detailed results. Raw output:</i>")
            self.insights_view.append(str(results))

    @pyqtSlot(str)
    def handle_error(self, error_message: str):
        if "cancelled by user" in error_message:
            self.log_view.append(f"\n🛑 {error_message}")
            self.title_label.setText("🛑 Experiment Cancelled")
            self.conclusion_button.setText("⬅️ Back to Configuration")
        else:
            full_error_message = f"An unexpected error occurred:\n\n{error_message}"
            self.log_view.append(f"\n❌ {full_error_message}")
            self.title_label.setText("❌ Error!")
            self.conclusion_button.setText("Acknowledged Error")
            QMessageBox.critical(
                self,
                "Experiment Error",
                full_error_message,
                QMessageBox.StandardButton.Ok
            )

    def _update_metrics(self, metrics: dict):
        if not self.current_algorithm or not metrics:
            return

        table = self.metric_tables.get(self.current_algorithm)
        if table is None:
            self.log_view.append(f"Warning: Could not find metric table for algorithm '{self.current_algorithm}'")
            return

        table.setRowCount(len(metrics))
        for row, (key, value) in enumerate(metrics.items()):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            table.setItem(row, 1, QTableWidgetItem(display_value))

    def _display_insights(self, insights: list):
        self.insights_view.append("\n--- New Insights ---")
        for insight in insights:
            try:
                markdown_text = f"<b>{insight.type.upper()}</b> (Confidence: {insight.confidence:.2f})<br>"
                implications_html = "<ul>" + "".join([f"<li>{imp}</li>" for imp in insight.implications]) + "</ul>"
                self.insights_view.insertHtml(markdown_text + implications_html)
            except AttributeError:
                self.insights_view.append(str(insight))
