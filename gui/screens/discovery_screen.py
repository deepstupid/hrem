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
    QSplitter,
)
from PyQt6.QtCore import Qt
from ..worker import ExperimentWorker

class DiscoveryScreen(QWidget):
    """
    The 'Run Experiment' screen (Discovery Phase). Displays live progress,
    logs, metrics, and final insights from the experiment.
    """
    experiment_concluded = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.current_algorithm = ""
        self.metric_tables = {}

        # --- Layout ---
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # --- Top Section (Title & Progress) ---
        top_layout = QHBoxLayout()
        self.title_label = QLabel("🔬 Discovery in Progress...")
        self.title_label.setObjectName("titleLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setToolTip("Overall experiment progress.")
        self.progress_bar.setTextVisible(False)
        top_layout.addWidget(self.title_label)
        top_layout.addWidget(self.progress_bar)
        layout.addLayout(top_layout)

        # --- Main Content (Splitter) ---
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter, 1) # Give it stretch factor

        # --- Left Side (Metrics & Logs) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0,0,0,0)

        # Metrics
        metrics_group = QGroupBox("📊 Live Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_tabs = QTabWidget()
        self.metrics_tabs.setToolTip("Displays live metrics for each running algorithm.")
        metrics_layout.addWidget(self.metrics_tabs)

        # Logs
        log_group = QGroupBox("📜 Live Log")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setToolTip("Displays real-time events from the experiment.")
        log_layout.addWidget(self.log_view)

        left_layout.addWidget(metrics_group, 1) # Stretch
        left_layout.addWidget(log_group, 1) # Stretch

        # --- Right Side (Insights) ---
        insights_group = QGroupBox("💡 Scientific Insights")
        insights_layout = QVBoxLayout(insights_group)
        self.insights_view = QTextEdit()
        self.insights_view.setReadOnly(True)
        self.insights_view.setToolTip("Shows high-level insights and final results after the experiment.")
        insights_layout.addWidget(self.insights_view)

        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(insights_group)
        main_splitter.setSizes([600, 400]) # Initial size distribution

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("🛑 Cancel Experiment")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setToolTip("Stops the currently running experiment.")
        self.conclusion_button = QPushButton("🎉 Finish")
        self.conclusion_button.setObjectName("finishButton")
        self.conclusion_button.setToolTip("Return to the setup screen.")

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.conclusion_button)
        layout.addLayout(button_layout)

        # --- Connections & Event Handlers ---
        self._setup_event_handlers()
        self.cancel_button.clicked.connect(self._cancel_experiment)
        self.conclusion_button.clicked.connect(self.experiment_concluded.emit)


    def _create_metric_table_widget(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.setWordWrap(True)
        return table

    def start_experiment_run(self, config: dict):
        self._reset_ui()
        self.log_view.append(f"<b>--- Starting Experiment ---</b>")
        self.log_view.append(f"Config: {config}")

        models_to_run = config.get("models", [])
        if config.get("run_type") == "optimization":
            models_to_run = [config.get("model_to_optimize")]

        for model_name in models_to_run:
            if model_name:
                table = self._create_metric_table_widget()
                self.metric_tables[model_name] = table
                self.metrics_tabs.addTab(table, model_name)

        self.thread = QThread()
        self.worker = ExperimentWorker(config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_thread_finished)
        self.worker.progress_updated.connect(self.handle_progress_update)
        self.worker.experiment_finished.connect(self.handle_experiment_finished)
        self.worker.error_occurred.connect(self.handle_error)

        self.thread.start()

    def _reset_ui(self):
        self.log_view.clear()
        self.insights_view.clear()
        self.insights_view.setPlaceholderText("Insights will be generated here as the experiment concludes...")
        self.metrics_tabs.clear()
        self.metric_tables.clear()
        self.progress_bar.setValue(0)
        self.title_label.setText("🔬 Discovery in Progress...")
        self.current_algorithm = ""
        self.conclusion_button.hide()
        self.cancel_button.show()
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("🛑 Cancel Experiment")


    def _cancel_experiment(self):
        if self.worker:
            self.log_view.append("\n<b>--- User requested cancellation ---</b>")
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("🛑 Cancelling...")
            self.worker.stop()

    def _handle_thread_finished(self):
        self.log_view.append("<b>--- Worker thread finished ---</b>")
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
        self.event_handlers = {
            'start_phase': self._handle_start_phase,
            'start_algorithm': self._handle_start_algorithm,
            'trainer:train_batch': self._handle_train_batch,
            'end_algorithm': self._handle_end_algorithm,
            'insights_generated': self._handle_insights_generated,
            'patience_update': self._handle_patience_update,
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

    def _handle_patience_update(self, data: dict):
        used = data.get('used', 0)
        total = data.get('total', 1)
        if total > 0:
            self.progress_bar.setValue(int((used / total) * 100))

    def _handle_start_phase(self, data: dict):
        phase = data.get('phase', '...').replace('_', ' ').title()
        self.title_label.setText(f"Phase: {phase}")
        self.log_view.append(f"<b>--- Starting Phase: {phase} ---</b>")

    def _handle_start_algorithm(self, data: dict):
        self.current_algorithm = data.get('algorithm', '')
        self.log_view.append(f"<font color='#007BFF'><b>--- Running Algorithm: {self.current_algorithm} ---</b></font>")
        # Switch to the correct metric tab
        for i in range(self.metrics_tabs.count()):
            if self.metrics_tabs.tabText(i) == self.current_algorithm:
                self.metrics_tabs.setCurrentIndex(i)
                break

    def _handle_train_batch(self, data: dict):
        self._update_metrics(data.get('metrics', {}))

    def _handle_end_algorithm(self, data: dict):
        self.log_view.append(f"<b>--- Finished Algorithm: {self.current_algorithm} ---</b>")
        self._update_metrics(data.get('final_metrics', {}))

    def _handle_insights_generated(self, data: dict):
        self._display_insights(data.get('insights', []))

    @pyqtSlot(object)
    def handle_experiment_finished(self, results):
        self.log_view.append("\n<h2><font color='#28A745'>🎉 Experiment Finished! 🎉</font></h2>")
        self.title_label.setText("✅ Discovery Complete!")
        self.progress_bar.setValue(100)

        self.insights_view.clear()
        self.insights_view.append("<h1>Final Report</h1>")

        if hasattr(results, 'insights') and results.insights:
            self.insights_view.append("<h2>Scientific Insights</h2>")
            self._display_insights(results.insights)
        else:
            self.insights_view.append("<h2>Scientific Insights</h2><i>No significant insights were generated.</i>")

        if hasattr(results, 'metadata') and 'total_elapsed_time' in results.metadata:
            total_time = results.metadata['total_elapsed_time']
            self.insights_view.append(f"<h2>Execution Summary</h2>"
                                      f"<b>Total Duration:</b> {total_time:.2f} seconds")

        if hasattr(results, 'timing_data'):
            self.insights_view.append("<b>Timing Breakdown:</b><ul>")
            for phase, data in results.timing_data.items():
                self.insights_view.append(f"<li><b>{phase.replace('_', ' ').title()}:</b> {data['duration']:.2f}s</li>")
            self.insights_view.append("</ul>")

        if not hasattr(results, 'insights') and not hasattr(results, 'metadata'):
            self.insights_view.append("<h2>Raw Results</h2>")
            self.insights_view.append(f"<pre>{results}</pre>")

    @pyqtSlot(str)
    def handle_error(self, error_message: str):
        if "cancelled by user" in error_message:
            self.log_view.append(f"<h3><font color='{WARNING_COLOR}'>🛑 {error_message}</font></h3>")
            self.title_label.setText("🛑 Experiment Cancelled")
        else:
            full_error_message = f"An unexpected error occurred:\n\n{error_message}"
            self.log_view.append(f"<h3><font color='{DANGER_COLOR}'>❌ {full_error_message}</font></h3>")
            self.title_label.setText("❌ Error!")
            QMessageBox.critical(self, "Experiment Error", full_error_message)

    def _update_metrics(self, metrics: dict):
        if not self.current_algorithm or not metrics:
            return
        table = self.metric_tables.get(self.current_algorithm)
        if not table: return

        # Efficiently update or add new rows
        current_metrics = {table.item(r, 0).text(): r for r in range(table.rowCount())}
        for key, value in metrics.items():
            display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            if key in current_metrics:
                table.item(current_metrics[key], 1).setText(display_value)
            else:
                row_pos = table.rowCount()
                table.insertRow(row_pos)
                table.setItem(row_pos, 0, QTableWidgetItem(str(key)))
                table.setItem(row_pos, 1, QTableWidgetItem(display_value))


    def _display_insights(self, insights: list):
        self.insights_view.append("---")
        for insight in insights:
            try:
                # Using HTML for rich text formatting
                insight_html = (
                    f"<b>Insight Type:</b> {insight.type.upper()}<br>"
                    f"<b>Confidence:</b> {insight.confidence:.2f}<br>"
                    f"<b>Description:</b> {insight.description}<br>"
                    f"<b>Implications:</b><ul>"
                )
                for imp in insight.implications:
                    insight_html += f"<li>{imp}</li>"
                insight_html += "</ul>"
                self.insights_view.append(insight_html)
            except AttributeError:
                self.insights_view.append(f"<pre>{insight}</pre>")
