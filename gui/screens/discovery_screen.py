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
)
from ..worker import ExperimentWorker

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

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.title_label = QLabel("🔬 Discovery in Progress...")
        self.title_label.setObjectName("titleLabel")
        layout.addWidget(self.title_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        log_group = QGroupBox("📜 Live Log")
        log_layout = QVBoxLayout()
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        metrics_layout = QHBoxLayout()
        self.hrm_metrics_table = self._create_metrics_table("HRM Metrics")
        self.hrem_metrics_table = self._create_metrics_table("HREM Metrics")
        metrics_layout.addWidget(self.hrm_metrics_table)
        metrics_layout.addWidget(self.hrem_metrics_table)
        layout.addLayout(metrics_layout)

        insights_group = QGroupBox("💡 Scientific Insights")
        insights_layout = QVBoxLayout()
        self.insights_view = QTextEdit()
        self.insights_view.setReadOnly(True)
        insights_layout.addWidget(self.insights_view)
        insights_group.setLayout(insights_layout)
        layout.addWidget(insights_group)

        self.back_button = QPushButton("⬅️ Back to Configuration")
        self.back_button.clicked.connect(self.experiment_concluded.emit)
        self.back_button.hide()
        layout.addWidget(self.back_button)

    def _apply_styles(self):
        self.setStyleSheet("""
            #titleLabel {
                font-size: 16pt;
                font-weight: bold;
                color: #2c3e50;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #CCC;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QProgressBar {
                text-align: center;
                height: 28px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
            QTextEdit {
                font-family: Consolas, monospace;
            }
            QPushButton {
                background-color: #34495e;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
        """)

    def _create_metrics_table(self, title: str) -> QGroupBox:
        group_box = QGroupBox(f"📊 {title}")
        layout = QVBoxLayout()
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        group_box.setLayout(layout)
        group_box.table = table
        return group_box

    def start_experiment_run(self, config: dict):
        self.config = config
        self._reset_ui()
        self.log_view.append(f"Starting experiment with config: {config}")

        self.thread = QThread()
        self.worker = ExperimentWorker(config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress_updated.connect(self.handle_progress_update)
        self.worker.experiment_finished.connect(self.handle_experiment_finished)
        self.worker.error_occurred.connect(self.handle_error)

        self.worker.experiment_finished.connect(self.thread.quit)
        self.worker.error_occurred.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.worker.deleteLater) # type: ignore

        self.thread.start()

    def _reset_ui(self):
        self.log_view.clear()
        self.insights_view.clear()
        self.hrm_metrics_table.table.setRowCount(0)
        self.hrem_metrics_table.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.title_label.setText("🔬 Discovery in Progress...")
        self.current_algorithm = ""
        self.back_button.hide()

    @pyqtSlot(dict)
    def handle_progress_update(self, payload: dict):
        event = payload.get('event')
        data = payload.get('data', {})
        self.log_view.append(f"EVENT: {event}")

        if event == 'start_phase':
            self.title_label.setText(f"Phase: {data.get('phase', '...')}")
            self.progress_bar.setValue(0)
        elif event == 'start_algorithm':
            self.current_algorithm = data.get('algorithm', '')
        elif event == 'trainer:train_batch':
            step = data.get('step', 0)
            total_steps = data.get('total_steps', 1)
            self.progress_bar.setValue(int((step / total_steps) * 100))
            self._update_metrics(data.get('metrics', {}))
        elif event == 'end_algorithm':
            self._update_metrics(data.get('metrics', {}))
        elif event == 'insights_generated':
            self._display_insights(data.get('insights', []))

    @pyqtSlot(object)
    def handle_experiment_finished(self, results):
        self.log_view.append("\n🎉 Experiment Finished! 🎉")
        self.title_label.setText("✅ Discovery Complete!")
        self.progress_bar.setValue(100)
        self.insights_view.append("\n--- FINAL RESULTS ---")
        self.insights_view.append(str(results))
        self.back_button.show()
        self.experiment_concluded.emit()

    @pyqtSlot(str)
    def handle_error(self, error_message: str):
        self.log_view.append(f"\n❌ An error occurred: {error_message}")
        self.title_label.setText("❌ Error!")
        self.back_button.show()
        self.experiment_concluded.emit()

    def _update_metrics(self, metrics: dict):
        table = None
        if 'hrm' in self.current_algorithm.lower():
            table = self.hrm_metrics_table.table
        elif 'hrem' in self.current_algorithm.lower():
            table = self.hrem_metrics_table.table

        if table is None: return

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
