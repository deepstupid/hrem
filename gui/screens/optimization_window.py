from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout
from PyQt6.QtCore import QThread, pyqtSlot

from ..worker import ExperimentWorker
from ..styles import DANGER_COLOR, WARNING_COLOR

class OptimizationWindow(QMainWindow):
    """
    A dedicated window for running and monitoring a hyperparameter
    optimization experiment for a single model.
    """
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.worker = None
        self.thread = None

        model_name = self.config.get("model_to_optimize", "Unknown")
        challenge_id = self.config.get("challenge_id", "Unknown")

        self.setWindowTitle(f"Optimization: {model_name} on {challenge_id}")
        self.setGeometry(150, 150, 800, 600)

        # --- Main Widget and Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- UI Elements (Scaffold) ---
        self.title_label = QLabel(f"Optimizing <b>{model_name}</b> on <b>{challenge_id}</b>")
        self.title_label.setObjectName("titleLabel")

        self.results_view = QTextEdit()
        self.results_view.setReadOnly(True)
        self.results_view.setPlaceholderText("Optimization trial results will appear here...")

        # --- Buttons ---
        button_layout = QHBoxLayout()
        self.pause_button = QPushButton("⏸️ Pause")
        self.pause_button.setEnabled(False)
        button_layout.addStretch()
        button_layout.addWidget(self.pause_button)

        layout.addWidget(self.title_label)
        layout.addWidget(self.results_view)
        layout.addLayout(button_layout)

        # --- Connections and Start ---
        self.pause_button.clicked.connect(self._on_pause_button_clicked)
        self.start_optimization_run()

    def start_optimization_run(self):
        """Sets up and starts the ExperimentWorker for optimization."""
        self.results_view.append(f"<b>--- Starting Optimization ---</b>")
        self.results_view.append(f"Config: {self.config}")

        self.thread = QThread()
        self.worker = ExperimentWorker(self.config)
        self.worker.moveToThread(self.thread)

        self.worker.progress_updated.connect(self.handle_progress_update)
        self.worker.finished.connect(self._handle_thread_finished)
        self.worker.error_occurred.connect(self.handle_error)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
        self.pause_button.setEnabled(True)

    @pyqtSlot(dict)
    def handle_progress_update(self, payload: dict):
        """Handles progress updates from the optimization worker."""
        event = payload.get('event')
        data = payload.get('data', {})

        if event == 'optimization_trial_result':
            trial_num = data.get('trial', {}).get('number', -1)
            value = data.get('trial', {}).get('value', float('inf'))
            params = data.get('trial', {}).get('params', {})
            self.results_view.append(
                f"<b>Trial {trial_num} completed:</b> Value = {value:.4f}"
                f"<br>Params: <i>{params}</i>"
            )
        else:
            self.results_view.append(f"<font color='gray'><i>Event: {event}</i></font>")

        self.results_view.verticalScrollBar().setValue(self.results_view.verticalScrollBar().maximum())


    @pyqtSlot(str)
    def handle_error(self, error_message: str):
        self.results_view.append(f"<h3><font color='{DANGER_COLOR}'>❌ {error_message}</font></h3>")

    def _on_pause_button_clicked(self):
        """Handles the pause/resume button click."""
        if not self.worker:
            return

        if self.worker.is_paused():
            self.worker.resume()
            self.pause_button.setText("⏸️ Pause")
            self.results_view.append("<i>--- Optimization Resumed ---</i>")
        else:
            self.worker.pause()
            self.pause_button.setText("▶️ Resume")
            self.results_view.append("<i>--- Optimization Paused ---</i>")

    def _handle_thread_finished(self):
        self.results_view.append("<b>--- Optimization Worker Thread Finished ---</b>")
        self.pause_button.setEnabled(False)
        self.pause_button.setText("⏸️ Pause")
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
            self.thread = None
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def closeEvent(self, event):
        """Handle the user closing the window."""
        self.results_view.append("--- Window closed by user. Stopping worker... ---")
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        event.accept()
