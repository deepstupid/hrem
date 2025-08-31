from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QGroupBox,
    QPushButton,
    QLabel,
    QCheckBox,
    QSpinBox,
    QStyle,
)
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry
from sc_engine.core.schemas import ChallengeSchema


class InvestmentScreen(QWidget):
    """The 'Create Experiment' screen (Investment Phase)."""
    experiment_started = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._apply_styles()

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # --- Configuration Group ---
        config_group = QGroupBox("🔬 Experiment Configuration")
        main_layout.addWidget(config_group)
        form_layout = QFormLayout()
        config_group.setLayout(form_layout)

        # 1. Experiment Type, 2. Challenge, 3. Patience
        self.run_type_combo = QComboBox()
        self.run_type_combo.addItems(["Comparison", "Optimization", "Demo"])
        self.challenge_combo = QComboBox()
        self.patience_combo = QComboBox()
        self.patience_combo.addItems(["Low", "Medium", "High"])
        form_layout.addRow("Experiment Type:", self.run_type_combo)
        form_layout.addRow("Scientific Challenge:", self.challenge_combo)
        form_layout.addRow("Patience Level:", self.patience_combo)

        # --- Optimization Group (hidden) ---
        self.optimization_group = QGroupBox("⚙️ Optimization Parameters")
        optimization_layout = QFormLayout()
        self.optimization_group.setLayout(optimization_layout)
        main_layout.addWidget(self.optimization_group)
        self.model_to_optimize_combo = QComboBox()
        self.n_trials_spinbox = QSpinBox()
        self.n_trials_spinbox.setRange(1, 1000)
        self.n_trials_spinbox.setValue(10)
        optimization_layout.addRow("Model to Optimize:", self.model_to_optimize_combo)
        optimization_layout.addRow("Number of Trials:", self.n_trials_spinbox)
        self.optimization_group.hide()

        # --- Algorithm Selection Group ---
        self.models_group = QGroupBox("🤖 Algorithm Selection")
        self.models_layout = QVBoxLayout()
        self.models_group.setLayout(self.models_layout)
        self.models_group.setToolTip("Select algorithms to run.")
        main_layout.addWidget(self.models_group)

        # --- Start Button ---
        self.start_button = QPushButton("🚀 Start Experiment")
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.start_button.setIcon(icon)
        main_layout.addWidget(self.start_button)

        main_layout.addStretch()

        # --- Init & Connect ---
        self.challenge_registry = self._initialize_challenge_registry()
        self._populate_challenges()
        self.run_type_combo.currentTextChanged.connect(self._on_run_type_changed)
        self.challenge_combo.currentIndexChanged.connect(self._on_challenge_selected)
        self.start_button.clicked.connect(self._gather_config_and_emit)
        self._on_run_type_changed(self.run_type_combo.currentText())

    def _apply_styles(self):
        self.setStyleSheet("""
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
            QPushButton {
                background-color: #007BFF;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #9E9E9E;
            }
            QComboBox, QSpinBox {
                padding: 4px;
            }
        """)

    def _initialize_challenge_registry(self):
        try:
            config_manager = ConfigManager()
            return ChallengeRegistry(config_manager)
        except Exception as e:
            self._show_error(f"Failed to initialize Challenge Registry: {e}")
            return None

    def _populate_challenges(self):
        if not self.challenge_registry: return
        try:
            challenges = self.challenge_registry.get_all_challenges()
            for challenge in challenges:
                self.challenge_combo.addItem(challenge.name, userData=challenge)
            if challenges:
                self._update_models_for_challenge(challenges[0])
        except Exception as e:
            self._show_error(f"Error loading challenges: {e}")

    def _on_challenge_selected(self, index: int):
        challenge: ChallengeSchema = self.challenge_combo.itemData(index)
        if challenge:
            self._update_models_for_challenge(challenge)

    def _on_run_type_changed(self, run_type: str):
        is_optimization = (run_type == "Optimization")
        self.optimization_group.setVisible(is_optimization)
        self.models_group.setVisible(not is_optimization)

    def _update_models_for_challenge(self, challenge: ChallengeSchema):
        while self.models_layout.count():
            child = self.models_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.model_to_optimize_combo.clear()

        for model_name in challenge.models:
            checkbox = QCheckBox(model_name)
            checkbox.setChecked(True)
            self.models_layout.addWidget(checkbox)
            self.model_to_optimize_combo.addItem(model_name)

    def _gather_config_and_emit(self):
        challenge: ChallengeSchema = self.challenge_combo.currentData()
        if not challenge:
            self._show_error("Please select a valid challenge.")
            return

        config = {
            "run_type": self.run_type_combo.currentText().lower(),
            "challenge_id": challenge.id,
            "patience_level": self.patience_combo.currentText().lower(),
        }

        if config["run_type"] == "optimization":
            config["model_to_optimize"] = self.model_to_optimize_combo.currentText()
            config["n_trials"] = self.n_trials_spinbox.value()
        else:
            selected_models = [
                self.models_layout.itemAt(i).widget().text()
                for i in range(self.models_layout.count())
                if isinstance(self.models_layout.itemAt(i).widget(), QCheckBox) and self.models_layout.itemAt(i).widget().isChecked()
            ]
            if not selected_models:
                self._show_error("Please select at least one algorithm.")
                return
            config["models"] = selected_models

        self.start_button.setEnabled(False)
        self.start_button.setText("🚀 Experiment Running...")
        self.experiment_started.emit(config)

    def enable_start_button(self):
        """Allows the main window to re-enable the button after a run."""
        self.start_button.setEnabled(True)
        self.start_button.setText("🚀 Start Experiment")

    def _show_error(self, message: str):
        error_label = QLabel(f"⚠️ {message}")
        error_label.setStyleSheet("color: #D32F2F; font-weight: bold;")
        self.layout().insertWidget(0, error_label)
