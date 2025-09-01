import json
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QGroupBox,
    QPushButton,
    QLabel,
    QCheckBox,
    QSpinBox,
    QMessageBox,
    QFileDialog,
)
from sc_engine.core.config_manager import ConfigManager
from sc_engine.core.challenge_registry import ChallengeRegistry
from sc_engine.core.schemas import ChallengeSchema


class InvestmentScreen(QWidget):
    """
    The 'Create Experiment' screen (Investment Phase). Guides the user through
    setting up a new scientific experiment.
    """
    experiment_started = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # --- Header ---
        header_label = QLabel("Configure Your Experiment")
        header_label.setObjectName("headerLabel")
        description_label = QLabel(
            "Define the parameters for your scientific investigation. "
            "Start by selecting a challenge, then configure the experiment type and algorithms."
        )
        description_label.setObjectName("descriptionLabel")
        main_layout.addWidget(header_label)
        main_layout.addWidget(description_label)

        # --- Step 1: Scientific Challenge ---
        challenge_group = QGroupBox("Step 1: Select Scientific Challenge")
        challenge_layout = QFormLayout(challenge_group)
        self.challenge_combo = QComboBox()
        self.challenge_combo.setToolTip("Select the scientific problem or dataset to address.")
        challenge_layout.addRow("Scientific Challenge:", self.challenge_combo)
        main_layout.addWidget(challenge_group)

        # --- Step 2: Experiment Type & Budget ---
        type_group = QGroupBox("Step 2: Define Experiment Type & Budget")
        type_layout = QFormLayout(type_group)
        self.run_type_combo = QComboBox()
        self.run_type_combo.addItems(["Comparison", "Optimization"])
        self.run_type_combo.setToolTip(
            "Select the type of experiment to run.\n"
            "- Comparison: A standard, sequential run comparing the final performance of selected algorithms.\n"
            "- Optimization: A run to find the best hyperparameters for a single selected algorithm."
        )
        self.patience_combo = QComboBox()
        self.patience_combo.addItem("Quick (~2 mins)", "low")
        self.patience_combo.addItem("Standard (~10 mins)", "medium")
        self.patience_combo.addItem("Thorough (~30 mins)", "high")
        self.patience_combo.setToolTip("Set the computational budget for the experiment.\nThis controls the approximate maximum runtime.")
        type_layout.addRow("Experiment Type:", self.run_type_combo)
        type_layout.addRow("Computational Budget:", self.patience_combo)
        main_layout.addWidget(type_group)

        # --- Step 3: Algorithm Configuration ---
        self.algorithm_group = QGroupBox("Step 3: Configure Algorithms")
        self.algorithm_layout = QVBoxLayout(self.algorithm_group)

        # --- Optimization Group (hidden by default) ---
        self.optimization_group = QWidget()
        optimization_layout = QFormLayout(self.optimization_group)
        self.model_to_optimize_combo = QComboBox()
        self.model_to_optimize_combo.setToolTip("Choose the single algorithm you want to optimize.")
        self.n_trials_spinbox = QSpinBox()
        self.n_trials_spinbox.setRange(1, 1000)
        self.n_trials_spinbox.setValue(10)
        self.n_trials_spinbox.setToolTip("Set the number of different hyperparameter sets to try.")
        optimization_layout.addRow("Algorithm to Optimize:", self.model_to_optimize_combo)
        optimization_layout.addRow("Number of Trials:", self.n_trials_spinbox)
        self.optimization_group.hide()

        # --- Algorithm Selection Group ---
        self.models_group = QWidget()
        self.models_layout = QVBoxLayout(self.models_group)
        self.models_layout.setContentsMargins(0, 0, 0, 0)

        self.algorithm_layout.addWidget(self.optimization_group)
        self.algorithm_layout.addWidget(self.models_group)
        main_layout.addWidget(self.algorithm_group)

        main_layout.addStretch()

        # --- Action Buttons ---
        action_button_layout = QHBoxLayout()
        self.load_button = QPushButton("📂 Load Config")
        self.load_button.setToolTip("Load an experiment configuration from a JSON file.")
        self.save_button = QPushButton("💾 Save Config")
        self.save_button.setToolTip("Save the current configuration to a JSON file.")
        self.start_button = QPushButton("🚀 Start Experiment")
        self.start_button.setToolTip("Start the experiment with the current configuration.")

        action_button_layout.addWidget(self.load_button)
        action_button_layout.addWidget(self.save_button)
        action_button_layout.addStretch()
        action_button_layout.addWidget(self.start_button)
        main_layout.addLayout(action_button_layout)

        # --- Init & Connect ---
        self.challenge_registry = self._initialize_challenge_registry()
        self._populate_challenges()
        self.run_type_combo.currentTextChanged.connect(self._on_run_type_changed)
        self.challenge_combo.currentIndexChanged.connect(self._on_challenge_selected)
        self.start_button.clicked.connect(self._gather_config_and_emit)
        self.save_button.clicked.connect(self._save_configuration)
        self.load_button.clicked.connect(self._load_configuration)
        self._on_run_type_changed(self.run_type_combo.currentText())

    def reset_ui(self):
        """Resets the UI to its default state, called when the screen is shown."""
        self.start_button.setEnabled(True)
        self.start_button.setText("🚀 Start Experiment")
        # You can add other resets here if needed, e.g., resetting selections.
        # For now, we keep user's previous selections as it can be convenient.

    def _save_configuration(self):
        """Saves the current experiment configuration to a JSON file."""
        challenge: ChallengeSchema = self.challenge_combo.currentData()
        if not challenge:
            self._show_error("Cannot save config without a selected challenge.")
            return

        # Create a list of all model checkboxes, even if they are not visible
        all_models = []
        if self.models_layout.count() > 0:
            all_models.extend([
                self.models_layout.itemAt(i).widget().text()
                for i in range(self.models_layout.count())
                if isinstance(self.models_layout.itemAt(i).widget(), QCheckBox)
            ])
        else: # If in optimization mode, get models from the combo box
            all_models.extend([self.model_to_optimize_combo.itemText(i) for i in range(self.model_to_optimize_combo.count())])


        config = {
            "run_type": self.run_type_combo.currentText(),
            "challenge_name": challenge.name,
            "patience_level": self.patience_combo.currentData(),
            "model_to_optimize": self.model_to_optimize_combo.currentText(),
            "n_trials": self.n_trials_spinbox.value(),
            "selected_models": all_models,
            "checked_models": [
                self.models_layout.itemAt(i).widget().text()
                for i in range(self.models_layout.count())
                if isinstance(self.models_layout.itemAt(i).widget(), QCheckBox) and self.models_layout.itemAt(i).widget().isChecked()
            ]
        }

        filePath, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json)")
        if not filePath:
            return

        try:
            with open(filePath, 'w') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            self._show_error(f"Failed to save configuration: {e}")

    def _load_configuration(self):
        """Loads an experiment configuration from a JSON file."""
        filePath, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")
        if not filePath:
            return

        try:
            with open(filePath, 'r') as f:
                config = json.load(f)
            self._apply_configuration(config)
        except (IOError, json.JSONDecodeError, KeyError) as e:
            self._show_error(f"Failed to load or apply configuration: {e}")

    def _apply_configuration(self, config: dict):
        """Applies a loaded configuration dictionary to the UI."""
        # Set Challenge
        challenge_name = config["challenge_name"]
        challenge_index = self.challenge_combo.findText(challenge_name)
        if challenge_index == -1:
            raise KeyError(f"Challenge '{challenge_name}' not found.")
        self.challenge_combo.setCurrentIndex(challenge_index)

        # Set simple dropdowns and values
        run_type = config.get("run_type", "Comparison")
        self.run_type_combo.setCurrentText(run_type)

        patience_value = config.get("patience_level", "medium")
        patience_map = {"low": "low", "medium": "medium", "high": "high"}
        patience_data = patience_map.get(patience_value, "medium")
        patience_index = self.patience_combo.findData(patience_data)
        if patience_index != -1:
            self.patience_combo.setCurrentIndex(patience_index)

        # Set optimization params
        self.n_trials_spinbox.setValue(config.get("n_trials", 10))
        model_to_optimize = config.get("model_to_optimize")
        if model_to_optimize:
            model_to_optimize_index = self.model_to_optimize_combo.findText(model_to_optimize)
            if model_to_optimize_index != -1:
                self.model_to_optimize_combo.setCurrentIndex(model_to_optimize_index)

        # Set selected models for comparison
        checked_models = config.get("checked_models", [])
        for i in range(self.models_layout.count()):
            widget = self.models_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(widget.text() in checked_models)


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
                # Trigger update for the first challenge in the list
                self._on_challenge_selected(0)
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
        self.algorithm_group.setTitle(
            "Step 3: Configure Optimization" if is_optimization else "Step 3: Select Algorithms for Comparison"
        )


    def _update_models_for_challenge(self, challenge: ChallengeSchema):
        # Clear previous model checkboxes
        while self.models_layout.count():
            child = self.models_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.model_to_optimize_combo.clear()

        # Populate with new models from the selected challenge
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
            "patience_level": self.patience_combo.currentData(),
        }

        run_type = config["run_type"]
        if run_type == "optimization":
            model_to_optimize = self.model_to_optimize_combo.currentText()
            if not model_to_optimize:
                self._show_error("Please select an algorithm to optimize.")
                return
            config["model_to_optimize"] = model_to_optimize
            config["n_trials"] = self.n_trials_spinbox.value()
        else: # Comparison
            selected_models = [
                self.models_layout.itemAt(i).widget().text()
                for i in range(self.models_layout.count())
                if isinstance(self.models_layout.itemAt(i).widget(), QCheckBox) and self.models_layout.itemAt(i).widget().isChecked()
            ]
            if not selected_models:
                self._show_error("Please select at least one algorithm for comparison.")
                return
            config["models"] = selected_models

        self.start_button.setEnabled(False)
        self.start_button.setText("🚀 Experiment Running...")
        self.experiment_started.emit(config)

    def _show_error(self, message: str):
        """Displays a critical error message in a modal dialog box."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Configuration Error")
        msg_box.exec()
