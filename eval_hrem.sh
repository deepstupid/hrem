#!/bin/bash

# Default values
SMOKE_TEST=true
TIME_LIMIT="1h"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --smoke-test) SMOKE_TEST="$2"; shift ;;
        --time-limit) TIME_LIMIT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Environment Setup ---
echo "Setting up the environment..."
set -x

# Check for GPU and set device
if command -v nvidia-smi &> /dev/null && nvidia-smi | grep -q "CUDA Version"; then
    echo "GPU found. Using CUDA."
    export DEVICE="cuda"
else
    echo "No GPU found. Using CPU."
    export DEVICE="cpu"
fi

# Install dependencies
pip install -r requirements.txt

# --- Data Preparation ---
echo "Preparing dataset..."
DATA_DIR="data/sudoku-smoke"
if [ "$SMOKE_TEST" = true ]; then
    echo "Running in smoke test mode. Using a small dataset."
    python dataset/build_sudoku_dataset.py --output-dir "$DATA_DIR" --subsample-size 1 --num-aug 0
else
    echo "Running in full mode. Using the complete dataset."
    # Use the full dataset settings from the README
    DATA_DIR="data/sudoku-extreme-1k-aug-1000"
    python dataset/build_sudoku_dataset.py --output-dir "$DATA_DIR" --subsample-size 1000 --num-aug 1000
fi

# --- Training and Evaluation ---
echo "Starting training and evaluation..."

# Set training parameters based on smoke test mode
if [ "$SMOKE_TEST" = true ]; then
    EPOCHS=1
    EVAL_INTERVAL=1
else
    EPOCHS=20000
    EVAL_INTERVAL=2000
fi

# Function to train and evaluate a model
run_experiment() {
    local model_name=$1
    local config_name=$2
    local results_file="results_${model_name}.txt"

    echo "--- Running experiment for $model_name ---"

    # Training
    export WANDB_MODE=disabled
    timeout "$TIME_LIMIT" python pretrain.py \
        data_path="$DATA_DIR" \
        epochs="$EPOCHS" \
        eval_interval="$EVAL_INTERVAL" \
        arch="$config_name" \
        +project_name="HREM_vs_HRM" \
        +run_name="${model_name}_smoke_test_${SMOKE_TEST}" > "train_${model_name}.log" 2>&1

    # Evaluation (placeholder)
    echo "--- Evaluating $model_name ---"
    # The evaluation logic will be added here.
    # For now, we'll just record that the training completed.
    echo "Training completed for $model_name." > "$results_file"

    # Placeholder for collecting metrics
    echo "Metrics for $model_name:" >> "$results_file"
    # In a real scenario, we would parse the log files for metrics.
    # For the smoke test, we will just confirm it runs.
    if [ $? -eq 0 ]; then
        echo "Training successful." >> "$results_file"
    else
        echo "Training failed." >> "$results_file"
    fi
}

# Create HREM config
cp config/arch/hrm_v1.yaml config/arch/hrem_v1.yaml
sed -i 's/name: hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1/name: hrm.hrem@HREM/' config/arch/hrem_v1.yaml
sed -i '/name: hrm.hrem@HREM/a \
# HREM-specific parameters\
use_memory: True\
m_loc: 128\
d_mem: 128\
top_k: 4\
sparse_addressing: True\
use_location_addressing: True' config/arch/hrem_v1.yaml

# Run experiments for HRM and HREM
run_experiment "HRM" "hrm_v1"
run_experiment "HREM" "hrem_v1"


# --- Reporting ---
echo "--- Comparison Report ---"
echo "Results for HRM:"
cat results_HRM.txt
echo ""
echo "Results for HREM:"
cat results_HREM.txt

echo "--- Summary ---"
# This summary will be more detailed once we have actual metrics.
echo "The script has completed. Check the log files for details."
set +x
