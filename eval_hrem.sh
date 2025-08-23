#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Print each command to stdout before executing it.
set -x

# Default values
SMOKE_TEST=true
TIME_LIMIT="1h"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --smoke_test) SMOKE_TEST="$2"; shift ;;
        --time_limit) TIME_LIMIT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Automatically detect if a GPU is present
if command -v nvidia-smi &> /dev/null
then
    DEVICE="cuda"
    FORWARD_DTYPE="bfloat16"
else
    DEVICE="cpu"
    FORWARD_DTYPE="float32"
fi

echo "Running on device: $DEVICE"

# --- Install Dependencies ---
echo "Installing dependencies..."
pip install -r requirements.txt

# --- Dataset Preparation ---
echo "Preparing dataset..."
if [ "$SMOKE_TEST" = "true" ]; then
    # Use a small dataset for the smoke test
    echo "Using small dataset for smoke test."
    python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 10 --num-aug 10
else
    # Use the full dataset for a complete run
    echo "Using full dataset."
    python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000
fi
DATA_PATH="data/sudoku-extreme-1k-aug-1000"

# --- HRM Training and Evaluation ---
echo "Training and evaluating HRM..."
HRM_CHECKPOINT_PATH="checkpoints/HRM"
HRM_LOG_PATH="hrm_metrics.json"

HRM_TRAIN_CMD="python pretrain.py \
    data_path=$DATA_PATH \
    arch=hrm_v1 \
    +arch.forward_dtype=$FORWARD_DTYPE \
    +checkpoint_path=$HRM_CHECKPOINT_PATH \
    +log_path=$HRM_LOG_PATH \
    +smoke_test=$SMOKE_TEST"

if [ "$SMOKE_TEST" = "true" ]; then
    HRM_TRAIN_CMD="$HRM_TRAIN_CMD \
        arch.hidden_size=16 \
        arch.H_layers=1 \
        arch.L_layers=1 \
        arch.puzzle_emb_ndim=16 \
        arch.num_heads=1 \
        arch.expansion=1.0 \
        global_batch_size=1 \
        epochs=1 \
        eval_interval=1 \
        checkpoint_every_eval=True"
fi

# Run the training command
$HRM_TRAIN_CMD

echo "Evaluating HRM..."
HRM_LATEST_CHECKPOINT=$(ls -t $HRM_CHECKPOINT_PATH/step_* | head -1)
python evaluate.py \
    checkpoint="$HRM_LATEST_CHECKPOINT"

# --- HREM Training and Evaluation ---
echo "Training and evaluating HREM..."
HREM_CHECKPOINT_PATH="checkpoints/HREM"
HREM_LOG_PATH="hrem_metrics.json"

# Create HREM config
if [ "$SMOKE_TEST" = "true" ]; then
    cat > config/arch/hrem_v1.yaml <<EOL
name: hrm.hrem@HREM
use_memory: true
hidden_size: 16
expansion: 1.0
num_heads: 1
pos_encodings: "rope"
puzzle_emb_ndim: 16
H_layers: 1
L_layers: 1
H_cycles: 1
L_cycles: 12
halt_max_steps: 12
halt_exploration_prob: 0.0
forward_dtype: "$FORWARD_DTYPE"
m_loc: 128
d_mem: 128
top_k: 4
sparse_addressing: true
use_location_addressing: true

loss:
  name: losses@ACTLossHead
  loss_type: stablemax_cross_entropy
  q_learning_gamma: 0.9
EOL
else
    cat > config/arch/hrem_v1.yaml <<EOL
name: hrm.hrem@HREM
use_memory: true
hidden_size: 512
expansion: 2.0
num_heads: 8
pos_encodings: "rope"
puzzle_emb_ndim: 512
H_layers: 12
L_layers: 12
H_cycles: 1
L_cycles: 12
halt_max_steps: 12
halt_exploration_prob: 0.0
forward_dtype: "$FORWARD_DTYPE"
m_loc: 128
d_mem: 128
top_k: 4
sparse_addressing: true
use_location_addressing: true

loss:
  name: losses@ACTLossHead
  loss_type: stablemax_cross_entropy
  q_learning_gamma: 0.9
EOL
fi

HREM_TRAIN_CMD="python pretrain.py \
    data_path=$DATA_PATH \
    arch=hrem_v1 \
    +checkpoint_path=$HREM_CHECKPOINT_PATH \
    +log_path=$HREM_LOG_PATH \
    +smoke_test=$SMOKE_TEST"

if [ "$SMOKE_TEST" = "true" ]; then
    HREM_TRAIN_CMD="$HREM_TRAIN_CMD \
        global_batch_size=1 \
        epochs=1 \
        eval_interval=1 \
        checkpoint_every_eval=True"
fi

# Run the training command
$HREM_TRAIN_CMD

echo "Evaluating HREM..."
HREM_LATEST_CHECKPOINT=$(ls -t $HREM_CHECKPOINT_PATH/step_* | head -1)
python evaluate.py \
    checkpoint="$HREM_LATEST_CHECKPOINT"


# --- Reporting ---
echo "Generating report..."

echo "HREM vs HRM Performance Summary:" > comparison_report.md
echo "---------------------------------" >> comparison_report.md
echo "HRM Metrics:" >> comparison_report.md
cat $HRM_LOG_PATH >> comparison_report.md
echo "" >> comparison_report.md
echo "HREM Metrics:" >> comparison_report.md
cat $HREM_LOG_PATH >> comparison_report.md
echo "---------------------------------" >> comparison_report.md

echo "Evaluation complete. See comparison_report.md for results."
