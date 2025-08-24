#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Print each command to stdout before executing it.
set -x

# --- Install Dependencies ---
pip install -r requirements.txt

# --- Dataset Preparation ---
echo "Preparing dataset..."
python dataset/build_synthetic_dataset.py --output-dir data/synthetic-smoke --num-samples 1 --task-type copy
DATA_PATH="data/synthetic-smoke"

# --- HRM Training ---
echo "Training HRM..."
HRM_CHECKPOINT_PATH="checkpoints/smoke_test/HRM"
HRM_LOG_PATH="smoke_test_hrm_metrics.json"
python pretrain.py \
    data_path=$DATA_PATH \
    arch=hrm_v1 \
    +arch.forward_dtype=float32 \
    arch.hidden_size=16 \
    arch.H_layers=1 \
    arch.L_layers=1 \
    arch.puzzle_emb_ndim=16 \
    global_batch_size=1 \
    +checkpoint_path=$HRM_CHECKPOINT_PATH \
    +log_path=$HRM_LOG_PATH \
    +smoke_test=true \
    epochs=1 \
    eval_interval=1

echo "Smoke test completed successfully."
