#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Print each command to stdout before executing it.
set -x

DATA_PATH="data/sudoku-extreme-1k-aug-1000"

# --- HRM Training ---
echo "Training HRM for a single step..."
python pretrain.py \
    data_path=$DATA_PATH \
    arch=hrm_v1 \
    +arch.forward_dtype=float32 \
    arch.hidden_size=8 \
    arch.H_layers=1 \
    arch.L_layers=1 \
    arch.puzzle_emb_ndim=8 \
    arch.num_heads=1 \
    arch.expansion=1.0 \
    global_batch_size=1 \
    +smoke_test=true \
    epochs=1 \
    eval_interval=1 \
    checkpoint_every_eval=False

echo "Barebones test completed successfully."
