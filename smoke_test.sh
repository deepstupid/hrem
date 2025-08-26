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
python run.py evaluate \
    --dataset synthetic \
    --models HRM \
    --smoke-test \
    --arch-overrides '{"forward_dtype": "float32", "hidden_size": 16, "H_layers": 1, "L_layers": 1, "puzzle_emb_ndim": 16}'

echo "Smoke test completed successfully."
