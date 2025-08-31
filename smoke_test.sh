#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Print each command to stdout before executing it.
set -x

# --- Install Dependencies ---
pip install -r requirements.txt

# --- Dataset Preparation ---
echo "Preparing dataset for successful run..."
python dataset/build_synthetic_dataset.py --output-dir data/synthetic-sort-smoke --num-samples 1 --task-type sort

echo "Preparing dataset for failure test..."
python dataset/build_synthetic_dataset.py --output-dir data/synthetic-duplicate-smoke --num-samples 1 --task-type duplicate

# --- HRM Training (Successful Run) ---
echo "Running a successful evaluation..."
python run.py evaluate \
    --challenge synthetic_sort \
    --models HRM \
    --smoke-test \
    --arch-overrides '{"forward_dtype": "float32", "hidden_size": 16, "H_layers": 1, "L_layers": 1, "puzzle_emb_ndim": 16}'

# --- Comparison with a Failing Run (to trigger report) ---
echo "Running a comparison with a non-existent model to trigger a failure insight..."
# We compare HRM (which should work) with a non-existent model "FakeModel"
# The engine should handle the missing model and generate a failure insight, thus creating a report.
python run.py evaluate \
    --challenge synthetic_duplicate \
    --models HRM --models FakeModel \
    --smoke-test

echo "Smoke test completed successfully."
