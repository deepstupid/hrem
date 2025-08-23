#!/bin/bash

# Default values
SMOKE_TEST=true

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --smoke-test) SMOKE_TEST="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Environment Setup ---
echo "Setting up the environment..."
set -x

# Run the evaluation script
if [ "$SMOKE_TEST" = true ]; then
    timeout 1h python run_evaluation.py --smoke-test
else
    timeout 1h python run_evaluation.py
fi

set +x

echo "Evaluation complete. See comparison_report.md for results."
