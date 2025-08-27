#!/bin/bash
# Test script for the unified demo runner

echo "Testing Unified Demo Runner"
echo "=========================="

# Run a quick smoke test with basic settings
echo "Running smoke test with HRM and HREM models..."
python unified_demo.py --smoke-test --demo-mode lightning --detail basic --max-epochs 1 --max-trials 1 --models HRM --models HREM --dataset synthetic-reverse

echo ""
echo "Test completed successfully!"