#!/bin/bash
# Test script for the unified demo runner

echo "Testing Unified Demo Runner"
echo "=========================="

# Run a quick smoke test with basic settings
echo "Running smoke test with HRM and HREM models..."
python run.py demo --smoke-test --patience low --models HRM --models HREM --dataset synthetic --task reverse --n-trials 1

echo ""
echo "Test completed successfully!"