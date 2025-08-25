#!/usr/bin/env python3
"""
Test script to verify the enhanced HREM implementation.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import hrm_system
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hrm_system.algorithms.enhanced_hrem import EnhancedHREMAlgorithm
from hrm_system.config import ModelConfig, TrainingConfig

def test_enhanced_hrem_initialization():
    """Test that we can initialize the EnhancedHREMAlgorithm."""
    print("Testing EnhancedHREMAlgorithm initialization...")
    
    # Create a minimal model config
    model_config = ModelConfig(
        name="EnhancedHREM",
        algorithm_class="hrm_system.algorithms.enhanced_hrem.EnhancedHREMAlgorithm",
        base_arch_config="enhanced_hrem_v1"
    )
    
    # Create a minimal training config
    training_config = TrainingConfig(
        epochs=1,
        global_batch_size=2,
        lr=1e-4,
        puzzle_emb_lr=1e-3
    )
    
    # Try to instantiate the algorithm
    try:
        algorithm = EnhancedHREMAlgorithm(model_config, training_config)
        print("✓ EnhancedHREMAlgorithm instantiated successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to instantiate EnhancedHREMAlgorithm: {e}")
        return False

def test_config_files():
    """Test that our config files exist and are valid."""
    print("Testing config files...")
    
    config_files = [
        "config/arch/enhanced_hrem_v1.yaml",
        "config/cfg_enhanced_pretrain.yaml",
        "config/enhanced_hrem_search_space.yaml"
    ]
    
    all_passed = True
    for config_file in config_files:
        path = project_root / config_file
        if path.exists():
            print(f"✓ {config_file} exists")
        else:
            print(f"✗ {config_file} not found")
            all_passed = False
            
    return all_passed

def main():
    """Run all tests."""
    print("Running Enhanced HREM tests...\n")
    
    tests = [
        test_config_files,
        test_enhanced_hrem_initialization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} tests passed!")
        return 0
    else:
        print(f"✗ {passed}/{total} tests passed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())