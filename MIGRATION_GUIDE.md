# Migration Guide: HRM/HREM Demo System Refactoring

This document explains the recent refactoring of the HRM/HREM demo system and provides guidance for developers adapting to the new structure.

## Overview of Changes

The refactoring addresses several key issues:
1. Code duplication across timing, model running, and configuration functions
2. Scattered functionality that made maintenance difficult
3. Inconsistent interfaces across similar components

## Key Changes

### 1. Timing Functions Consolidation
**Before**: Timing functions were scattered across `demo_timing.py`, `demo_metrics.py`, and `adaptive_demo_runner.py`
**After**: Unified in `demo_timing_utils.py` with `TimingManager` and `TimingContext` classes

### 2. Model Running Functions Consolidation
**Before**: Model running functions were duplicated in multiple files
**After**: Unified in `demo_model_runner.py`

### 3. Configuration Management Consolidation
**Before**: Configuration handling was spread across multiple files
**After**: Centralized in `demo_config_manager.py`

## Migration Path

### For New Development
Use the new unified modules directly:
- `demo_timing_utils.TimingManager` for timing functionality
- `demo_model_runner` for model execution functions
- `demo_config_manager.ConfigManager` for configuration handling

### For Existing Code
Existing imports and interfaces remain functional for backward compatibility:
- `demo_timing.TimingCollector` still works
- `demo_config.load_ui_config()` still works
- `demo_shared.run_trial()` still works

## Benefits of the New Structure

1. **Easier Maintenance**: Changes only need to be made in one place
2. **Better Code Organization**: Related functionality is grouped together
3. **Improved Reusability**: Components can be easily reused in new features
4. **Reduced Bugs**: Eliminates inconsistencies from duplicated code

## Testing Your Code

After migrating, run the verification tests to ensure everything works correctly:

```bash
python test_refactoring.py
python comprehensive_test.py
```