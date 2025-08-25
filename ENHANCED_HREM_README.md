# Enhanced HREM Implementation

This directory contains enhancements to the HREM (Hierarchical Reasoning with External Memory) model to improve its training performance and competitiveness with the HRM model.

## Key Enhancements

1. **Enhanced HREM Model** (`models/hrm/enhanced_hrem.py`):
   - **Increased Memory Capacity**: Expanded `d_mem` from 128 to 256 for better information storage
   - **Improved Memory Retrieval**: Increased `top_k` from 4 to 6 for better memory access
   - **Additional Processing Layers**: Added memory read/write MLP layers for better feature extraction
   - **Larger Model Capacity**: Increased `L_layers` from 2 to 4 for enhanced computation
   - **Fixed Dimension Issues**: Corrected memory processing layer dimensions for compatibility

2. **Improved Training Algorithm** (`hrm_system/algorithms/enhanced_hrem.py`):
   - **Better Optimizer**: Replaced Adam with AdamW for superior weight decay handling
   - **Numerical Stability**: Added lower epsilon (1e-5) for better numerical stability
   - **Maintained Compatibility**: Preserved all existing functionality while adding enhancements

3. **Enhanced Training Utilities** (`hrm_system/algorithms/enhanced_utils.py`):
   - **Faster Data Loading**: Increased prefetch factor from 8 to 16 for better throughput
   - **More Workers**: Increased data loading workers from 1 to 2 for parallelism
   - **Flexible Scheduling**: Added linear learning rate scheduling option for stability
   - **Mixed Precision Support**: Implemented automatic mixed precision (AMP) training

4. **Dynamic Reporting System** (`hrm_system/reporting.py`):
   - **N-Model Support**: Completely rewritten to handle any number of models dynamically
   - **Flexible Comparison**: No longer hardcoded to just HRM and HREM
   - **Enhanced Metrics**: Better organization and presentation of metrics for multiple models
   - **Improved Parameter Display**: Shows parameters for all HREM-like models in the comparison

5. **Optimized Architecture Configuration** (`config/arch/enhanced_hrem_v1.yaml`):
   - **Larger Model**: Increased `hidden_size` to 256 and `num_heads` to 8
   - **Enhanced Memory**: Doubled `d_mem` to 256 and increased `top_k` to 6
   - **Deeper Processing**: Increased `L_layers` to 4 for more computation
   - **Additional Parameters**: Added `memory_read_layers` and `memory_write_layers`

6. **Improved Training Configuration** (`config/cfg_enhanced_pretrain.yaml`):
   - **Larger Batches**: Increased `global_batch_size` to 1024 for better stability
   - **Higher Learning Rates**: Set `lr` to 3e-4 and `puzzle_emb_lr` to 3e-3 for faster convergence
   - **More Frequent Evaluation**: Reduced `eval_interval` to 5000 steps
   - **Better Regularization**: Lowered weight decay to 0.01 for better generalization

7. **Extended Hyperparameter Search Space** (`config/enhanced_hrem_search_space.yaml`):
   - **Expanded Ranges**: Broader parameter ranges for optimal configurations
   - **New Parameters**: Added memory read/write layer parameters
   - **Higher Values**: Increased maximum values for key parameters

## Usage

To run an evaluation of the enhanced HREM model:

```bash
python run_enhanced_hrem.py --dataset arc --num-aug 100 --n-runs 3
```

To compare HRM, HREM, and EnhancedHREM:

```bash
python run_enhanced_hrem.py --dataset arc --num-aug 100 --n-runs 3 --compare
```

## Performance Improvements

The enhanced HREM implementation provides several performance improvements over the baseline HREM:

1. **Increased Model Capacity**: Larger hidden size and memory dimensions allow for more complex reasoning
2. **Better Memory Retrieval**: Higher top-k values and additional processing layers improve memory access
3. **Faster Training**: AMP and improved optimizers reduce training time
4. **Better Convergence**: Linear scheduling and higher learning rates lead to faster convergence
5. **Enhanced Stability**: Gradient clipping and better hyperparameter tuning improve training stability
6. **Fixed Dimension Issues**: Corrected dimension mismatches for better compatibility
7. **Dynamic Reporting**: Supports comparison of any number of models with flexible metrics display

These enhancements make the HREM model more competitive with the HRM model while maintaining its unique external memory capabilities.