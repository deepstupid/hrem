import torch
from models.hrm.act import HierarchicalReasoningModel_ACTV1

def test_hrm_without_memory():
    """Test that HRM works correctly without memory."""
    # Create a minimal config for testing
    config = {
        "batch_size": 2,
        "seq_len": 10,
        "vocab_size": 100,
        "num_puzzle_identifiers": 10,
        "hidden_size": 16,
        "expansion": 1.0,
        "num_heads": 1,
        "pos_encodings": "rope",
        "H_cycles": 1,
        "L_cycles": 1,
        "H_layers": 1,
        "L_layers": 1,
        "halt_max_steps": 3,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "puzzle_emb_ndim": 8
    }
    
    # Initialize model
    model = HierarchicalReasoningModel_ACTV1(config)
    
    # Create a mock batch
    batch = {
        "inputs": torch.randint(0, 100, (2, 10)),
        "puzzle_identifiers": torch.randint(0, 10, (2,))
    }
    
    # Test initial carry
    hrm_carry = model.initial_carry(batch)
    
    # Verify carry structure
    assert hasattr(hrm_carry, 'inner_carry'), "HRM carry should have inner_carry"
    assert hasattr(hrm_carry, 'steps'), "HRM carry should have steps"
    assert hasattr(hrm_carry, 'halted'), "HRM carry should have halted"
    assert hasattr(hrm_carry, 'current_data'), "HRM carry should have current_data"
    
    # Test forward pass
    new_hrm_carry, outputs = model(hrm_carry, batch)
    
    # Verify outputs structure
    assert "logits" in outputs, "Outputs should contain logits"
    assert "q_halt_logits" in outputs, "Outputs should contain q_halt_logits"
    assert "q_continue_logits" in outputs, "Outputs should contain q_continue_logits"
    
    print("HRM without memory test passed!")

if __name__ == "__main__":
    test_hrm_without_memory()