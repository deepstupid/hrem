import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .act import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Carry
from .external_memory import ExternalMemory


class HREM(HierarchicalReasoningModel_ACTV1):
    """
    Hierarchical Recurrent-External Memory (HREM) model.

    This model extends the Hierarchical Reasoning Model (HRM) by inheriting
    from it and augmenting its forward pass with an external DNC/NTM memory
    module.
    """

    def __init__(self, config_dict: dict):
        # Pass the config to the parent HRM class
        super().__init__(config_dict)

        self.use_memory = getattr(self.config, 'use_memory', False)
        if not self.use_memory:
            # If not using memory, this class is identical to HRM.
            return

        dtype = getattr(torch, self.config.forward_dtype)

        # Memory-specific initializations, using the unified self.config
        self.memory = ExternalMemory(
            d_model=self.config.hidden_size,
            m_loc=getattr(self.config, 'm_loc', 128),
            d_mem=getattr(self.config, 'd_mem', 128),
            top_k=getattr(self.config, 'top_k', 4),
            sparse_addressing=getattr(self.config, 'sparse_addressing', True),
            use_location_addressing=getattr(self.config, 'use_location_addressing', True),
            forward_dtype=self.config.forward_dtype,
        )
        self.memory_readout_proj = nn.Linear(
            getattr(self.config, 'd_mem', 128), self.config.hidden_size, dtype=dtype
        )

    def initial_carry(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict]:
        """Initializes the carry state for both HRM and the external memory."""
        # Get the standard HRM carry from the parent
        hrm_carry = super().initial_carry(batch)

        mem_states = {}
        if self.use_memory:
            # Initialize memory states
            M, mem_states = self.memory.init_memory(
                batch["inputs"].size(0), batch["inputs"].device
            )
            mem_states["M"] = M

        return hrm_carry, mem_states

    def forward(
        self,
        carry: Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict], Dict]:
        """
        Overrides the HRM forward pass to inject memory operations.
        """
        if not getattr(self, 'use_memory', False):
            # If no memory, just call the parent's forward pass.
            # We need to adjust the carry format.
            hrm_carry, _ = carry
            new_hrm_carry, outputs = super().forward(hrm_carry, batch)
            return (new_hrm_carry, {}), outputs

        hrm_carry, mem_states = carry

        # Extract current hidden state before any modifications
        # Use the current z_H from the carry to generate the memory interface vector
        # This is more efficient than resetting and avoids potential inconsistencies
        current_z_H = hrm_carry.inner_carry.z_H
        z_H_summary = current_z_H.mean(dim=1)

        # Run the memory controller
        M_prev = mem_states["M"]
        prev_mem_states = {k: v for k, v in mem_states.items() if k != "M"}
        M, r, new_mem_states_inner = self.memory(z_H_summary, M_prev, prev_mem_states)

        # Update memory states for the next step
        mem_states.update(new_mem_states_inner)
        mem_states["M"] = M

        # Project readout to be injected into the HRM
        memory_readout = self.memory_readout_proj(r).unsqueeze(1)

        # Call the parent's forward method, passing the memory readout.
        # The parent class will handle the entire ACT loop.
        new_hrm_carry, outputs = super().forward(
            hrm_carry, batch, memory_readout=memory_readout
        )

        # Package the new carries and outputs
        return (new_hrm_carry, mem_states), outputs
