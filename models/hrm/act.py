from typing import Tuple, List, Dict, Optional
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from models.hrm.common import (
    HierarchicalReasoningModel_ACTV1InnerCarry,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1Config,
)
from models.hrm.reasoning_block import HierarchicalReasoningModel_ACTV1Block, HierarchicalReasoningModel_ACTV1ReasoningModule


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[
            HierarchicalReasoningModel_ACTV1Block(
                hidden_size=self.config.hidden_size,
                expansion=self.config.expansion,
                num_heads=self.config.num_heads,
                rms_norm_eps=self.config.rms_norm_eps,
            ) for _i in range(self.config.H_layers)
        ])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[
            HierarchicalReasoningModel_ACTV1Block(
                hidden_size=self.config.hidden_size,
                expansion=self.config.expansion,
                num_heads=self.config.num_heads,
                rms_norm_eps=self.config.rms_norm_eps,
            ) for _i in range(self.config.L_layers)
        ])
        
        # Initial states
        self.register_buffer('H_init', trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.register_buffer('L_init', trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
        memory_readout: Optional[torch.Tensor] = None,
    ) -> Tuple[
        HierarchicalReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )

        if memory_readout is not None:
            # Add memory readout to input embeddings
            # Ensure proper broadcasting if dimensions don't match
            if memory_readout.shape[1] != input_embeddings.shape[1]:
                # Pad or slice memory_readout to match sequence length
                if memory_readout.shape[1] < input_embeddings.shape[1]:
                    # Pad
                    pad_len = input_embeddings.shape[1] - memory_readout.shape[1]
                    memory_readout = F.pad(memory_readout, (0, 0, 0, pad_len), "constant", 0)
                else:
                    # Slice
                    memory_readout = memory_readout[:, :input_embeddings.shape[1], :]
            input_embeddings = input_embeddings + memory_readout

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L

        with torch.no_grad():
            # Unroll all but the last H-cycle
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.H_level(z_H, z_L, **seq_info)

            # For the last H-cycle, unroll all but the last L-step
            if self.config.H_cycles > 0:
                for _ in range(self.config.L_cycles - 1):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

        # 1-step grad for the last L and H steps
        if self.config.H_cycles > 0:
            if self.config.L_cycles > 0:
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head for current state
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        # Create new carry with detached state
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach()
        )

        # Q head for NEXT state (no gradients)
        with torch.no_grad():
            next_q_logits = self.q_head(new_carry.z_H[:, 0]).to(torch.float32)

        return (
            new_carry,
            output,
            (q_logits[..., 0], q_logits[..., 1]),
            (next_q_logits[..., 0], next_q_logits[..., 1]),
        )


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
        memory_readout: Optional[torch.Tensor] = None,
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        (
            new_inner_carry,
            logits,
            (q_halt_logits, q_continue_logits),
            (next_q_halt_logits, next_q_continue_logits),
        ) = self.inner(new_inner_carry, new_current_data, memory_readout=memory_readout)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration with more stable implementation
                if self.config.halt_exploration_prob > 0:
                    min_halt_steps = (
                        torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                    ) * torch.randint_like(
                        new_steps, low=2, high=self.config.halt_max_steps + 1
                    )

                    halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q with numerical stability
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                with torch.no_grad():  # Additional safety for no gradients
                    target_q_continue_raw = torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                    # Use stable sigmoid
                    outputs["target_q_continue"] = torch.sigmoid(target_q_continue_raw)

        return (
            HierarchicalReasoningModel_ACTV1Carry(
                new_inner_carry, new_steps, halted, new_current_data
            ),
            outputs,
        )
