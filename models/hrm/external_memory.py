import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalMemory(nn.Module):
    """
    A Differentiable Neural Computer (DNC) memory module.

    This implementation includes content-based addressing, allocation,
    and temporal linking.
    """

    def __init__(
        self,
        d_model,
        m_loc,
        d_mem,
        top_k,
        sparse_addressing,
        use_location_addressing,
        forward_dtype="float32",
    ):
        super().__init__()
        self.m_loc = m_loc
        self.d_mem = d_mem
        self.top_k = min(top_k, m_loc) if sparse_addressing else m_loc
        self.sparse_addressing = sparse_addressing
        self.use_location_addressing = use_location_addressing
        self.dtype = getattr(torch, forward_dtype)

        # Controller to produce the interface vector from the model's hidden state
        # Base: read_key, read_beta, write_key, write_beta, erase, add
        output_dim_L = (4 * d_mem) + 2
        if self.use_location_addressing:
            # Add gates for DNC: write, alloc, read_modes (3)
            output_dim_L += 1 + 1 + 3
        self.memory_controller = nn.Linear(d_model, output_dim_L, dtype=self.dtype)

    def forward(self, z, M_prev, states):
        # Unpack previous states
        w_r_prev = states["w_r_prev"]
        w_w_prev = states["w_w_prev"]
        usage = states["usage"]
        precedence = states["precedence"]
        link_matrix = states["link_matrix"]

        # 1. Produce interface vector from controller
        interface_vec = self.memory_controller(z)
        idx = 0

        # Base NTM interface
        read_key = interface_vec[:, idx : idx + self.d_mem]
        idx += self.d_mem
        # Use a more stable parameterization for betas
        read_beta = F.softplus(interface_vec[:, idx : idx + 1]) + 1.0
        idx += 1
        write_key = interface_vec[:, idx : idx + self.d_mem]
        idx += self.d_mem
        write_beta = F.softplus(interface_vec[:, idx : idx + 1]) + 1.0
        idx += 1
        # Use more stable sigmoid for erase vector
        erase = torch.sigmoid(interface_vec[:, idx : idx + self.d_mem])
        idx += self.d_mem
        add = interface_vec[:, idx : idx + self.d_mem]
        idx += self.d_mem

        # 2. Compute content-based addressing
        w_c_r = self._content_addressing(read_key, read_beta, M_prev)
        w_c_w = self._content_addressing(write_key, write_beta, M_prev)

        if self.use_location_addressing:
            # DNC gates
            write_gate = torch.sigmoid(interface_vec[:, idx : idx + 1])
            idx += 1
            alloc_gate = torch.sigmoid(interface_vec[:, idx : idx + 1])
            idx += 1
            read_modes = F.softmax(interface_vec[:, idx : idx + 3], dim=1)
            idx += 3

            # 3. Compute allocation and write weightings
            # Update usage vector with numerical stability
            usage = usage + w_w_prev - usage * w_w_prev
            # Clamp usage to [0, 1] for numerical stability
            usage = torch.clamp(usage, 0.0, 1.0)

            # Allocation weighting is based on usage
            free_list = 1 - usage
            # Add small epsilon for numerical stability
            alloc_weights = free_list / (
                torch.sum(free_list, dim=1, keepdim=True) + 1e-8
            )

            # Final write weighting is a mix of content and allocation
            w_w = write_gate * (alloc_gate * alloc_weights + (1 - alloc_gate) * w_c_w)

            # 4. Update temporal links and precedence
            precedence_prev = precedence
            precedence = (1 - torch.sum(w_w, dim=1, keepdim=True)) * precedence + w_w

            # More efficient link matrix update
            link_matrix_update = torch.einsum("bi,bj->bij", w_w, precedence_prev)
            link_matrix = (
                (1 - w_w.unsqueeze(2) - w_w.unsqueeze(1)) * link_matrix 
                + link_matrix_update
            )
            # Zero out diagonal more efficiently
            link_matrix.diagonal(dim1=-2, dim2=-1)[:] = 0

            # 5. Compute read weighting
            # Use more efficient matrix operations
            forward_w = torch.bmm(link_matrix, w_r_prev.unsqueeze(-1)).squeeze(-1)
            backward_w = torch.bmm(
                link_matrix.transpose(-2, -1), w_r_prev.unsqueeze(-1)
            ).squeeze(-1)

            # Vectorized read mode computation
            read_mode_weights = read_modes.unsqueeze(-1)  # (batch, 3, 1)
            read_weight_components = torch.stack([backward_w, w_c_r, forward_w], dim=1)  # (batch, 3, m_loc)
            w_r = torch.sum(read_mode_weights * read_weight_components, dim=1)
        else:
            # If not using location addressing, it's a simpler NTM
            w_w = w_c_w
            w_r = w_c_r

        # 6. Write to memory with numerical stability
        # Clamp erase vector for stability
        erase = torch.clamp(erase, 0.0, 1.0)
        erase_m = torch.einsum("bi,bj->bij", w_w, erase)
        add_m = torch.einsum("bi,bj->bij", w_w, add)
        M = M_prev * (1 - erase_m) + add_m

        # 7. Read from memory
        r = torch.einsum("bi,bij->bj", w_r, M)

        # 8. Pack new states
        new_states = {
            "w_r_prev": w_r,
            "w_w_prev": w_w,
            "usage": usage,
            "precedence": precedence,
            "link_matrix": link_matrix,
        }
        return M, r, new_states

    def _content_addressing(self, key, beta, M):
        # key: (batch, d_mem), beta: (batch, 1), M: (batch, m_loc, d_mem)
        # Use normalized cosine similarity for better stability
        key_normalized = F.normalize(key, p=2, dim=-1)
        M_normalized = F.normalize(M, p=2, dim=-1)
        sim = torch.einsum("bd,bmd->bm", key_normalized, M_normalized)
        weighted_sim = sim * beta

        if self.sparse_addressing:
            # Use topk for sparse addressing
            values, indices = torch.topk(weighted_sim, min(self.top_k, weighted_sim.shape[1]), dim=1)
            sparse_w = F.softmax(values, dim=1)
            # Create sparse weight vector
            w = torch.zeros_like(weighted_sim).scatter_(1, indices, sparse_w)
        else:
            w = F.softmax(weighted_sim, dim=1)
        return w

    def init_memory(self, batch_size, device):
        # Initialize with small random values for better training dynamics
        M = torch.randn(
            batch_size, self.m_loc, self.d_mem, device=device, dtype=self.dtype
        ) * 0.01
        states = {
            "w_r_prev": torch.zeros(
                batch_size, self.m_loc, device=device, dtype=self.dtype
            ),
            "w_w_prev": torch.zeros(
                batch_size, self.m_loc, device=device, dtype=self.dtype
            ),
            "usage": torch.zeros(
                batch_size, self.m_loc, device=device, dtype=self.dtype
            ),
            "precedence": torch.zeros(
                batch_size, self.m_loc, device=device, dtype=self.dtype
            ),
            "link_matrix": torch.zeros(
                batch_size, self.m_loc, self.m_loc, device=device, dtype=self.dtype
            ),
        }
        return M, states
