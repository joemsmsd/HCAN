"""
HCAN: Hypergraph-based Cross-Attention Network for Irregular Multivariate Time Series

This module implements the HCAN model for forecasting irregular multivariate time series.
"""

import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import *

from utils.globals import logger 
from utils.ExpConfigs import ExpConfigs

# ======================================================================
# Utility Classes
# ======================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class SwiGLU(nn.Module):
    """SwiGLU activation function with feed-forward network."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.w3(self.act(self.w1(x)) * self.w2(x)))


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention block with optional layer normalization and FFN."""
    def __init__(self, dim_Q, dim_K, dim_V, n_dim, num_heads, ln=False, dropout=0.):
        super(MultiHeadAttentionBlock, self).__init__()
        self.num_heads, self.n_dim = num_heads, n_dim
        self.fc_q, self.fc_k, self.fc_v = nn.Linear(dim_Q, n_dim), nn.Linear(dim_K, n_dim), nn.Linear(dim_V, n_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        if ln:
            self.ln0 = RMSNorm(n_dim)
            self.ln1 = RMSNorm(n_dim)
        self.ffn = SwiGLU(in_features=n_dim, out_features=n_dim, drop=dropout)
        self.fc_o = nn.Linear(n_dim, n_dim)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor = None, mask: Tensor = None, bias: Tensor = None):
        if V is None: V = K
        B, N_q, _ = Q.shape
        Q_p, K_p, V_p = self.fc_q(Q), self.fc_k(K), self.fc_v(V)
        dim_split = self.n_dim // self.num_heads
        
        Q_ = rearrange(Q_p, 'b nq (h d) -> (b h) nq d', h=self.num_heads)
        K_ = rearrange(K_p, 'b nk (h d) -> (b h) nk d', h=self.num_heads)
        V_ = rearrange(V_p, 'b nk (h d) -> (b h) nk d', h=self.num_heads)
        
        Att_mat = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)
        
        if bias is not None:
            bias_repeated = repeat(bias, 'b nq nk -> (b h) nq nk', h=self.num_heads)
            Att_mat = Att_mat + bias_repeated
            
        if mask is not None:
            mask_repeated = repeat(mask, 'b nq nk -> (b h) nq nk', h=self.num_heads)
            Att_mat = Att_mat.masked_fill(mask_repeated == 0, -1e9)
            
        A = self.attn_dropout(torch.softmax(Att_mat, -1))
        O = rearrange(A.bmm(V_), '(b h) nq d -> b nq (h d)', h=self.num_heads)
        
        O = Q + self.proj_dropout(self.fc_o(O))
        if hasattr(self, 'ln0'): O = self.ln0(O)
        O = O + self.ffn(self.ln1(O) if hasattr(self, 'ln1') else O)
        return O


# ===================================================================================
# Core Hypergraph Learner
# ===================================================================================
class HypergraphLearner(nn.Module):
    """
    Main hypergraph learning module that processes observation nodes through
    temporal and variable hyperedge cores with local-global attention fusion.
    """
    def __init__(self, n_layers: int, d_model: int, n_heads: int, n_vars: int, dropout: float, **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_vars = n_vars

        # Learnable queries for temporal and variable cores
        self.time_core_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.var_core_query = nn.Parameter(torch.randn(1, n_vars, d_model))
        
        # Core generators
        self.time_core_generator = nn.ModuleList(
            MultiHeadAttentionBlock(d_model, d_model, d_model, d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        )
        self.var_core_generator = nn.ModuleList(
            MultiHeadAttentionBlock(d_model, d_model, d_model, d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        )
        
        # Global context aggregation via learnable query
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.global_context_generator = nn.ModuleList(
            MultiHeadAttentionBlock(d_model, d_model, d_model, d_model, n_heads, ln=True, dropout=dropout)
            for _ in range(n_layers)
        )
        
        # Local/Global specialized interactions
        self.local_core_interaction = nn.ModuleList(
            MultiHeadAttentionBlock(d_model, d_model, d_model, d_model, n_heads, ln=True, dropout=dropout)
            for _ in range(n_layers)
        )
        self.global_core_interaction = nn.ModuleList(
            MultiHeadAttentionBlock(d_model, d_model, d_model, d_model, n_heads, ln=True, dropout=dropout)
            for _ in range(n_layers)
        )
        self.core_gate = nn.ModuleList(nn.Linear(d_model * 2, d_model) for _ in range(n_layers))
        
        # Dynamic fusion MLP for local/global attention weights
        self.fusion_mlp = nn.ModuleList(
            nn.Linear(d_model * 2, 2) for _ in range(n_layers)
        )

        self.unified_update_mlp = nn.ModuleList(
            SwiGLU(d_model * 3, d_model * 2, d_model, drop=dropout)
            for _ in range(n_layers)
        )
        self.update_norm = nn.ModuleList(RMSNorm(d_model) for _ in range(n_layers))
        
        # Pre-/Post LayerNorm for attention blocks
        self.ln_local_pre = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.ln_local_post = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.ln_global_pre = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.ln_global_post = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        
        # Structural biases: relative time distance and variable similarity
        self.time_bias_gamma = nn.Parameter(torch.tensor(0.1))
        self.var_bias = nn.Parameter(torch.zeros(n_vars, n_vars))
        self.var_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, observation_nodes: Tensor, temporal_incidence_matrix: Tensor,
                variable_incidence_matrix: Tensor, time_indices_flattened: Tensor,
                variable_indices_flattened: Tensor, x_y_mask_flattened: Tensor,
                y_mask_L_flattened: Tensor):
        B, N, D = observation_nodes.shape
        _, C, _ = variable_incidence_matrix.shape
        _, L, _ = temporal_incidence_matrix.shape

        for i in range(self.n_layers):
            if i == 0:
                # Apply mask decay for first layer
                t_mask_decay = (1 - repeat(y_mask_L_flattened, "B N -> B L N", L=L)).clamp(min=1e-8)
                temp_mask = temporal_incidence_matrix * t_mask_decay
                v_mask_decay = (1 - repeat(y_mask_L_flattened, "B N -> B C N", C=C)).clamp(min=1e-8)
                var_mask = variable_incidence_matrix * v_mask_decay
            else:
                temp_mask, var_mask = temporal_incidence_matrix, variable_incidence_matrix

            # Generate temporal and variable cores
            time_cores = self.time_core_generator[i](Q=self.time_core_query.expand(B, L, -1), K=observation_nodes, mask=temp_mask)
            var_cores = self.var_core_generator[i](Q=self.var_core_query.expand(B, -1, -1), K=observation_nodes, mask=var_mask)
            all_cores = torch.cat([time_cores, var_cores], dim=1)

            # Global context via learnable query attention
            global_query_exp = self.global_query.expand(B, -1, -1)
            global_context = self.global_context_generator[i](Q=global_query_exp, K=all_cores)

            # Structural bias for local attention
            device = observation_nodes.device
            
            # Time-time relative distance bias
            if L > 0:
                t_idx = torch.arange(L, device=device)
                t_dist = (t_idx[None, :] - t_idx[:, None]).abs().float()
                t_bias = -torch.abs(self.time_bias_gamma) * t_dist
            else:
                t_bias = torch.zeros((0, 0), device=device)
            
            # Variable-variable similarity bias (symmetrized)
            if C > 0:
                v_bias = 0.5 * (self.var_bias + self.var_bias.t()) * self.var_bias_scale
            else:
                v_bias = torch.zeros((0, 0), device=device)
            
            # Assemble block bias for (L+C) x (L+C)
            LC = L + C
            local_bias = torch.zeros((B, LC, LC), device=device)
            if L > 0:
                local_bias[:, :L, :L] = t_bias
            if C > 0:
                local_bias[:, L:, L:] = v_bias

            # Local and global attention
            local_attn = self.local_core_interaction[i](Q=all_cores, K=all_cores, bias=local_bias)
            global_attn = self.global_core_interaction[i](Q=all_cores, K=global_context, V=global_context)
            
            # Dynamic fusion of local and global attention
            fusion_input = torch.cat([local_attn, global_attn], dim=-1)
            fw = F.softmax(self.fusion_mlp[i](fusion_input), dim=-1)
            fused_attn = fw[:, :, 0:1] * local_attn + fw[:, :, 1:2] * global_attn
            
            # Gated update
            gate = torch.sigmoid(self.core_gate[i](torch.cat([fused_attn, all_cores], dim=-1)))
            all_cores_aware = gate * fused_attn + (1 - gate) * all_cores
            
            time_cores_aware, var_cores_aware = torch.split(all_cores_aware, [L, C], dim=1)

            # Update observation nodes
            gathered_time = time_cores_aware.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=D))
            gathered_var = var_cores_aware.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=D))
            
            fused = torch.cat([observation_nodes, gathered_time, gathered_var], dim=-1)
            update_signal = self.unified_update_mlp[i](fused)
            
            observation_nodes = self.update_norm[i](observation_nodes + update_signal)
            observation_nodes = observation_nodes * repeat(x_y_mask_flattened, "B N -> B N D", D=D)

        return observation_nodes, time_cores_aware, var_cores_aware


# ===================================================================================
# Hypergraph Encoder
# ===================================================================================
class HypergraphEncoder(nn.Module):
    """Encodes observations into hypergraph node representations."""
    def __init__(self, enc_in, d_model):
        super().__init__()
        self.enc_in, self.d_model = enc_in, d_model
        self.unified_node_encoder = nn.Linear(3, d_model)
        self.final_norm = RMSNorm(d_model)

    def forward(self, x_L_flattened, x_y_mask_flattened, x_y_mark, 
                variable_indices_flattened, time_indices_flattened, N_OBSERVATIONS_MAX, seq_len):
        
        is_x_flag = (time_indices_flattened < seq_len).long()
        
        gathered_time_mark = torch.gather(x_y_mark, 1, repeat(time_indices_flattened, "B N -> B N 1"))
        
        unified_input = torch.cat([
            x_L_flattened.unsqueeze(-1),
            is_x_flag.float().unsqueeze(-1),
            gathered_time_mark
        ], dim=-1)

        obs_nodes = self.unified_node_encoder(unified_input)
        
        obs_nodes = self.final_norm(obs_nodes)
        obs_nodes = obs_nodes * repeat(x_y_mask_flattened, "B N -> B N D", D=self.d_model)

        B, L_total, C = x_y_mark.shape[0], x_y_mark.shape[1], self.enc_in
        t_inc = (repeat(time_indices_flattened, "B N -> B L N", L=L_total) == repeat(torch.arange(L_total, device=x_L_flattened.device), "L -> B L N", B=B, N=N_OBSERVATIONS_MAX)).float()
        t_inc *= repeat(x_y_mask_flattened, "B N -> B L N", L=L_total)
        v_inc = (repeat(variable_indices_flattened, "B N -> B C N", C=C) == repeat(torch.arange(C, device=x_L_flattened.device), "C -> B C N", B=B, N=N_OBSERVATIONS_MAX)).float()
        v_inc *= repeat(x_y_mask_flattened, "B N -> B C N", C=C)
        
        return obs_nodes, t_inc, v_inc


class Model(nn.Module):
    """
    HCAN: Hypergraph-based Cross-Attention Network
    
    A neural network model for irregular multivariate time series forecasting
    that leverages hypergraph structures to capture complex temporal and
    variable dependencies.
    """
    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        
        self.hypergraph_encoder = HypergraphEncoder(
            enc_in=self.enc_in,
            d_model=self.d_model
        )
        self.hypergraph_learner = HypergraphLearner(
            n_layers=configs.n_layers, 
            d_model=configs.d_model, 
            n_heads=configs.n_heads,
            n_vars=configs.enc_in, 
            dropout=configs.dropout,
        )
        self.hypergraph_decoder = nn.Linear(self.d_model * 3, 1)

    def forward(self, x: Tensor, x_mark: Tensor = None, x_mask: Tensor = None, 
                y: Tensor = None, y_mark: Tensor = None, y_mask: Tensor = None,
                exp_stage: str = "train", **kwargs):
        
        BATCH_SIZE, SEQ_LEN, _ = x.shape
        PRED_LEN = y.shape[1] if y is not None else (self.configs.pred_len_max_irr or self.configs.pred_len)

        if x_mark is None: x_mark = repeat(torch.arange(SEQ_LEN, dtype=x.dtype, device=x.device), "L -> B L 1", B=BATCH_SIZE)
        if x_mask is None: x_mask = torch.ones_like(x, dtype=torch.bool)
        if y is None: y = torch.zeros((BATCH_SIZE, PRED_LEN, self.enc_in), dtype=x.dtype, device=x.device)
        if y_mark is None: y_mark = repeat(torch.arange(start=SEQ_LEN, end=SEQ_LEN+PRED_LEN, dtype=y.dtype, device=y.device), "L -> B L 1", B=BATCH_SIZE)
        if y_mask is None: y_mask = torch.ones_like(y, dtype=torch.bool)

        x_y_mark = torch.cat([x_mark, y_mark], dim=1)
        x_y_mask_bool = torch.cat([x_mask, y_mask], dim=1).bool()

        x_L = torch.cat([x, torch.zeros_like(y)], dim=1)
        y_L = torch.cat([torch.zeros_like(x), y], dim=1)
        
        ones = torch.ones_like(x_L, dtype=torch.int64)
        time_indices = torch.cumsum(ones, dim=1) - 1
        variable_indices = torch.cumsum(ones, dim=2) - 1

        N_OBSERVATIONS_MAX = x_y_mask_bool.sum((1, 2)).max().item()
        def pad(v): return F.pad(v, [0, N_OBSERVATIONS_MAX - len(v)], value=0)

        x_L_flattened = torch.stack([pad(r[m]) for r, m in zip(x_L, x_y_mask_bool)]).contiguous()
        y_L_flattened = torch.stack([pad(r[m]) for r, m in zip(y_L, x_y_mask_bool)]).contiguous()
        y_mask_L = torch.cat([torch.zeros_like(x_mask, dtype=torch.bool), y_mask.bool()], dim=1)
        y_mask_L_flattened = torch.stack([pad(r[m]) for r, m in zip(y_mask_L, x_y_mask_bool)]).contiguous().float()
        x_y_mask_flattened = torch.stack([pad(torch.ones(m.sum(), device=x.device)) for m in x_y_mask_bool]).contiguous()

        time_indices_flattened = torch.stack([pad(r[m]) for r, m in zip(time_indices, x_y_mask_bool)]).contiguous().long()
        variable_indices_flattened = torch.stack([pad(r[m]) for r, m in zip(variable_indices, x_y_mask_bool)]).contiguous().long()
        
        observation_nodes, t_inc, v_inc = self.hypergraph_encoder(
            x_L_flattened=x_L_flattened, x_y_mask_flattened=x_y_mask_flattened, x_y_mark=x_y_mark,
            variable_indices_flattened=variable_indices_flattened, time_indices_flattened=time_indices_flattened,
            N_OBSERVATIONS_MAX=N_OBSERVATIONS_MAX, seq_len=SEQ_LEN
        )
        
        observation_nodes, time_cores, var_cores = self.hypergraph_learner(
            observation_nodes=observation_nodes, temporal_incidence_matrix=t_inc, variable_incidence_matrix=v_inc,
            time_indices_flattened=time_indices_flattened, variable_indices_flattened=variable_indices_flattened,
            x_y_mask_flattened=x_y_mask_flattened, y_mask_L_flattened=y_mask_L_flattened
        )
        
        gathered_time = time_cores.gather(1, repeat(time_indices_flattened, "B N -> B N D", D=self.d_model))
        gathered_var = var_cores.gather(1, repeat(variable_indices_flattened, "B N -> B N D", D=self.d_model))
        
        decoder_input = torch.cat([observation_nodes, gathered_time, gathered_var], dim=-1)
        pred_flattened = self.hypergraph_decoder(decoder_input).squeeze(-1)

        if exp_stage in ["train", "val"]:
            pred_loss = (torch.pow(pred_flattened - y_L_flattened, 2) * y_mask_L_flattened).sum() / (y_mask_L_flattened.sum() + 1e-8)
            total_loss = pred_loss

            return {
                "loss": total_loss, 
                "pred_loss": pred_loss,
                "pred": pred_flattened,
                "true": y_L_flattened,
                "mask": y_mask_L_flattened
            }
        else:  # test stage
            pred = self.unpad_and_reshape(
                tensor_flattened=pred_flattened,
                original_mask=x_y_mask_bool,
                original_shape=(BATCH_SIZE, SEQ_LEN + PRED_LEN, self.enc_in)
            )
            f_dim = -1 if getattr(self.configs, 'features', 'MS') == 'MS' else 0
            
            return {
                "pred": pred[:, -PRED_LEN:, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:]
            }

    def unpad_and_reshape(self, tensor_flattened: Tensor, original_mask: Tensor, original_shape: Tuple):
        batch_size, _, _ = original_shape
        result = torch.zeros(original_shape, dtype=tensor_flattened.dtype, device=tensor_flattened.device)
        for i in range(batch_size):
            masked_indices = original_mask[i].view(-1).nonzero(as_tuple=True)[0]
            unpadded_sequence = tensor_flattened[i][:len(masked_indices)]
            result[i].view(-1)[masked_indices] = unpadded_sequence
        return result
