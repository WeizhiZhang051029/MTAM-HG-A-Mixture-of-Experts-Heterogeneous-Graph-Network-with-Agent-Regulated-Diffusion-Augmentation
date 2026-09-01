"""IPOHGN graph encoder for MTAM-HG experts."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models.graph_structure import KnowledgeGuidedHeteroGraphLearning
from utils.graph import (
    build_mechanistic_prior_graph,
    build_node_type_ids,
    build_process_order_ids,
    build_relation_templates,
    build_stage_ids,
)


def _heads_for_dim(d_model: int) -> int:
    for heads in (8, 4, 2):
        if d_model % heads == 0:
            return heads
    return 1


def _sinusoid_encoding_table(length: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * -(math.log(10000.0) / d_model)
    )
    table = torch.zeros(length, d_model, device=device, dtype=dtype)
    table[:, 0::2] = torch.sin(position * div_term)
    if d_model > 1:
        table[:, 1::2] = torch.cos(position * div_term[: table[:, 1::2].shape[1]])
    return table


class ProcessOrderAttention(nn.Module):
    """Multi-head attention across the ordered CAPL variable sequence [B, N, D]."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Process-order attention expects [B, N, D], got {tuple(x.shape)}.")
        batch_size, sequence_length, _ = x.shape
        q = self.W_Q(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_size, sequence_length, self.d_model)
        return self.fc_out(context)


class ProcessOrderTransformer(nn.Module):
    """Transformer over variables sorted by their implicit CAPL process order."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, forward_expansion: int = 2) -> None:
        super().__init__()
        self.attention = ProcessOrderAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, forward_expansion * d_model),
            nn.ReLU(),
            nn.Linear(forward_expansion * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Process-order transformer expects [B, N, D], got {tuple(x.shape)}.")
        _, sequence_length, d_model = x.shape
        pos = _sinusoid_encoding_table(sequence_length, d_model, x.device, x.dtype).unsqueeze(0)
        query = x + pos
        attention = self.attention(query)
        h = self.dropout(self.norm1(attention + query))
        forward = self.ffn(h)
        return self.dropout(self.norm2(forward + h))


class RelationalGraphConv(nn.Module):
    """Basis-decomposed RGCN over generated heterogeneous relation templates."""

    def __init__(self, d_model: int, num_relations: int, num_bases: int = 2, bias: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_relations = num_relations
        self.num_bases = max(1, min(num_bases, num_relations + 1))
        self.w_bases = nn.Parameter(torch.empty(self.num_bases, d_model, d_model))
        self.w_rel = nn.Parameter(torch.empty(num_relations + 1, self.num_bases))
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_bases)
        nn.init.xavier_uniform_(self.w_rel)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, A_het: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        weights = torch.einsum("rb,bio->rio", self.w_rel, self.w_bases)
        supports = []
        eye = torch.eye(A_het.shape[-1], device=A_het.device, dtype=A_het.dtype)
        for rel_id in range(self.num_relations):
            rel_A = A_het[rel_id] * (1.0 - eye)
            message = torch.einsum("ij,bjd->bid", rel_A, x)
            supports.append(message)
        A_base = A_het.sum(dim=0)
        self_A = torch.diag_embed(torch.diagonal(A_base))
        supports.append(torch.einsum("ij,bjd->bid", self_A, x))
        stacked = torch.cat(supports, dim=-1)
        out = torch.matmul(stacked.float(), weights.reshape(-1, self.d_model).float()).to(dtype=x.dtype)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class IPOHGNBlock(nn.Module):
    """One IPOHGN block: process-order attention followed by relational convolution."""

    def __init__(self, d_model: int, num_relations: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.temporal = ProcessOrderTransformer(d_model, num_heads, dropout)
        self.rgcn = RelationalGraphConv(d_model, num_relations)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(
        self,
        A_het: torch.Tensor,
        x: torch.Tensor,
        process_sequence_indices: torch.Tensor,
    ) -> torch.Tensor:
        sequence_indices = process_sequence_indices.to(device=x.device)
        ordered_variables = x.index_select(1, sequence_indices)
        ordered_output = self.temporal(ordered_variables)
        temporal_out = torch.zeros_like(x).index_copy(1, sequence_indices, ordered_output)
        temporal_out = self.norm1(temporal_out + x)
        spatial_out = F.relu(self.rgcn(A_het, temporal_out))
        return self.dropout(self.norm2(spatial_out))


class IPOHGNExpert(nn.Module):
    """IPOHGN expert used by MTAM-HG."""

    def __init__(
        self,
        input_dim: int,
        d_model: int | None = None,
        num_layers: int | None = None,
        use_laplace: bool = True,
        enable_relation_scaling: bool = False,
        expert_name: str | None = None,
        mechanism_focus_nodes: list[str] | None = None,
    ) -> None:
        super().__init__()
        d_model = d_model or config.D_MODEL
        num_layers = num_layers or config.GRAPH_BACKBONE_LAYERS
        self.expert_name = expert_name or "ipohgn"
        self.node_names = config.active_node_names(config.USE_EL_AS_INPUT)
        self.input_node_names = config.input_node_names(config.USE_EL_AS_INPUT)
        if input_dim != len(self.input_node_names):
            raise ValueError(f"Expected input_dim={len(self.input_node_names)}, got {input_dim}.")

        self.num_nodes = len(self.node_names)
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_laplace = use_laplace
        self.attenuation = float(getattr(config, "IPOHGN_LAYER_ATTENUATION", 0.6))
        node_to_idx = {name: idx for idx, name in enumerate(self.node_names)}
        self.input_graph_indices = torch.tensor([node_to_idx[name] for name in self.input_node_names], dtype=torch.long)
        self.ys_virtual_index = (
            node_to_idx[config.VIRTUAL_QUALITY_NODE_NAME]
            if config.VIRTUAL_QUALITY_NODE_NAME in node_to_idx
            else None
        )
        focus_mask = torch.zeros(self.num_nodes, dtype=torch.float32)
        for node in mechanism_focus_nodes or []:
            if node in node_to_idx:
                focus_mask[node_to_idx[node]] = 1.0
        self.has_mechanism_focus = bool(focus_mask.sum().item() > 0)

        node_type_ids = build_node_type_ids(self.node_names)
        stage_ids = build_stage_ids(self.node_names)
        process_order_ids = build_process_order_ids(self.node_names)
        relation_templates, relation_names = build_relation_templates(self.node_names)
        A0 = torch.tensor(build_mechanistic_prior_graph(self.node_names), dtype=torch.float32)
        if self.has_mechanism_focus:
            prior_gain = float(getattr(config, "EXPERT_FOCUS_PRIOR_GAIN", 0.0))
            focus_pair = torch.maximum(focus_mask.view(-1, 1), focus_mask.view(1, -1))
            A0 = A0 * (1.0 + prior_gain * focus_pair)
        num_relations = relation_templates.shape[0]

        self.relation_names = relation_names
        self.register_buffer("node_ids", torch.arange(self.num_nodes, dtype=torch.long))
        self.register_buffer("node_type_ids", node_type_ids.long())
        self.register_buffer("stage_ids", stage_ids.long())
        self.register_buffer("process_order_ids", process_order_ids.long())
        ordered_input_offsets = sorted(
            range(len(self.input_graph_indices)),
            key=lambda offset: (int(process_order_ids[int(self.input_graph_indices[offset])]), offset),
        )
        process_sequence_indices = torch.tensor(
            [int(self.input_graph_indices[offset]) for offset in ordered_input_offsets],
            dtype=torch.long,
        )
        self.register_buffer("process_sequence_indices", process_sequence_indices)
        self.register_buffer("relation_templates", relation_templates)
        self.register_buffer("A0", A0)
        self.register_buffer("mechanism_focus_mask", focus_mask)
        self.register_buffer("input_graph_index_buffer", self.input_graph_indices)
        if enable_relation_scaling:
            self.relation_scaling = nn.Parameter(torch.ones(relation_templates.shape[0]))
        else:
            self.register_parameter("relation_scaling", None)

        part_dim = d_model // 3
        rest_dim = d_model - 2 * part_dim
        self.node_embed = nn.Embedding(self.num_nodes, part_dim)
        self.type_embed = nn.Embedding(len(config.NODE_TYPES), part_dim)
        self.stage_embed = nn.Embedding(config.NUM_GRAPH_STAGES, rest_dim)
        self.W_s = nn.Linear(d_model, d_model, bias=False)
        self.process_order_embed = nn.Embedding(config.NUM_PROCESS_ORDERS, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.virtual_quality_scalar = nn.Parameter(torch.zeros(1))

        self.graph_learning = KnowledgeGuidedHeteroGraphLearning(
            num_nodes=self.num_nodes,
            num_node_types=len(config.NODE_TYPES),
            num_stages=config.NUM_GRAPH_STAGES,
            graph_embed_dim=config.GRAPH_EMBED_DIM,
            node_type_ids=node_type_ids,
            stage_ids=stage_ids,
        )
        num_heads = _heads_for_dim(d_model)
        self.blocks = nn.ModuleList(
            [IPOHGNBlock(d_model, num_relations, num_heads, config.DROPOUT) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
        )
        self.mu_head = nn.Linear(d_model, 1)
        self.log_b_head = nn.Linear(d_model, 1)

    def _structural_embeddings(self) -> torch.Tensor:
        structural_concat = torch.cat(
            [
                self.node_embed(self.node_ids),
                self.type_embed(self.node_type_ids),
                self.stage_embed(self.stage_ids),
            ],
            dim=-1,
        )
        return self.W_s(structural_concat) + self.process_order_embed(self.process_order_ids)

    def _prepare_graph_values(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected x with {self.input_dim} input variables, got {x.shape[1]}.")
        graph_x = torch.zeros(x.shape[0], self.num_nodes, device=x.device, dtype=x.dtype)
        graph_x[:, self.input_graph_index_buffer.to(x.device)] = x
        if self.ys_virtual_index is not None:
            graph_x[:, self.ys_virtual_index] = self.virtual_quality_scalar.to(dtype=x.dtype)
        if self.has_mechanism_focus:
            gain = float(getattr(config, "EXPERT_FOCUS_INPUT_GAIN", 0.0))
            mask = self.mechanism_focus_mask.to(device=x.device, dtype=x.dtype).view(1, -1)
            graph_x = graph_x * (1.0 + gain * mask)
        return graph_x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        graph_x = self._prepare_graph_values(x)
        structural = self._structural_embeddings()
        value_tokens = self.value_proj(graph_x.unsqueeze(-1))
        h = value_tokens + structural.to(x.device, dtype=x.dtype).unsqueeze(0)

        A0 = self.A0.to(x.device, dtype=x.dtype)
        A_kg, A_het = self.graph_learning(
            A0,
            self.relation_templates.to(x.device, dtype=x.dtype),
        )
        A_kg = A_kg.to(dtype=x.dtype)
        A_het = A_het.to(dtype=x.dtype)
        if self.relation_scaling is not None:
            scale = torch.clamp(self.relation_scaling.to(A_het.device, dtype=A_het.dtype), min=1.0e-3)
            A_het = A_het * scale.view(-1, 1, 1)

        block_input = h
        layer_outputs = []
        for layer_idx, block in enumerate(self.blocks):
            block_out = block(
                A_het,
                block_input,
                self.process_sequence_indices,
            )
            layer_outputs.append((self.attenuation ** layer_idx) * block_out)
            block_input = block_out
        if layer_outputs:
            h = torch.stack(layer_outputs, dim=0).sum(dim=0)

        node_h = h
        if self.ys_virtual_index is not None:
            quality_pooled = node_h[:, self.ys_virtual_index, :]
        else:
            quality_pooled = node_h.mean(dim=1)
        pooled = quality_pooled
        if self.has_mechanism_focus:
            mask = self.mechanism_focus_mask.to(device=x.device, dtype=x.dtype).view(1, -1, 1)
            focus_pooled = (node_h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            alpha = float(getattr(config, "EXPERT_FOCUS_READOUT_ALPHA", 0.0))
            pooled = (1.0 - alpha) * quality_pooled + alpha * focus_pooled
        pooled = self.readout(pooled)

        outputs = {
            "mu": self.mu_head(pooled),
            "A_kg": A_kg,
            "A_het": A_het,
            "A_hat": A_kg,
            "A0": A0,
            "hidden": pooled,
            "node_hidden": node_h,
            "mechanism_focus_mask": self.mechanism_focus_mask.to(x.device),
            "process_order_ids": self.process_order_ids.to(x.device),
            "process_sequence_indices": self.process_sequence_indices.to(x.device),
        }
        if self.use_laplace:
            outputs["b"] = F.softplus(self.log_b_head(pooled)) + config.LAPLACE_EPS
        return outputs
