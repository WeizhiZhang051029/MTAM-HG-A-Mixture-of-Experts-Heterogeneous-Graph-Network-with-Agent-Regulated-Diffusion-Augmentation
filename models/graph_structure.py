"""Knowledge-guided heterogeneous graph construction."""

from __future__ import annotations

import torch
import torch.nn as nn


class KnowledgeGuidedHeteroGraphLearning(nn.Module):
    """Generate knowledge-guided heterogeneous adjacency matrices."""

    def __init__(
        self,
        num_nodes: int,
        num_node_types: int,
        num_stages: int,
        graph_embed_dim: int,
        node_type_ids: torch.Tensor,
        stage_ids: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.graph_embed_dim = graph_embed_dim
        self.register_buffer("node_ids", torch.arange(num_nodes, dtype=torch.long))
        self.register_buffer("node_type_ids", node_type_ids.long())
        self.register_buffer("stage_ids", stage_ids.long())

        part_dim = graph_embed_dim // 3
        rest = graph_embed_dim - 2 * part_dim
        self.type_embed = nn.Embedding(num_node_types, part_dim)
        self.node_embed = nn.Embedding(num_nodes, part_dim)
        self.stage_embed = nn.Embedding(num_stages, rest)

        self.W_Q = nn.Linear(graph_embed_dim, graph_embed_dim, bias=False)
        self.W_K = nn.Linear(graph_embed_dim, graph_embed_dim, bias=False)
        self.W_G = nn.Linear(graph_embed_dim, num_nodes, bias=False)

    def _structural_embeddings(self) -> torch.Tensor:
        return torch.cat(
            [
                self.type_embed(self.node_type_ids),
                self.node_embed(self.node_ids),
                self.stage_embed(self.stage_ids),
            ],
            dim=-1,
        )

    @staticmethod
    def sigmoid_ad(x: torch.Tensor, expand: float = 0.5) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(expand * (-x))) - 0.5

    def forward(
        self,
        A0: torch.Tensor,
        relation_templates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._structural_embeddings()
        queries = self.W_Q(embeddings)
        keys = self.W_K(embeddings)
        gate_logits = self.W_G(embeddings)
        attention = torch.tanh(queries @ keys.T)
        gate = self.sigmoid_ad(0.1 * gate_logits + A0)
        A_kg = attention * gate
        A_het = A_kg.unsqueeze(0) * relation_templates.to(A_kg.device)
        return A_kg, A_het
