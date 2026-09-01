"""MR-LoRA adapters for real-domain calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LoRAInjectionSummary:
    graph_modules: int
    attention_modules: int
    routing_modules: int

    @property
    def total_modules(self) -> int:
        return self.graph_modules + self.attention_modules + self.routing_modules


class LoRALinear(nn.Module):
    """Wrap an existing Linear layer with a trainable low-rank residual branch."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}.")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling


GRAPH_LORA_TARGETS = (
    "graph_learning.W_Q",
    "graph_learning.W_K",
    "graph_learning.W_G",
)

ATTENTION_LORA_TARGETS = (
    "attention.W_Q",
    "attention.W_K",
    "attention.W_V",
)

ROUTING_LORA_TARGETS = (
    "gate_state_proj.1",
    "router.agent_encoder.0",
    "router.agent_encoder.4",
    "router.routing_head",
    "router.expert_reliability_head.0",
)

MR_LORA_SCOPE_FAMILIES: dict[str, frozenset[str]] = {
    "graph_attention_routing": frozenset({"graph", "attention", "routing"}),
}


def mr_lora_scope_families(scope: str) -> frozenset[str]:
    """Return the adapter families enabled by a validated scope name."""

    normalized = str(scope).strip().lower()
    try:
        return MR_LORA_SCOPE_FAMILIES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported MR_LORA_SCOPE={scope!r}; "
            "the released MTAM-HG model uses graph_attention_routing."
        ) from exc


def _get_parent_and_attr(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _matches_any(module_name: str, targets: tuple[str, ...]) -> bool:
    return any(module_name.endswith(target) for target in targets)


def inject_mr_lora(
    model: nn.Module,
    *,
    graph_rank: int,
    routing_rank: int,
    graph_alpha: float,
    routing_alpha: float,
    dropout: float = 0.0,
) -> LoRAInjectionSummary:
    """Inject graph, attention, and routing LoRA adapters."""

    graph_matches: list[str] = []
    attention_matches: list[str] = []
    routing_matches: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if _matches_any(name, GRAPH_LORA_TARGETS):
            graph_matches.append(name)
        elif _matches_any(name, ATTENTION_LORA_TARGETS):
            attention_matches.append(name)
        elif _matches_any(name, ROUTING_LORA_TARGETS):
            routing_matches.append(name)

    for name in graph_matches:
        parent, attr = _get_parent_and_attr(model, name)
        base = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]  # type: ignore[index]
        wrapped = LoRALinear(base, rank=graph_rank, alpha=graph_alpha, dropout=dropout)
        if attr.isdigit():
            parent[int(attr)] = wrapped  # type: ignore[index]
        else:
            setattr(parent, attr, wrapped)

    for name in attention_matches:
        parent, attr = _get_parent_and_attr(model, name)
        base = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]  # type: ignore[index]
        wrapped = LoRALinear(base, rank=graph_rank, alpha=graph_alpha, dropout=dropout)
        if attr.isdigit():
            parent[int(attr)] = wrapped  # type: ignore[index]
        else:
            setattr(parent, attr, wrapped)

    for name in routing_matches:
        parent, attr = _get_parent_and_attr(model, name)
        base = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]  # type: ignore[index]
        wrapped = LoRALinear(base, rank=routing_rank, alpha=routing_alpha, dropout=dropout)
        if attr.isdigit():
            parent[int(attr)] = wrapped  # type: ignore[index]
        else:
            setattr(parent, attr, wrapped)

    return LoRAInjectionSummary(
        graph_modules=len(graph_matches),
        attention_modules=len(attention_matches),
        routing_modules=len(routing_matches),
    )


def mr_lora_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _param in model.named_parameters() if ".lora_" in name]
