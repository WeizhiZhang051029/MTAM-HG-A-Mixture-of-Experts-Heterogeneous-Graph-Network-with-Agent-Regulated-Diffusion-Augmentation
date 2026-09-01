"""Build graph schemas, priors, and relation templates."""

from __future__ import annotations

import numpy as np
import torch

import config


def build_node_type_ids(node_names: list[str] | None = None) -> torch.LongTensor:
    node_names = node_names or config.active_node_names()
    ids = [config.NODE_TYPE_TO_ID[config.NODE_TYPE_MAP[name]] for name in node_names]
    return torch.tensor(ids, dtype=torch.long)


def build_stage_ids(node_names: list[str] | None = None) -> torch.LongTensor:
    node_names = node_names or config.active_node_names()
    stage_lookup: dict[str, int] = {}
    for stage_name, names in config.STAGE_NODE_MAP.items():
        stage_id = config.STAGE_TO_ID[stage_name]
        for name in names:
            stage_lookup[name] = stage_id
    stage_lookup.setdefault("EL", config.STAGE_TO_ID["stage3_annealing_thermal"])
    return torch.tensor([stage_lookup[name] for name in node_names], dtype=torch.long)


def build_process_order_ids(node_names: list[str] | None = None) -> torch.LongTensor:
    """Return process-order ids for graph nodes."""
    node_names = node_names or config.active_node_names()
    order_lookup: dict[str, int] = {}
    for order_name, names in config.PROCESS_ORDER_NODE_MAP.items():
        order_id = config.PROCESS_ORDER_TO_ID[order_name]
        for name in names:
            order_lookup[name] = order_id
    fallback = config.PROCESS_ORDER_TO_ID["order11_shear_coiler_quality"]
    return torch.tensor([order_lookup.get(name, fallback) for name in node_names], dtype=torch.long)


def _add_edges(A: np.ndarray, node_to_idx: dict[str, int], sources: list[str], targets: list[str], weight: float) -> None:
    for src in sources:
        for dst in targets:
            if src in node_to_idx and dst in node_to_idx:
                A[node_to_idx[dst], node_to_idx[src]] = max(A[node_to_idx[dst], node_to_idx[src]], weight)


def build_mechanistic_prior_graph(
    node_names: list[str] | None = None,
    same_type_weight: float = 0.08,
    self_loop_weight: float = 1.0,
) -> np.ndarray:
    """Build the metallurgical prior graph."""
    node_names = node_names or config.active_node_names()
    n = len(node_names)
    node_to_idx = {name: idx for idx, name in enumerate(node_names)}
    A = np.zeros((n, n), dtype=np.float32)

    composition = ["C", "Mn", "S", "P"]
    hot_history = ["HT", "FRT", "CT"]
    cold_deformation = ["RF", "BF", "ATh", "AWd", "CRR"]
    annealing = ["FS", "JPF_PT", "HF_T", "SF_T", "SC_T", "FC1_T", "OA_T", "FC2_T", "Q_T"]
    result_nodes = []
    if "EL" in node_to_idx:
        result_nodes.append("EL")
    if config.VIRTUAL_QUALITY_NODE_NAME in node_to_idx:
        result_nodes.append(config.VIRTUAL_QUALITY_NODE_NAME)

    _add_edges(A, node_to_idx, composition, hot_history, 1.00)
    _add_edges(A, node_to_idx, hot_history, cold_deformation, 0.90)
    _add_edges(A, node_to_idx, cold_deformation, annealing, 0.90)
    _add_edges(A, node_to_idx, annealing, result_nodes, 1.00)

    ordered_groups = list(config.PROCESS_ORDER_NODE_MAP.values())
    for sources, targets in zip(ordered_groups, ordered_groups[1:]):
        _add_edges(A, node_to_idx, sources, targets, 0.75)

    for type_name in ["procedure", "conditional", "operating"]:
        type_nodes = [name for name in node_names if config.NODE_TYPE_MAP[name] == type_name]
        _add_edges(A, node_to_idx, type_nodes, type_nodes, same_type_weight)

    np.fill_diagonal(A, self_loop_weight)
    return A


def build_relation_templates(node_names: list[str] | None = None) -> tuple[torch.FloatTensor, list[tuple[str, str]]]:
    """Build heterogeneous relation templates."""
    node_names = node_names or config.active_node_names()
    type_ids = build_node_type_ids(node_names)
    templates = []
    relation_names: list[tuple[str, str]] = []
    for src_type in config.NODE_TYPES:
        src_id = config.NODE_TYPE_TO_ID[src_type]
        for dst_type in config.NODE_TYPES:
            dst_id = config.NODE_TYPE_TO_ID[dst_type]
            src_mask = type_ids == src_id
            dst_mask = type_ids == dst_id
            template = torch.outer(dst_mask.float(), src_mask.float())
            templates.append(template)
            relation_names.append((src_type, dst_type))
    return torch.stack(templates, dim=0), relation_names


def row_normalize_adjacency(A: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Row-normalize adjacency so incoming weights for each target sum to one."""
    return A / (A.sum(dim=-1, keepdim=True) + eps)
