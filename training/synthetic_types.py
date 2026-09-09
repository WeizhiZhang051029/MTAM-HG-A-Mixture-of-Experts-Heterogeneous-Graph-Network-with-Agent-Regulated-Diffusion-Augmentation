from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


@dataclass
class SyntheticBundle:
    loader: DataLoader
    frame: pd.DataFrame
    y_raw: np.ndarray
    is_tail: np.ndarray
    synthetic_source: np.ndarray
    generation_condition: np.ndarray
    process_consistency: np.ndarray
    range_score: np.ndarray
    manifold_score: np.ndarray
    label_consistency_score: np.ndarray
    mechanism_consistency: np.ndarray
    nearest_train_distance: np.ndarray
    nearest_train_index: np.ndarray
    nearest_train_y_raw: np.ndarray
    synthetic_sha256: str = ""
    provenance_sha256: str = ""


@dataclass
class TrainTensorBundle:
    x: torch.Tensor
    y: torch.Tensor
    y_raw: torch.Tensor


@dataclass
class DynamicSyntheticState:
    weights: np.ndarray
    selected_indices: np.ndarray
    selected_mask: np.ndarray
    previous_raw_weights: np.ndarray
    scarcity_bonus: np.ndarray
    bin_ids: np.ndarray
    bin_edges: np.ndarray
    train_bin_counts: np.ndarray
    feedback_features: np.ndarray
    feedback_target: np.ndarray
    previous_train_score: float | None = None
    previous_bin_scores: np.ndarray | None = None
    refresh_count: int = 0
    cluster_ids: np.ndarray | None = None
    previous_cluster_rmse: np.ndarray | None = None
    feedback_history: list[np.ndarray] = field(default_factory=list)
