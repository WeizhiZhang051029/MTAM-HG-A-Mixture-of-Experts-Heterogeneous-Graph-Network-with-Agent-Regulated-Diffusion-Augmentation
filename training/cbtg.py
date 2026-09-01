"""Train MTAM-HG with Agent-governed TabDiff samples."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import config
from dataset import CAPLDataset, DataBundle
from evaluate import evaluate_model, save_evaluation_outputs
from losses import total_loss as paper_total_loss
from metrics import compute_metrics
from models.mtam_hg import MECHANISM_EXPERT_NODE_GROUPS
from train import (
    build_experiment_model,
    checkpoint_payload,
    count_total_parameters,
    count_trainable_parameters,
    load_checkpoint,
    resolve_device,
    save_split_artifacts,
    supervised_finetune,
)
from training.clusters import WorkingConditionCluster
from utils.logger import append_csv, ensure_dirs
from utils.seed import set_seed


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
    cross_run_validation_std: np.ndarray | None = None


@dataclass(frozen=True)
class CrossRunValidationStats:
    metric_std: np.ndarray
    num_runs: int
    split_sha256: str
    source: str
    path: Path
    sha256: str


PAPER_CBTG_METRIC_NAMES = ("RMSE", "MAE", "MAPE", "ONE_MINUS_R2", "TAIL_MAE")
PAPER_CBTG_METRIC_WEIGHTS = np.asarray((1.0, 0.3, 0.1, 0.3, 0.2), dtype=np.float64)
PAPER_CBTG_LAMBDA_S = 0.1
PAPER_CBTG_LAMBDA_C = 0.3
PAPER_CBTG_LAMBDA_V = 0.3
PAPER_CBTG_REWARD_MIN = -1.0
PAPER_CBTG_REWARD_MAX = 1.0
PAPER_CBTG_LAMBDA_M = 0.01
PAPER_CBTG_LAMBDA_H = 0.001
PAPER_CBTG_TARGET_CONFIDENCE = 0.6

PAPER_CBTG_STATE_COMPONENTS = (
    "overall_mean",
    "run_std",
    "cluster_value",
    "cluster_variance",
)
AGENT_FEEDBACK_FEATURE_NAMES = tuple(
    f"{metric.lower()}_{component}"
    for metric in PAPER_CBTG_METRIC_NAMES
    for component in PAPER_CBTG_STATE_COMPONENTS
)


def _validated_cross_run_std(values: np.ndarray | list[float]) -> np.ndarray:
    std = np.asarray(values, dtype=np.float64).reshape(-1)
    expected = (len(PAPER_CBTG_METRIC_NAMES),)
    if std.shape != expected or not np.isfinite(std).all() or np.any(std < 0.0):
        raise ValueError(f"validation_metric_std must contain {expected[0]} finite non-negative values.")
    return std


def load_cross_run_validation_stats(
    data_bundle: DataBundle,
    path: str | Path | None = None,
) -> CrossRunValidationStats:
    raw_path = path or getattr(config, "CBTG_CROSS_RUN_VALIDATION_STD_PATH", "")
    if not raw_path:
        raise RuntimeError("CBTG cross-run validation statistics are required.")
    stats_path = Path(raw_path)
    if not stats_path.is_absolute():
        stats_path = Path(config.PROJECT_ROOT) / stats_path
    from protocol_integrity import validate_cbtg_cross_run_validation_std

    config_sha256 = str(getattr(config, "CONFIG_SHA256", "") or "")
    result = validate_cbtg_cross_run_validation_std(
        stats_path,
        expected_split_sha256=str(data_bundle.combined_split_hash),
        expected_source_data_sha256=str(data_bundle.source_sha256),
        expected_config_sha256=config_sha256 or None,
    )
    payload = result["payload"]
    num_runs = int(payload["num_runs"])
    if num_runs != 10:
        raise ValueError("CBTG statistics require ten independent validation runs.")
    split_sha256 = str(payload["combined_split_sha256"])
    return CrossRunValidationStats(
        metric_std=_validated_cross_run_std(payload["validation_metric_std"]),
        num_runs=num_runs,
        split_sha256=split_sha256,
        source="independent_validation_runs",
        path=Path(result["path"]),
        sha256=str(result["sha256"]),
    )


def assert_cross_run_validation_stats_current(
    stats: CrossRunValidationStats,
    data_bundle: DataBundle,
) -> None:
    current = load_cross_run_validation_stats(data_bundle, stats.path)
    if current.sha256 != stats.sha256:
        raise RuntimeError("CBTG cross-run validation statistics changed during training.")


class SyntheticQualityAgent(nn.Module):
    """Score synthetic samples from process and model feedback."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        feedback_dim: int = len(AGENT_FEEDBACK_FEATURE_NAMES),
        attention_dim: int = 64,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.base_input_dim = int(input_dim)
        self.feedback_dim = int(feedback_dim)
        hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.attention_heads = int(attention_heads)
        if self.attention_dim <= 0:
            raise ValueError("attention_dim must be positive.")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        if self.attention_dim % self.attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads.")
        self.sample_query = nn.Sequential(
            nn.Linear(self.base_input_dim, self.attention_dim),
            nn.LayerNorm(self.attention_dim),
            nn.GELU(),
        )
        self.feedback_value = nn.Linear(1, self.attention_dim)
        self.feedback_type_embedding = nn.Parameter(torch.zeros(self.feedback_dim, self.attention_dim))
        self.feedback_norm = nn.LayerNorm(self.attention_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.attention_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        head_input_dim = self.base_input_dim + self.feedback_dim + 2 * self.attention_dim
        self.net = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        y_generated: torch.Tensor,
        feedback_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        if feedback_features is None:
            feedback = x.new_zeros((batch_size, self.feedback_dim))
        else:
            feedback = feedback_features.to(device=x.device, dtype=x.dtype).reshape(batch_size, -1)
            if feedback.shape[1] != self.feedback_dim:
                raise ValueError(
                    f"Agent feedback feature dim {feedback.shape[1]} does not match expected {self.feedback_dim}."
                )
        sample_features = torch.cat(
            [x.reshape(batch_size, -1), y_generated.reshape(batch_size, -1)],
            dim=-1,
        )
        if sample_features.shape[1] != self.base_input_dim:
            raise ValueError(
                f"Agent sample feature dim {sample_features.shape[1]} does not match expected {self.base_input_dim}."
            )
        query = self.sample_query(sample_features).unsqueeze(1)
        feedback_tokens = self.feedback_value(feedback.unsqueeze(-1))
        feedback_tokens = self.feedback_norm(feedback_tokens + self.feedback_type_embedding.unsqueeze(0))
        attended, _ = self.cross_attention(query, feedback_tokens, feedback_tokens, need_weights=False)
        attended = self.attention_dropout(attended.squeeze(1))
        features = torch.cat(
            [
                sample_features,
                feedback,
                query.squeeze(1),
                attended,
            ],
            dim=-1,
        )
        return self.net(features)


def build_synthetic_quality_agent(data_bundle: DataBundle, device: torch.device) -> SyntheticQualityAgent:
    input_dim = len(data_bundle.feature_columns) + 1
    return SyntheticQualityAgent(
        input_dim=input_dim,
        hidden_dim=int(getattr(config, "SYNTHETIC_AGENT_HIDDEN_DIM", 128)),
        dropout=float(getattr(config, "SYNTHETIC_AGENT_DROPOUT", 0.1)),
        attention_dim=int(getattr(config, "SYNTHETIC_AGENT_ATTENTION_DIM", 64)),
        attention_heads=int(getattr(config, "SYNTHETIC_AGENT_ATTENTION_HEADS", 4)),
    ).to(device)


def synthetic_agent_reward_from_real_mse(mse_real: torch.Tensor | float) -> torch.Tensor | float:
    """Real-error reward component: reward_mse = 1 / (1 + MSE_real)."""
    if torch.is_tensor(mse_real):
        return 1.0 / (1.0 + mse_real)
    return 1.0 / (1.0 + float(mse_real))


def _synthetic_reward_weights() -> tuple[float, float, float]:
    weights = np.array(
        [
            float(getattr(config, "SYNTHETIC_REWARD_MSE_WEIGHT", 0.50)),
            float(getattr(config, "SYNTHETIC_REWARD_PROCESS_WEIGHT", 0.25)),
            float(getattr(config, "SYNTHETIC_REWARD_MECHANISM_WEIGHT", 0.25)),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.array([0.50, 0.25, 0.25], dtype=np.float64)
    weights = weights / weights.sum()
    return float(weights[0]), float(weights[1]), float(weights[2])


def synthetic_agent_reward_components(
    mse_real: torch.Tensor | np.ndarray | float,
    process_consistency: torch.Tensor | np.ndarray | float,
    mechanism_consistency: torch.Tensor | np.ndarray | float,
) -> dict[str, torch.Tensor | np.ndarray | float]:
    """Blend train-only real-error, distribution, and mechanism consistency rewards."""
    mse_reward = synthetic_agent_reward_from_real_mse(mse_real)
    w_mse, w_process, w_mechanism = _synthetic_reward_weights()
    if torch.is_tensor(mse_reward):
        process = torch.as_tensor(process_consistency, dtype=mse_reward.dtype, device=mse_reward.device).reshape_as(mse_reward)
        mechanism = torch.as_tensor(mechanism_consistency, dtype=mse_reward.dtype, device=mse_reward.device).reshape_as(mse_reward)
        process = process.clamp(0.0, 1.0)
        mechanism = mechanism.clamp(0.0, 1.0)
        combined = (w_mse * mse_reward + w_process * process + w_mechanism * mechanism).clamp(0.0, 1.0)
        return {
            "reward": combined,
            "reward_mse": mse_reward.clamp(0.0, 1.0),
            "reward_process": process,
            "reward_mechanism": mechanism,
        }
    mse_arr = np.asarray(mse_reward, dtype=np.float32)
    process_arr = np.asarray(process_consistency, dtype=np.float32).reshape(mse_arr.shape)
    mechanism_arr = np.asarray(mechanism_consistency, dtype=np.float32).reshape(mse_arr.shape)
    mse_arr = np.clip(mse_arr, 0.0, 1.0)
    process_arr = np.clip(process_arr, 0.0, 1.0)
    mechanism_arr = np.clip(mechanism_arr, 0.0, 1.0)
    combined_arr = np.clip(w_mse * mse_arr + w_process * process_arr + w_mechanism * mechanism_arr, 0.0, 1.0)
    return {
        "reward": combined_arr.astype(np.float32),
        "reward_mse": mse_arr.astype(np.float32),
        "reward_process": process_arr.astype(np.float32),
        "reward_mechanism": mechanism_arr.astype(np.float32),
    }


def select_synthetic_by_quality_score(
    quality_score: np.ndarray | torch.Tensor,
    threshold: float | None = None,
) -> np.ndarray | torch.Tensor:
    """Return the synthetic-pretrain confidence mask for the current batch."""
    threshold = float(getattr(config, "SYNTHETIC_CONFIDENCE_THRESHOLD", 0.5) if threshold is None else threshold)
    if torch.is_tensor(quality_score):
        return quality_score.reshape(-1) > threshold
    return np.asarray(quality_score).reshape(-1) > threshold


def _unit_interval(values: np.ndarray, neutral: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.where(np.isfinite(arr), arr, neutral)
    return np.clip(out, 0.0, 1.0)


def _normalize_positive(values: np.ndarray, neutral: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr) & (arr >= 0.0)]
    if len(finite) == 0:
        return np.full(arr.shape, float(neutral), dtype=np.float64)
    low = float(np.nanquantile(finite, 0.05))
    high = float(np.nanquantile(finite, 0.95))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1.0e-12:
        return np.full(arr.shape, float(neutral), dtype=np.float64)
    clipped = np.clip(np.where(np.isfinite(arr) & (arr >= 0.0), arr, low), low, high)
    return np.clip((clipped - low) / (high - low), 0.0, 1.0)


def _project_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (config.PROJECT_ROOT / raw)


def _read_table(path: Path) -> pd.DataFrame:
    from protocol_integrity import read_table_snapshot

    return read_table_snapshot(path).frame


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported synthetic data file type: {path.suffix}")


def _inverse_x(data_bundle: DataBundle, x: np.ndarray) -> np.ndarray:
    return data_bundle.x_scaler.inverse_transform(x) if config.STANDARDIZE_X else x


def _inverse_y(data_bundle: DataBundle, y: np.ndarray) -> np.ndarray:
    return data_bundle.y_scaler.inverse_transform(y) if config.STANDARDIZE_Y else y


def _train_arrays(data_bundle: DataBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = data_bundle.train_loader.dataset
    x_scaled = dataset.x.detach().cpu().numpy().astype(np.float32)
    y_scaled = dataset.y.detach().cpu().numpy().astype(np.float32)
    x_raw = _inverse_x(data_bundle, x_scaled).astype(np.float32)
    y_raw = _inverse_y(data_bundle, y_scaled).astype(np.float32)
    return x_scaled, y_scaled, x_raw, y_raw


def _train_tensor_bundle(data_bundle: DataBundle, device: torch.device) -> TrainTensorBundle:
    dataset = data_bundle.train_loader.dataset
    return TrainTensorBundle(
        x=dataset.x.detach().to(device),
        y=dataset.y.detach().to(device),
        y_raw=torch.tensor(data_bundle.y_train_raw, dtype=torch.float32, device=device).reshape(-1, 1),
    )


def _normalize_indices(indices: np.ndarray, length: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    return np.clip(indices, 0, max(length - 1, 0))


@torch.no_grad()
def _real_mse_reward_for_indices(
    model: torch.nn.Module,
    train_tensors: TrainTensorBundle,
    nearest_train_index: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate train-only real-label MSE for synthetic samples' nearest regions."""
    idx_np = _normalize_indices(nearest_train_index, train_tensors.x.shape[0])
    idx = torch.tensor(idx_np, dtype=torch.long, device=device)
    outputs = model(train_tensors.x.index_select(0, idx))
    if torch.is_tensor(outputs):
        y_pred = outputs.reshape(-1, 1)
    else:
        y_pred = outputs["mu"].reshape(-1, 1)
    y_true = train_tensors.y.index_select(0, idx).reshape(-1, 1)
    mse_real = ((y_pred - y_true) ** 2).reshape(y_pred.shape[0], -1).mean(dim=-1)
    reward = synthetic_agent_reward_from_real_mse(mse_real).clamp(0.0, 1.0)
    return mse_real.detach(), reward.detach(), y_pred.detach(), y_true.detach()


def _pairwise_min_distance(
    query: np.ndarray,
    reference: np.ndarray,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    if len(reference) == 0:
        raise ValueError("Reference array is empty.")
    distances: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    ref = reference.astype(np.float32, copy=False)
    for start in range(0, len(query), chunk_size):
        q = query[start:start + chunk_size].astype(np.float32, copy=False)
        diff = q[:, None, :] - ref[None, :, :]
        dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
        idx = np.argmin(dist, axis=1)
        distances.append(dist[np.arange(len(q)), idx])
        indices.append(idx.astype(np.int64))
    return np.concatenate(distances).astype(np.float32), np.concatenate(indices).astype(np.int64)


def _knn_label_mean(
    query: np.ndarray,
    reference: np.ndarray,
    y_reference: np.ndarray,
    k: int,
    chunk_size: int = 1024,
) -> np.ndarray:
    if len(reference) == 0:
        raise ValueError("Reference array is empty.")
    k = max(1, min(int(k), len(reference)))
    means: list[np.ndarray] = []
    ref = reference.astype(np.float32, copy=False)
    y_ref = y_reference.reshape(-1).astype(np.float32, copy=False)
    for start in range(0, len(query), chunk_size):
        q = query[start:start + chunk_size].astype(np.float32, copy=False)
        diff = q[:, None, :] - ref[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        if k == len(reference):
            idx = np.argsort(dist_sq, axis=1)[:, :k]
        else:
            idx = np.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]
        means.append(y_ref[idx].mean(axis=1))
    return np.concatenate(means).reshape(-1, 1).astype(np.float32)


def _leave_one_out_distance(reference: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
    if len(reference) <= 1:
        return np.ones(len(reference), dtype=np.float32)
    distances: list[np.ndarray] = []
    ref = reference.astype(np.float32, copy=False)
    for start in range(0, len(reference), chunk_size):
        q = ref[start:start + chunk_size]
        diff = q[:, None, :] - ref[None, :, :]
        dist = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
        rows = np.arange(len(q))
        dist[rows, start + rows] = np.inf
        distances.append(np.min(dist, axis=1))
    return np.concatenate(distances).astype(np.float32)


def compute_process_consistency_scores(
    data_bundle: DataBundle,
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
) -> dict[str, np.ndarray]:
    """Score synthetic samples against train-only process distribution constraints."""
    train_x_scaled, train_y_scaled, train_x_raw, train_y_raw = _train_arrays(data_bundle)
    low_q = float(getattr(config, "SYNTHETIC_PROCESS_RANGE_QUANTILE_LOW", 0.01))
    high_q = float(getattr(config, "SYNTHETIC_PROCESS_RANGE_QUANTILE_HIGH", 0.99))
    margin = float(getattr(config, "SYNTHETIC_PROCESS_RANGE_MARGIN", 0.10))
    train_raw = np.concatenate([train_x_raw, train_y_raw], axis=1)
    syn_raw = np.concatenate([x_raw, y_raw], axis=1)
    lower = np.nanquantile(train_raw, low_q, axis=0)
    upper = np.nanquantile(train_raw, high_q, axis=0)
    spread = np.maximum(upper - lower, 1.0e-6)
    lower = lower - margin * spread
    upper = upper + margin * spread
    below = np.maximum(lower.reshape(1, -1) - syn_raw, 0.0)
    above = np.maximum(syn_raw - upper.reshape(1, -1), 0.0)
    excess = (below + above) / spread.reshape(1, -1)
    range_score = np.exp(-excess).mean(axis=1)

    nearest_distance, nearest_index = _pairwise_min_distance(x_scaled, train_x_scaled)
    train_nn = _leave_one_out_distance(train_x_scaled)
    distance_scale = float(np.nanquantile(train_nn, 0.95)) if len(train_nn) else 1.0
    if not np.isfinite(distance_scale) or distance_scale < 1.0e-6:
        distance_scale = float(np.nanmedian(train_nn[train_nn > 0])) if np.any(train_nn > 0) else 1.0
    manifold_score = np.exp(-((nearest_distance / max(distance_scale, 1.0e-6)) ** 2))

    k = int(getattr(config, "SYNTHETIC_PROCESS_KNN_K", 5))
    knn_y_scaled = _knn_label_mean(x_scaled, train_x_scaled, train_y_scaled, k=k)
    y_scale = float(np.nanstd(train_y_scaled))
    if not np.isfinite(y_scale) or y_scale < 1.0e-6:
        y_scale = 1.0
    label_consistency_score = np.exp(-np.abs(y_scaled.reshape(-1) - knn_y_scaled.reshape(-1)) / y_scale)

    weights = np.array(
        [
            float(getattr(config, "SYNTHETIC_PROCESS_RANGE_WEIGHT", 0.35)),
            float(getattr(config, "SYNTHETIC_PROCESS_MANIFOLD_WEIGHT", 0.35)),
            float(getattr(config, "SYNTHETIC_PROCESS_LABEL_WEIGHT", 0.30)),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.array([0.35, 0.35, 0.30], dtype=np.float32)
    weights = weights / weights.sum()
    process_consistency = (
        weights[0] * range_score
        + weights[1] * manifold_score
        + weights[2] * label_consistency_score
    )
    process_consistency = np.clip(process_consistency, 0.0, 1.0)
    return {
        "process_consistency": process_consistency.astype(np.float32),
        "range_score": np.clip(range_score, 0.0, 1.0).astype(np.float32),
        "manifold_score": np.clip(manifold_score, 0.0, 1.0).astype(np.float32),
        "label_consistency_score": np.clip(label_consistency_score, 0.0, 1.0).astype(np.float32),
        "nearest_train_distance": nearest_distance.astype(np.float32),
        "nearest_train_index": nearest_index.astype(np.int64),
        "nearest_train_y_raw": train_y_raw[nearest_index].reshape(-1).astype(np.float32),
    }


def compute_mechanism_consistency_scores(
    data_bundle: DataBundle,
    x_raw: np.ndarray,
) -> np.ndarray:
    """Score synthetic samples against train-only metallurgical mechanism groups."""
    _, _, train_x_raw, _ = _train_arrays(data_bundle)
    feature_to_idx = {name: idx for idx, name in enumerate(data_bundle.standard_node_names)}
    group_scores: list[np.ndarray] = []
    low_q = float(getattr(config, "SYNTHETIC_PROCESS_RANGE_QUANTILE_LOW", 0.01))
    high_q = float(getattr(config, "SYNTHETIC_PROCESS_RANGE_QUANTILE_HIGH", 0.99))
    margin = float(getattr(config, "SYNTHETIC_PROCESS_RANGE_MARGIN", 0.10))
    for nodes in MECHANISM_EXPERT_NODE_GROUPS.values():
        indices = [feature_to_idx[node] for node in nodes if node in feature_to_idx]
        if not indices:
            continue
        train_group = train_x_raw[:, indices]
        syn_group = x_raw[:, indices]
        lower = np.nanquantile(train_group, low_q, axis=0)
        upper = np.nanquantile(train_group, high_q, axis=0)
        spread = np.maximum(upper - lower, 1.0e-6)
        lower = lower - margin * spread
        upper = upper + margin * spread
        below = np.maximum(lower.reshape(1, -1) - syn_group, 0.0)
        above = np.maximum(syn_group - upper.reshape(1, -1), 0.0)
        excess = (below + above) / spread.reshape(1, -1)
        group_scores.append(np.exp(-excess).mean(axis=1))
    if not group_scores:
        return np.ones(x_raw.shape[0], dtype=np.float32)
    return np.clip(np.stack(group_scores, axis=1).mean(axis=1), 0.0, 1.0).astype(np.float32)


def _agent_synthetic_weight(
    confidence: torch.Tensor,
    process_consistency: torch.Tensor,
    mechanism_consistency: torch.Tensor,
    synthetic_keep_score: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine Agent confidence with process and metallurgical mechanism consistency."""
    weight = confidence.reshape(-1).clamp(0.0, 1.0)
    if synthetic_keep_score is not None:
        governance = synthetic_keep_score.reshape(-1).clamp(0.0, 1.0)
        weight = weight * governance
    if bool(getattr(config, "SYNTHETIC_USE_PROCESS_CONSISTENCY", True)):
        process_power = float(getattr(config, "SYNTHETIC_PROCESS_SCORE_POWER", 1.0))
        weight = weight * process_consistency.clamp(0.0, 1.0).pow(process_power)
    if bool(getattr(config, "SYNTHETIC_USE_MECHANISM_CONSISTENCY", True)):
        mechanism_power = float(getattr(config, "SYNTHETIC_MECHANISM_SCORE_POWER", 1.0))
        weight = weight * mechanism_consistency.clamp(0.0, 1.0).pow(mechanism_power)
    return weight


def create_synthetic_smoke_file(data_bundle: DataBundle, output_path: Path, n_samples: int = 64) -> Path:
    dataset = data_bundle.train_loader.dataset
    n = min(n_samples, len(dataset))
    x_scaled = dataset.x[:n].detach().cpu().numpy()
    y_scaled = dataset.y[:n].detach().cpu().numpy()
    x_raw = _inverse_x(data_bundle, x_scaled)
    y_raw = _inverse_y(data_bundle, y_scaled)

    rng = np.random.default_rng(config.SEED)
    x_std = np.asarray(data_bundle.x_scaler.std_).reshape(-1) if config.STANDARDIZE_X else np.std(x_raw, axis=0)
    y_std = np.asarray(data_bundle.y_scaler.std_).reshape(-1) if config.STANDARDIZE_Y else np.std(y_raw, axis=0)
    x_noisy = x_raw + rng.normal(0.0, 0.01, size=x_raw.shape) * np.maximum(x_std.reshape(1, -1), 1.0e-6)
    y_noisy = y_raw + rng.normal(0.0, 0.01, size=y_raw.shape) * np.maximum(y_std.reshape(1, -1), 1.0e-6)

    df = pd.DataFrame(x_noisy, columns=data_bundle.feature_columns)
    df[data_bundle.label_column] = y_noisy.reshape(-1)
    low, high = data_bundle.tail_thresholds
    df["synthetic_source"] = "SmokeTestNotTabDiff"
    df["generation_condition"] = "smoke_test_only"
    df["is_tail_synthetic"] = (df[data_bundle.label_column] <= low) | (df[data_bundle.label_column] >= high)
    df["tabdiff_sample_id"] = np.arange(len(df), dtype=np.int64)
    _save_table(df, output_path)
    return output_path


def _load_synthetic_bundle(data_bundle: DataBundle, synthetic_path: str | Path | None = None) -> SyntheticBundle:
    path = _project_path(synthetic_path or getattr(config, "SYNTHETIC_DATA_PATH", "data/synthetic_CAPL_ma_tabdiff.xlsx"))
    if not path.exists():
        raise FileNotFoundError(f"Synthetic data file was not found: {path}")

    from protocol_integrity import (
        assert_file_snapshot_current,
        read_table_snapshot,
        validate_synthetic_provenance_for_data_bundle,
    )

    table_snapshot = read_table_snapshot(path)
    provenance = validate_synthetic_provenance_for_data_bundle(
        path,
        data_bundle,
        int(getattr(config, "TABDIFF_GENERATION_SEED", 0)),
        synthetic_snapshot=table_snapshot,
        expected_scientific_code_sha256=(
            str(getattr(config, "EXPECTED_SCIENTIFIC_CODE_SHA256", "") or "") or None
        ),
        expected_generation_protocol_sha256=(
            str(getattr(config, "EXPECTED_GENERATION_PROTOCOL_SHA256", "") or "") or None
        ),
    )
    synthetic_sha256 = str(provenance["synthetic_sha256"])
    provenance_sha256 = str(provenance["provenance_sha256"])

    df = table_snapshot.frame.copy()
    df.columns = [str(col) for col in df.columns]
    label_col = str(getattr(config, "SYNTHETIC_LABEL_COL", "") or data_bundle.label_column)
    if label_col not in df.columns:
        label_col = data_bundle.label_column
    if label_col not in df.columns:
        raise KeyError(f"Synthetic label column is missing: {label_col}")
    missing_features = [col for col in data_bundle.feature_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"Synthetic data is missing model feature columns: {missing_features}")

    x_raw = df[data_bundle.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y_raw = df[[label_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    valid = ~np.isnan(y_raw.reshape(-1))
    if not np.all(valid):
        df = df.loc[valid].reset_index(drop=True)
        x_raw = x_raw[valid]
        y_raw = y_raw[valid]

    fill = np.asarray(data_bundle.x_scaler.mean_, dtype=np.float32).reshape(1, -1)
    nan_rows, nan_cols = np.where(np.isnan(x_raw))
    if len(nan_rows):
        x_raw[nan_rows, nan_cols] = fill.reshape(-1)[nan_cols]
    y_fill = np.asarray(data_bundle.y_scaler.mean_, dtype=np.float32).reshape(1, -1)
    y_nan = np.where(np.isnan(y_raw))
    if len(y_nan[0]):
        y_raw[y_nan] = y_fill.reshape(-1)[y_nan[1]]

    x = data_bundle.x_scaler.transform(x_raw) if config.STANDARDIZE_X else x_raw
    y = data_bundle.y_scaler.transform(y_raw) if config.STANDARDIZE_Y else y_raw
    sample_ids = np.arange(len(df), dtype=np.int64)
    loader = DataLoader(
        CAPLDataset(x, y, sample_ids),
        batch_size=int(getattr(config, "SYNTHETIC_BATCH_SIZE", config.BATCH_SIZE)),
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    if "is_tail_synthetic" in df.columns:
        is_tail = df["is_tail_synthetic"].astype(bool).to_numpy()
    else:
        low, high = data_bundle.tail_thresholds
        is_tail = ((y_raw.reshape(-1) <= low) | (y_raw.reshape(-1) >= high))
    source = df["synthetic_source"].astype(str).to_numpy() if "synthetic_source" in df.columns else np.array(["TabDiff"] * len(df))
    condition = (
        df["generation_condition"].astype(str).to_numpy()
        if "generation_condition" in df.columns
        else np.array(["unconditional"] * len(df))
    )
    scores = compute_process_consistency_scores(data_bundle, x, y, x_raw, y_raw)
    mechanism_consistency = compute_mechanism_consistency_scores(data_bundle, x_raw)
    bundle = SyntheticBundle(
        loader=loader,
        frame=df,
        y_raw=y_raw.reshape(-1),
        is_tail=is_tail.reshape(-1),
        synthetic_source=source.reshape(-1),
        generation_condition=condition.reshape(-1),
        process_consistency=scores["process_consistency"],
        range_score=scores["range_score"],
        manifold_score=scores["manifold_score"],
        label_consistency_score=scores["label_consistency_score"],
        mechanism_consistency=mechanism_consistency,
        nearest_train_distance=scores["nearest_train_distance"],
        nearest_train_index=scores["nearest_train_index"],
        nearest_train_y_raw=scores["nearest_train_y_raw"],
        synthetic_sha256=synthetic_sha256,
        provenance_sha256=provenance_sha256,
    )
    assert_file_snapshot_current(table_snapshot.file, "synthetic data")
    return bundle


def _synthetic_equal_width_bins(y_values: np.ndarray, reference_y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference_y, dtype=np.float64).reshape(-1)
    ref = ref[np.isfinite(ref)]
    n_bins = max(2, int(n_bins))
    if len(ref) == 0:
        return np.zeros(len(y), dtype=np.int64), np.array([0.0, 1.0]), np.ones(1, dtype=np.int64)
    low = float(np.min(ref))
    high = float(np.max(ref))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, n_bins + 1, dtype=np.float64)
    counts, _ = np.histogram(ref, bins=edges)
    bin_ids = np.digitize(y, edges[1:-1], right=False)
    bin_ids = np.clip(bin_ids, 0, len(counts) - 1).astype(np.int64)
    return bin_ids, edges, counts.astype(np.int64)


def synthetic_scarcity_bonus(
    y_values: np.ndarray,
    reference_y: np.ndarray,
    n_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train-label scarcity bonuses for synthetic rows using train-only labels."""
    bins = int(n_bins or getattr(config, "DYNAMIC_SYNTHETIC_SCARCITY_BINS", 10))
    bin_ids, edges, counts = _synthetic_equal_width_bins(y_values, reference_y, bins)
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return np.ones(len(bin_ids), dtype=np.float64), bin_ids, edges, counts
    max_count = float(nonzero.max())
    safe_counts = np.maximum(counts, 1)
    raw = max_count / safe_counts[bin_ids]
    if len(raw) == 0:
        return raw.astype(np.float64), bin_ids, edges, counts
    raw_min = float(np.min(raw))
    raw_max = float(np.max(raw))
    if raw_max <= raw_min + 1.0e-12:
        bonus = np.ones_like(raw, dtype=np.float64)
    else:
        bonus = (raw - raw_min) / (raw_max - raw_min)
    return np.clip(bonus, 0.0, 1.0), bin_ids, edges, counts


def _bin_ids_from_edges(y_values: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    edge_values = np.asarray(edges, dtype=np.float64).reshape(-1)
    if len(edge_values) < 2:
        return np.zeros(len(y), dtype=np.int64)
    bin_ids = np.digitize(y, edge_values[1:-1], right=False)
    return np.clip(bin_ids, 0, max(int(n_bins) - 1, 0)).astype(np.int64)


def _bin_means(values: np.ndarray, bin_ids: np.ndarray, n_bins: int) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    ids = np.asarray(bin_ids, dtype=np.int64).reshape(-1)
    out = np.full(int(n_bins), np.nan, dtype=np.float64)
    for idx in range(int(n_bins)):
        mask = ids == idx
        if bool(mask.any()):
            finite = vals[mask]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                out[idx] = float(np.mean(finite))
    return out


def _fill_missing_bin_values(values: np.ndarray, neutral: float = 0.0) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.isfinite(vals).any():
        fallback = float(np.nanmean(vals))
    else:
        fallback = float(neutral)
    return np.where(np.isfinite(vals), vals, fallback).astype(np.float64)


def zero_agent_feedback_features(n: int) -> np.ndarray:
    return np.zeros((int(n), len(AGENT_FEEDBACK_FEATURE_NAMES)), dtype=np.float64)


def _paper_oriented_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tail_thresholds: tuple[float, float],
) -> np.ndarray:
    """Return the five paper metrics with a common lower-is-better direction."""
    values = compute_metrics(
        np.asarray(y_true, dtype=np.float64).reshape(-1, 1),
        np.asarray(y_pred, dtype=np.float64).reshape(-1, 1),
        tail_thresholds=tail_thresholds,
    )
    return np.asarray(
        (
            values["RMSE"],
            values["MAE"],
            values["MAPE"],
            1.0 - values["R2"],
            values["TAIL_MAE"],
        ),
        dtype=np.float64,
    )


def _zscore_across_metrics(values: np.ndarray) -> np.ndarray:
    """Standardize one state component so the five metrics are comparable."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    count = finite.sum(axis=-1, keepdims=True)
    row_sum = np.where(finite, array, 0.0).sum(axis=-1, keepdims=True)
    mean = np.divide(row_sum, np.maximum(count, 1), dtype=np.float64)
    filled = np.where(finite, array, mean)
    std = filled.std(axis=-1, keepdims=True)
    return np.where(std > 1.0e-12, (filled - mean) / std, 0.0).astype(np.float64)


def build_paper_cbtg_feedback(
    overall_metrics: np.ndarray,
    run_std: np.ndarray,
    per_cluster_metrics: np.ndarray,
    sample_cluster_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build CBTG states and clipped rewards."""
    overall = np.asarray(overall_metrics, dtype=np.float64).reshape(-1)
    std = _validated_cross_run_std(run_std)
    clusters = np.asarray(per_cluster_metrics, dtype=np.float64)
    ids = np.asarray(sample_cluster_ids, dtype=np.int64).reshape(-1)
    metric_count = len(PAPER_CBTG_METRIC_NAMES)
    if overall.shape != (metric_count,):
        raise ValueError(f"overall_metrics must have shape ({metric_count},).")
    if clusters.ndim != 2 or clusters.shape[1] != metric_count or clusters.shape[0] < 1:
        raise ValueError(f"per_cluster_metrics must have shape (G, {metric_count}) with G >= 1.")

    filled_clusters = clusters.copy()
    for metric_idx in range(metric_count):
        fallback = overall[metric_idx] if np.isfinite(overall[metric_idx]) else 0.0
        filled_clusters[:, metric_idx] = np.where(
            np.isfinite(filled_clusters[:, metric_idx]),
            filled_clusters[:, metric_idx],
            fallback,
        )
    cluster_variance = np.var(filled_clusters, axis=0)
    safe_ids = np.clip(ids, 0, filled_clusters.shape[0] - 1)
    assigned_cluster = filled_clusters[safe_ids]
    n = len(ids)
    raw_state = np.stack(
        (
            np.broadcast_to(overall, (n, metric_count)),
            np.broadcast_to(std, (n, metric_count)),
            assigned_cluster,
            np.broadcast_to(cluster_variance, (n, metric_count)),
        ),
        axis=-1,
    )
    standardized = np.empty_like(raw_state, dtype=np.float64)
    for component_idx in range(len(PAPER_CBTG_STATE_COMPONENTS)):
        standardized[:, :, component_idx] = _zscore_across_metrics(
            raw_state[:, :, component_idx]
        )

    metric_terms = (
        standardized[:, :, 0]
        + PAPER_CBTG_LAMBDA_S * standardized[:, :, 1]
        + PAPER_CBTG_LAMBDA_C * standardized[:, :, 2]
        + PAPER_CBTG_LAMBDA_V * standardized[:, :, 3]
    )
    reward = -np.sum(PAPER_CBTG_METRIC_WEIGHTS[None, :] * metric_terms, axis=1)
    reward = np.clip(reward, PAPER_CBTG_REWARD_MIN, PAPER_CBTG_REWARD_MAX)
    return {
        "features": standardized.reshape(n, -1).astype(np.float64),
        "target": reward.astype(np.float64),
        "raw_state": raw_state.astype(np.float64),
        "standardized_state": standardized.astype(np.float64),
        "overall_metrics": overall.astype(np.float64),
        "run_std": std.astype(np.float64),
        "per_cluster_metrics": filled_clusters.astype(np.float64),
        "cluster_variance": cluster_variance.astype(np.float64),
        "assigned_cluster_metrics": assigned_cluster.astype(np.float64),
    }


@torch.no_grad()
def evaluate_validation_pretrain_feedback(
    model: torch.nn.Module,
    data_bundle: DataBundle,
    device: torch.device,
    cross_run_validation_std: np.ndarray,
    cluster_model: WorkingConditionCluster | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute paper CBTG feedback exclusively from real validation rows."""
    model_was_training = model.training
    model.eval()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for batch in data_bundle.val_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        outputs = model(x)
        xs.append(x.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        preds.append(_output_mu(outputs).detach().cpu().numpy())
    if not ys:
        raise ValueError("CBTG-Agent requires a non-empty validation loader.")
    x_val = np.concatenate(xs, axis=0)
    y_scaled = np.concatenate(ys, axis=0)
    pred_scaled = np.concatenate(preds, axis=0)
    if bool(getattr(config, "STANDARDIZE_Y", True)):
        y_true = data_bundle.y_scaler.inverse_transform(y_scaled).reshape(-1)
        y_pred = data_bundle.y_scaler.inverse_transform(pred_scaled).reshape(-1)
    else:
        y_true = y_scaled.reshape(-1)
        y_pred = pred_scaled.reshape(-1)

    overall = _paper_oriented_metrics(y_true, y_pred, data_bundle.tail_thresholds)
    run_std = _validated_cross_run_std(cross_run_validation_std)

    if cluster_model is not None and cluster_model.is_fitted:
        validation_cluster_ids = cluster_model.predict(x_val)
        cluster_count = cluster_model.effective_n_clusters
    else:
        validation_cluster_ids = np.zeros(len(x_val), dtype=np.int64)
        cluster_count = 1
    per_cluster = np.full((cluster_count, len(PAPER_CBTG_METRIC_NAMES)), np.nan, dtype=np.float64)
    for cluster_idx in range(cluster_count):
        mask = validation_cluster_ids == cluster_idx
        if bool(mask.any()):
            per_cluster[cluster_idx] = _paper_oriented_metrics(
                y_true[mask],
                y_pred[mask],
                data_bundle.tail_thresholds,
            )
    per_cluster = np.where(np.isfinite(per_cluster), per_cluster, overall[None, :])
    if model_was_training:
        model.train()
    return {
        "overall_metrics": overall.astype(np.float64),
        "run_std": run_std.astype(np.float64),
        "cross_run_validation_std": run_std.astype(np.float64),
        "per_cluster_metrics": per_cluster.astype(np.float64),
        "validation_cluster_ids": validation_cluster_ids.astype(np.int64),
        "validation_rmse": float(overall[0]),
        "validation_mae": float(overall[1]),
        "validation_mape": float(overall[2]),
        "validation_one_minus_r2": float(overall[3]),
        "validation_tail_mae": float(overall[4]),
    }


def agent_policy_quota_multiplier(
    agent_policy_score: np.ndarray,
    reliability: np.ndarray,
) -> np.ndarray:
    """Allocate expected synthetic draws from the feedback-conditioned Agent output."""
    mapped = _unit_interval(agent_policy_score, neutral=0.5)
    strength = max(0.0, float(getattr(config, "DYNAMIC_SYNTHETIC_QUOTA_STRENGTH", 0.50)))
    quota_min = max(0.0, float(getattr(config, "DYNAMIC_SYNTHETIC_QUOTA_MIN", 0.50)))
    quota_max = max(quota_min, float(getattr(config, "DYNAMIC_SYNTHETIC_QUOTA_MAX", 1.75)))
    centered = mapped - float(np.nanmean(mapped)) if len(mapped) else mapped
    quota = 1.0 + strength * centered
    reliability = _unit_interval(reliability, neutral=1.0)
    quota = quota * np.clip(reliability, 0.0, 1.0)
    finite = quota[np.isfinite(quota)]
    if len(finite) and float(np.mean(finite)) > 1.0e-12:
        quota = quota / float(np.mean(finite))
    return np.clip(quota, quota_min, quota_max).astype(np.float64)


def build_dynamic_synthetic_state(
    synthetic_bundle: SyntheticBundle,
    data_bundle: DataBundle,
    cross_run_validation_std: np.ndarray,
    cluster_model: WorkingConditionCluster | None = None,
) -> DynamicSyntheticState:
    n = len(synthetic_bundle.frame)
    initial = np.ones(n, dtype=np.float64)
    for column in (
        "final_synthetic_weight",
        "training_weight",
        "agent_quality_score",
    ):
        if column in synthetic_bundle.frame.columns:
            values = pd.to_numeric(synthetic_bundle.frame[column], errors="coerce").to_numpy(dtype=np.float64)
            if len(values) == n:
                initial = np.clip(np.where(np.isfinite(values), values, 1.0), 0.0, None)
                break
    if np.isfinite(initial).any() and float(np.nanmean(initial)) > 1.0e-12:
        initial = initial / max(float(np.nanmean(initial)), 1.0e-12)
    initial = np.clip(
        initial,
        float(getattr(config, "DYNAMIC_SYNTHETIC_WEIGHT_MIN", 0.05)),
        float(getattr(config, "DYNAMIC_SYNTHETIC_WEIGHT_MAX", 3.0)),
    )
    scarcity, bin_ids, bin_edges, train_bin_counts = synthetic_scarcity_bonus(
        synthetic_bundle.y_raw,
        data_bundle.y_train_raw,
        int(getattr(config, "DYNAMIC_SYNTHETIC_SCARCITY_BINS", 10)),
    )
    cluster_ids: np.ndarray | None = None
    if cluster_model is not None and cluster_model.is_fitted:
        syn_x_scaled = synthetic_bundle.loader.dataset.x.detach().cpu().numpy()
        cluster_ids = cluster_model.predict(syn_x_scaled).astype(np.int64)
    selected_indices = np.arange(n, dtype=np.int64)
    selected_mask = np.ones(n, dtype=bool)
    return DynamicSyntheticState(
        weights=initial.astype(np.float64),
        selected_indices=selected_indices,
        selected_mask=selected_mask,
        previous_raw_weights=initial.astype(np.float64),
        scarcity_bonus=scarcity.astype(np.float64),
        bin_ids=bin_ids.astype(np.int64),
        bin_edges=bin_edges.astype(np.float64),
        train_bin_counts=train_bin_counts.astype(np.int64),
        feedback_features=zero_agent_feedback_features(n),
        feedback_target=np.zeros(n, dtype=np.float64),
        cluster_ids=cluster_ids,
        cross_run_validation_std=_validated_cross_run_std(cross_run_validation_std),
    )


def compute_dynamic_synthetic_weights(
    current_weights: np.ndarray,
    agent_policy_score: np.ndarray,
    synthetic_mse: np.ndarray,
    train_region_mse: np.ndarray,
    process_consistency: np.ndarray,
    mechanism_consistency: np.ndarray,
    scarcity_bonus: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute CBTG retention scores and sample weights."""
    process = _unit_interval(process_consistency, neutral=1.0)
    mechanism = _unit_interval(mechanism_consistency, neutral=1.0)
    agent_policy = np.clip(
        np.nan_to_num(np.asarray(agent_policy_score, dtype=np.float64), nan=0.5),
        0.0,
        1.0,
    )
    reliability = (
        agent_policy
        * np.power(process, 1.0)
        * np.power(mechanism, 1.0)
    )
    synthetic_error_score = _normalize_positive(synthetic_mse, neutral=0.5)
    train_region_score = _normalize_positive(train_region_mse, neutral=0.5)
    scarcity = _unit_interval(scarcity_bonus, neutral=0.0)
    rare_condition_bonus = 1.0 + scarcity
    selection_score = reliability * rare_condition_bonus
    value = agent_policy
    min_weight = float(getattr(config, "DYNAMIC_SYNTHETIC_WEIGHT_MIN", 0.05))
    max_weight = float(getattr(config, "DYNAMIC_SYNTHETIC_WEIGHT_MAX", 3.0))
    mean_score = float(np.mean(selection_score)) if len(selection_score) else 0.0
    raw = selection_score / max(mean_score, 1.0e-12)
    new_weights = np.clip(raw, min_weight, max_weight)
    reliable_mask = selection_score > 0.0
    return {
        "weights": new_weights.astype(np.float64),
        "raw_weights": raw.astype(np.float64),
        "reliability": reliability.astype(np.float64),
        "learning_value": value.astype(np.float64),
        "agent_policy_score": agent_policy.astype(np.float64),
        "selection_score": selection_score.astype(np.float64),
        "synthetic_error_score": synthetic_error_score.astype(np.float64),
        "train_region_score": train_region_score.astype(np.float64),
        "scarcity_bonus": rare_condition_bonus.astype(np.float64),
        "reliable_mask": reliable_mask,
    }


def select_top_synthetic_indices(
    score: np.ndarray,
    top_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable top-ratio indices and a boolean keep mask for synthetic rows."""
    scores = np.asarray(score, dtype=np.float64).reshape(-1)
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=bool)
    ratio = float(top_ratio if top_ratio is not None else getattr(config, "DYNAMIC_SYNTHETIC_TOP_RATIO", 0.60))
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"DYNAMIC_SYNTHETIC_TOP_RATIO must be in (0, 1], got {ratio}.")
    keep_n = max(1, int(np.ceil(n * ratio)))
    finite_scores = np.where(np.isfinite(scores), scores, -np.inf)
    ranked = np.lexsort((np.arange(n, dtype=np.int64), -finite_scores))
    selected_indices = np.sort(ranked[:keep_n]).astype(np.int64)
    mask = np.zeros(n, dtype=bool)
    mask[selected_indices] = True
    return selected_indices, mask


def rebuild_synthetic_loader(
    synthetic_bundle: SyntheticBundle,
    dynamic_state: DynamicSyntheticState | None = None,
) -> DataLoader:
    batch_size = int(getattr(config, "SYNTHETIC_BATCH_SIZE", config.BATCH_SIZE))
    base_dataset = synthetic_bundle.loader.dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    if dynamic_state is not None and bool(getattr(config, "DYNAMIC_SYNTHETIC_USE_SAMPLER", True)):
        selected = np.asarray(dynamic_state.selected_indices, dtype=np.int64).reshape(-1)
        if len(selected) < 1:
            selected, selected_mask = select_top_synthetic_indices(
                dynamic_state.weights,
                top_ratio=0.60,
            )
            dynamic_state.selected_indices = selected
            dynamic_state.selected_mask = selected_mask
        synthetic_bundle.loader = DataLoader(
            Subset(base_dataset, selected.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
        )
    else:
        synthetic_bundle.loader = DataLoader(
            base_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
        )
    return synthetic_bundle.loader


def _train_score_from_metrics(metrics: dict[str, float]) -> float:
    metric = str(getattr(config, "DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC", "rmse_tail")).lower()
    rmse = float(metrics.get("RMSE", float("inf")))
    tail_mae = float(metrics.get("TAIL_MAE", metrics.get("Tail_MAE", float("nan"))))
    if metric in {"rmse_tail", "rmse_plus_tail"}:
        tail_lambda = float(getattr(config, "DYNAMIC_SYNTHETIC_TRAIN_TAIL_LAMBDA", 0.25))
        return rmse + (0.0 if not np.isfinite(tail_mae) else tail_lambda * tail_mae)
    if metric == "mae":
        return float(metrics.get("MAE", rmse))
    return rmse


def refresh_dynamic_synthetic_weights(
    model: torch.nn.Module,
    quality_agent: SyntheticQualityAgent,
    synthetic_bundle: SyntheticBundle,
    data_bundle: DataBundle,
    train_tensors: TrainTensorBundle,
    dynamic_state: DynamicSyntheticState,
    device: torch.device,
    epoch: int,
    cross_run_validation_std: np.ndarray,
    cluster_model: WorkingConditionCluster | None = None,
    optimizer: AdamW | None = None,
) -> dict[str, float]:
    """Re-score synthetic rows and keep the top-scoring subset for the next pretraining stage."""
    model_was_training = model.training
    agent_was_training = quality_agent.training
    model.eval()
    quality_agent.eval()
    n = len(synthetic_bundle.frame)
    agent_policy = np.ones(n, dtype=np.float64)
    synthetic_mse = np.zeros(n, dtype=np.float64)
    train_region_mse = np.zeros(n, dtype=np.float64)
    base_dataset = synthetic_bundle.loader.dataset
    while isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    eval_loader = DataLoader(
        base_dataset,
        batch_size=int(getattr(config, "SYNTHETIC_BATCH_SIZE", config.BATCH_SIZE)),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    with torch.no_grad():
        for x, y, sample_ids in eval_loader:
            x = x.to(device)
            y = y.to(device)
            sample_ids_np = sample_ids.detach().cpu().numpy().reshape(-1)
            outputs = model(x)
            y_pred = _output_mu(outputs).reshape(y.shape[0], -1)
            y_true = y.reshape(y.shape[0], -1)
            synthetic_batch_mse = ((y_pred - y_true) ** 2).mean(dim=-1)
            mse_real, _, _, _ = _real_mse_reward_for_indices(
                model,
                train_tensors,
                synthetic_bundle.nearest_train_index[sample_ids_np],
                device,
            )
            synthetic_mse[sample_ids_np] = synthetic_batch_mse.detach().cpu().numpy().reshape(-1)
            train_region_mse[sample_ids_np] = mse_real.detach().cpu().numpy().reshape(-1)

    run_std = _validated_cross_run_std(cross_run_validation_std)
    if dynamic_state.cross_run_validation_std is None or not np.array_equal(
        run_std,
        dynamic_state.cross_run_validation_std,
    ):
        raise RuntimeError("CBTG cross-run validation statistics changed during training.")
    validation_feedback = evaluate_validation_pretrain_feedback(
        model,
        data_bundle,
        device,
        run_std,
        cluster_model=cluster_model,
    )
    sample_cluster_ids = (
        np.asarray(dynamic_state.cluster_ids, dtype=np.int64)
        if dynamic_state.cluster_ids is not None
        else np.zeros(n, dtype=np.int64)
    )
    feedback_parts = build_paper_cbtg_feedback(
        np.asarray(validation_feedback["overall_metrics"], dtype=np.float64),
        np.asarray(validation_feedback["run_std"], dtype=np.float64),
        np.asarray(validation_feedback["per_cluster_metrics"], dtype=np.float64),
        sample_cluster_ids,
    )
    dynamic_state.feedback_features = feedback_parts["features"].astype(np.float64)
    dynamic_state.feedback_target = feedback_parts["target"].astype(np.float64)

    refresh_agent_losses: list[float] = []
    if optimizer is not None:
        quality_agent.train()
        for x, y, sample_ids in eval_loader:
            x = x.to(device)
            y = y.to(device)
            sample_ids_np = sample_ids.detach().cpu().numpy().reshape(-1)
            feedback_batch = torch.tensor(
                dynamic_state.feedback_features[sample_ids_np],
                dtype=x.dtype,
                device=device,
            )
            reward_batch = torch.tensor(
                dynamic_state.feedback_target[sample_ids_np],
                dtype=y.dtype,
                device=device,
            )
            policy_batch = quality_agent(x, y, feedback_batch).reshape(-1)
            refresh_agent_loss, _ = paper_cbtg_agent_loss(policy_batch, reward_batch)
            optimizer.zero_grad(set_to_none=True)
            refresh_agent_loss.backward()
            torch.nn.utils.clip_grad_norm_(quality_agent.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            refresh_agent_losses.append(float(refresh_agent_loss.detach().cpu()))
        quality_agent.eval()

    with torch.no_grad():
        for x, y, sample_ids in eval_loader:
            x = x.to(device)
            y = y.to(device)
            sample_ids_np = sample_ids.detach().cpu().numpy().reshape(-1)
            feedback_batch = torch.tensor(
                dynamic_state.feedback_features[sample_ids_np],
                dtype=x.dtype,
                device=device,
            )
            policy_batch = quality_agent(x, y, feedback_batch).reshape(-1).clamp(0.0, 1.0)
            agent_policy[sample_ids_np] = policy_batch.detach().cpu().numpy().reshape(-1)

    weight_parts = compute_dynamic_synthetic_weights(
        current_weights=dynamic_state.weights,
        agent_policy_score=agent_policy,
        synthetic_mse=synthetic_mse,
        train_region_mse=train_region_mse,
        process_consistency=synthetic_bundle.process_consistency,
        mechanism_consistency=synthetic_bundle.mechanism_consistency,
        scarcity_bonus=dynamic_state.scarcity_bonus,
    )
    process = _unit_interval(synthetic_bundle.process_consistency, neutral=1.0)
    mechanism = _unit_interval(synthetic_bundle.mechanism_consistency, neutral=1.0)
    scarcity = weight_parts["scarcity_bonus"]
    reliability = weight_parts["reliability"]
    value = weight_parts["learning_value"]
    raw = weight_parts["raw_weights"]
    new_weights = weight_parts["weights"]
    reliable_mask = weight_parts["reliable_mask"].astype(bool)
    agent_policy_score = weight_parts["agent_policy_score"]
    selection_score = weight_parts["selection_score"]
    quota_multiplier = np.ones(n, dtype=np.float64)

    validation_score = float(
        np.dot(
            PAPER_CBTG_METRIC_WEIGHTS,
            np.asarray(validation_feedback["overall_metrics"], dtype=np.float64),
        )
    )
    previous = dynamic_state.previous_train_score
    validation_score_improvement = (
        0.0 if previous is None or not np.isfinite(previous) else previous - validation_score
    )
    dynamic_state.previous_train_score = validation_score
    dynamic_state.previous_raw_weights = raw.astype(np.float64)
    dynamic_state.weights = new_weights.astype(np.float64)
    top_ratio = 0.60
    selected_indices, selected_mask = select_top_synthetic_indices(selection_score, top_ratio=top_ratio)
    dynamic_state.selected_indices = selected_indices
    dynamic_state.selected_mask = selected_mask
    dynamic_state.refresh_count += 1

    frame_dict = {
        "epoch": int(epoch),
        "sample_index": np.arange(n, dtype=np.int64),
        "dynamic_weight": dynamic_state.weights,
        "raw_dynamic_weight": raw,
        "agent_quality_score": agent_policy,
        "agent_policy_score": agent_policy_score,
        "retention_score": selection_score,
        "synthetic_mse": synthetic_mse,
        "train_region_mse": train_region_mse,
        "process_consistency": process,
        "mechanism_consistency": mechanism,
        "scarcity_bonus": scarcity,
        "reliability": reliability,
        "learning_value": value,
        "agent_feedback_target": dynamic_state.feedback_target,
        "agent_quota_multiplier": quota_multiplier,
        "reliable_for_dynamic_weight": reliable_mask,
        "selected_for_pretrain": dynamic_state.selected_mask,
        "synthetic_label_bin": dynamic_state.bin_ids,
    }
    for feature_idx, feature_name in enumerate(AGENT_FEEDBACK_FEATURE_NAMES):
        frame_dict[f"agent_state_z_{feature_name}"] = dynamic_state.feedback_features[:, feature_idx]
    if dynamic_state.cluster_ids is not None:
        frame_dict["synthetic_cluster_id"] = dynamic_state.cluster_ids
    weight_frame = pd.DataFrame(frame_dict)
    weights_path = config.RESULT_DIR / f"dynamic_synthetic_weights_epoch_{epoch:03d}.csv"
    weight_frame.to_csv(weights_path, index=False, encoding="utf-8")
    if model_was_training:
        model.train()
    if agent_was_training:
        quality_agent.train()
    return {
        "dynamic_refresh_epoch": float(epoch),
        "dynamic_refresh_count": float(dynamic_state.refresh_count),
        "dynamic_weight_mean": float(np.mean(dynamic_state.weights)) if n else float("nan"),
        "dynamic_weight_std": float(np.std(dynamic_state.weights)) if n else float("nan"),
        "dynamic_weight_min": float(np.min(dynamic_state.weights)) if n else float("nan"),
        "dynamic_weight_max": float(np.max(dynamic_state.weights)) if n else float("nan"),
        "dynamic_top_ratio": top_ratio,
        "dynamic_selected_count": float(len(dynamic_state.selected_indices)),
        "dynamic_selected_ratio": float(np.mean(dynamic_state.selected_mask)) if n else float("nan"),
        "dynamic_selected_score_min": float(np.min(selection_score[dynamic_state.selected_mask]))
        if len(dynamic_state.selected_indices) else float("nan"),
        "dynamic_reliability_mean": float(np.mean(reliability)) if n else float("nan"),
        "dynamic_learning_value_mean": float(np.mean(value)) if n else float("nan"),
        "dynamic_scarcity_bonus_mean": float(np.mean(scarcity)) if n else float("nan"),
        "dynamic_agent_policy_score_mean": float(np.mean(agent_policy_score)) if n else float("nan"),
        "dynamic_agent_feedback_target_mean": float(np.mean(dynamic_state.feedback_target)) if n else float("nan"),
        "dynamic_refresh_agent_loss": float(np.mean(refresh_agent_losses))
        if refresh_agent_losses
        else float("nan"),
        "dynamic_agent_quota_multiplier_mean": float(np.mean(quota_multiplier)) if n else float("nan"),
        "dynamic_agent_quota_multiplier_min": float(np.min(quota_multiplier)) if n else float("nan"),
        "dynamic_agent_quota_multiplier_max": float(np.max(quota_multiplier)) if n else float("nan"),
        "dynamic_reliable_ratio": float(np.mean(reliable_mask)) if n else float("nan"),
        "dynamic_validation_rmse": float(validation_feedback["validation_rmse"]),
        "dynamic_validation_mae": float(validation_feedback["validation_mae"]),
        "dynamic_validation_mape": float(validation_feedback["validation_mape"]),
        "dynamic_validation_one_minus_r2": float(validation_feedback["validation_one_minus_r2"]),
        "dynamic_validation_tail_mae": float(validation_feedback["validation_tail_mae"]),
        "dynamic_validation_reward_score": float(validation_score),
        "dynamic_validation_score_improvement": float(validation_score_improvement),
        "dynamic_reward_min": float(np.min(dynamic_state.feedback_target)) if n else float("nan"),
        "dynamic_reward_max": float(np.max(dynamic_state.feedback_target)) if n else float("nan"),
        "dynamic_reward_clip_min": PAPER_CBTG_REWARD_MIN,
        "dynamic_reward_clip_max": PAPER_CBTG_REWARD_MAX,
        "dynamic_weights_path": str(weights_path),
        "dynamic_cluster_rmse_variance": float(feedback_parts["cluster_variance"][0]),
        "dynamic_cluster_mae_variance": float(feedback_parts["cluster_variance"][1]),
        "dynamic_cluster_mape_variance": float(feedback_parts["cluster_variance"][2]),
        "dynamic_cluster_one_minus_r2_variance": float(feedback_parts["cluster_variance"][3]),
        "dynamic_cluster_tail_mae_variance": float(feedback_parts["cluster_variance"][4]),
    }


def _output_mu(outputs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    return outputs if torch.is_tensor(outputs) else outputs["mu"]


def paper_cbtg_agent_loss(
    confidence: torch.Tensor,
    reward: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact contextual-bandit objective in the paper's Eq. (9)."""
    c = confidence.reshape(-1).clamp(1.0e-6, 1.0 - 1.0e-6)
    r = reward.detach().reshape_as(c).clamp(PAPER_CBTG_REWARD_MIN, PAPER_CBTG_REWARD_MAX)
    expected_reward_loss = -(c * r).mean()
    mean_regularizer = (c.mean() - PAPER_CBTG_TARGET_CONFIDENCE) ** 2
    entropy = -(
        c * torch.log(c)
        + (1.0 - c) * torch.log(1.0 - c)
    )
    total = (
        expected_reward_loss
        + PAPER_CBTG_LAMBDA_M * mean_regularizer
        - PAPER_CBTG_LAMBDA_H * entropy.mean()
    )
    return total, {
        "expected_reward_loss": expected_reward_loss,
        "mean_regularizer": mean_regularizer,
        "confidence_entropy": entropy.mean(),
    }


def _synthetic_step(
    model: torch.nn.Module,
    quality_agent: SyntheticQualityAgent,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: AdamW,
    synthetic_bundle: SyntheticBundle,
    train_tensors: TrainTensorBundle,
    device: torch.device,
    dynamic_state: DynamicSyntheticState | None = None,
    update_quality_agent: bool = True,
) -> dict[str, float]:
    x, y, sample_ids = batch
    x = x.to(device)
    y = y.to(device)
    sample_ids_np = sample_ids.detach().cpu().numpy().reshape(-1)
    is_tail = torch.tensor(synthetic_bundle.is_tail[sample_ids_np], dtype=y.dtype, device=device)
    outputs = model(x)
    if torch.is_tensor(outputs):
        outputs = {"mu": outputs}
    feedback_batch = None
    feedback_reward = None
    if dynamic_state is not None:
        feedback_batch = torch.tensor(
            dynamic_state.feedback_features[sample_ids_np],
            dtype=x.dtype,
            device=device,
        )
        feedback_reward = torch.tensor(
            dynamic_state.feedback_target[sample_ids_np],
            dtype=y.dtype,
            device=device,
        ).reshape(-1)
    if update_quality_agent:
        quality_score = quality_agent(x, y, feedback_batch).reshape(-1).clamp(1.0e-6, 1.0 - 1.0e-6)
    else:
        with torch.no_grad():
            quality_score = quality_agent(x, y, feedback_batch).reshape(-1).clamp(1.0e-6, 1.0 - 1.0e-6)
    mse_real, _, _, _ = _real_mse_reward_for_indices(
        model,
        train_tensors,
        synthetic_bundle.nearest_train_index[sample_ids_np],
        device,
    )
    mse_real = mse_real.to(device=device, dtype=y.dtype)
    process_consistency = torch.tensor(
        synthetic_bundle.process_consistency[sample_ids_np],
        dtype=y.dtype,
        device=device,
    ).clamp(0.0, 1.0)
    mechanism_consistency = torch.tensor(
        synthetic_bundle.mechanism_consistency[sample_ids_np],
        dtype=y.dtype,
        device=device,
    ).clamp(0.0, 1.0)
    reward_parts = synthetic_agent_reward_components(mse_real, process_consistency, mechanism_consistency)
    fallback_reward = 2.0 * reward_parts["reward"].to(device=device, dtype=y.dtype) - 1.0
    reward = (
        feedback_reward.clamp(PAPER_CBTG_REWARD_MIN, PAPER_CBTG_REWARD_MAX)
        if feedback_reward is not None
        else fallback_reward.clamp(PAPER_CBTG_REWARD_MIN, PAPER_CBTG_REWARD_MAX)
    )
    final_confidence = quality_score
    selected = torch.ones_like(final_confidence, dtype=torch.bool)

    dynamic_batch_weight = None
    if (
        dynamic_state is not None
        and bool(getattr(config, "DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT", True))
    ):
        dynamic_batch_weight = torch.tensor(
            dynamic_state.weights[sample_ids_np],
            dtype=y.dtype,
            device=device,
        ).reshape(-1, 1)
    elif bool(getattr(config, "SYNTHETIC_USE_AGENT_WEIGHT", True)):
        dynamic_batch_weight = final_confidence.detach().reshape(-1, 1)

    loss_outputs = dict(outputs)
    loss_outputs.pop("sample_confidence", None)
    loss_outputs.pop("training_weight", None)
    total, base_loss_logs = paper_total_loss(
        loss_outputs,
        y,
        x=x,
        batch_weights=dynamic_batch_weight,
    )

    agent_loss = torch.tensor(0.0, device=device)
    agent_logs: dict[str, float] = {}
    if update_quality_agent and bool(getattr(config, "SYNTHETIC_USE_REWARD_LOSS", True)):
        agent_loss, agent_components = paper_cbtg_agent_loss(quality_score, reward)
        agent_logs = {
            "agent_reward_loss": float(agent_components["expected_reward_loss"].detach().cpu()),
            "agent_total_loss": float(agent_loss.detach().cpu()),
            "agent_confidence_mean_regularizer": float(agent_components["mean_regularizer"].detach().cpu()),
            "agent_confidence_entropy": float(agent_components["confidence_entropy"].detach().cpu()),
            "reward_mean": float(reward.detach().mean().cpu()),
            "reward_std": float(reward.detach().std(unbiased=False).cpu()),
            "reward_min": float(reward.detach().min().cpu()),
            "reward_max": float(reward.detach().max().cpu()),
            "reward_mse_mean": float(reward_parts["reward_mse"].detach().mean().cpu()),
            "reward_process_mean": float(reward_parts["reward_process"].detach().mean().cpu()),
            "reward_mechanism_mean": float(reward_parts["reward_mechanism"].detach().mean().cpu()),
            "reward_feedback_mean": float(feedback_reward.detach().mean().cpu()) if feedback_reward is not None else float("nan"),
            "mse_real_mean": float(mse_real.detach().mean().cpu()),
            "mse_real_std": float(mse_real.detach().std(unbiased=False).cpu()),
        }
        total = total + float(getattr(config, "AGENT_REWARD_LAMBDA", 0.01)) * agent_loss

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
    if update_quality_agent:
        torch.nn.utils.clip_grad_norm_(quality_agent.parameters(), config.GRAD_CLIP_NORM)
    optimizer.step()

    gate_probs = outputs.get("gate_probs")
    entropy = (
        -(gate_probs.detach() * torch.log(gate_probs.detach() + 1.0e-8)).sum(dim=-1).mean()
        if gate_probs is not None
        else torch.tensor(float("nan"), device=device)
    )
    expert_uncertainty = outputs.get("expert_uncertainty")
    if expert_uncertainty is None and "expert_preds" in outputs:
        expert_uncertainty = outputs["expert_preds"].detach().var(dim=1, unbiased=False)
    reward_mean = agent_logs.get("reward_mean", float("nan"))
    reward_std = agent_logs.get("reward_std", float("nan"))
    return {
        "synthetic_pred_loss": float(base_loss_logs.get("pred_loss", float("nan"))),
        "synthetic_moe_aux_loss": float(base_loss_logs.get("moe_loss", float("nan"))),
        "synthetic_expert_calibration_loss": float(base_loss_logs.get("expert_calibration_loss", 0.0)),
        "synthetic_expert_diversity_loss": float(base_loss_logs.get("expert_diversity_loss", 0.0)),
        "synthetic_graph_loss": float(base_loss_logs.get("graph_loss", 0.0)),
        "synthetic_agent_reward_loss": agent_logs.get("agent_reward_loss", float(agent_loss.detach().cpu())),
        "synthetic_agent_total_loss": agent_logs.get("agent_total_loss", float(agent_loss.detach().cpu())),
        "synthetic_total_loss": float(total.detach().cpu()),
        "synthetic_reward_mean": reward_mean,
        "synthetic_reward_std": reward_std,
        "synthetic_reward_min": agent_logs.get("reward_min", float("nan")),
        "synthetic_reward_max": agent_logs.get("reward_max", float("nan")),
        "synthetic_reward_mse_mean": agent_logs.get("reward_mse_mean", float("nan")),
        "synthetic_reward_process_mean": agent_logs.get("reward_process_mean", float("nan")),
        "synthetic_reward_mechanism_mean": agent_logs.get("reward_mechanism_mean", float("nan")),
        "synthetic_reward_feedback_mean": agent_logs.get("reward_feedback_mean", float("nan")),
        "synthetic_mse_real_mean": agent_logs.get("mse_real_mean", float("nan")),
        "synthetic_mse_real_std": agent_logs.get("mse_real_std", float("nan")),
        "synthetic_confidence_mean": float(quality_score.detach().mean().cpu()),
        "synthetic_confidence_std": float(quality_score.detach().std(unbiased=False).cpu()),
        "synthetic_keep_score_mean": float(quality_score.detach().mean().cpu()),
        "synthetic_keep_score_std": float(quality_score.detach().std(unbiased=False).cpu()),
        "synthetic_quality_score_mean": float(quality_score.detach().mean().cpu()),
        "synthetic_quality_score_std": float(quality_score.detach().std(unbiased=False).cpu()),
        "synthetic_process_consistency_mean": float(process_consistency.detach().mean().cpu()),
        "synthetic_process_consistency_std": float(process_consistency.detach().std(unbiased=False).cpu()),
        "synthetic_mechanism_consistency_mean": float(mechanism_consistency.detach().mean().cpu()),
        "synthetic_mechanism_consistency_std": float(mechanism_consistency.detach().std(unbiased=False).cpu()),
        "synthetic_final_weight_mean": float(final_confidence.detach().mean().cpu()),
        "synthetic_final_weight_std": float(final_confidence.detach().std(unbiased=False).cpu()),
        "synthetic_selected_ratio": float(selected.detach().float().mean().cpu()),
        "synthetic_tail_ratio": float(is_tail.detach().float().mean().cpu()),
        "dynamic_synthetic_weight_mean": float(dynamic_batch_weight.detach().mean().cpu()) if dynamic_batch_weight is not None else float("nan"),
        "dynamic_synthetic_weight_std": float(dynamic_batch_weight.detach().std(unbiased=False).cpu()) if dynamic_batch_weight is not None else float("nan"),
        "synthetic_expert_uncertainty_mean": float(expert_uncertainty.detach().mean().cpu()) if expert_uncertainty is not None else float("nan"),
        "synthetic_gate_entropy": float(entropy.detach().cpu()),
        "synthetic_agent_update_active": float(update_quality_agent),
    }


def _average_logs(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def calibrate_quality_agent_on_real(
    model: torch.nn.Module,
    quality_agent: torch.nn.Module,
    data_bundle: DataBundle,
    device: torch.device,
    epoch: int,
    *,
    optimizer: AdamW,
    cross_run_validation_std: np.ndarray,
    cluster_model: WorkingConditionCluster,
) -> tuple[
    dict[str, float],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]:
    validation_feedback = evaluate_validation_pretrain_feedback(
        model,
        data_bundle,
        device,
        cross_run_validation_std,
        cluster_model=cluster_model,
    )
    quality_agent.train()
    losses: list[float] = []
    reward_losses: list[float] = []
    mean_regularizers: list[float] = []
    entropies: list[float] = []
    for batch in data_bundle.train_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        cluster_ids = cluster_model.predict_tensor(x).numpy()
        feedback = build_paper_cbtg_feedback(
            np.asarray(validation_feedback["overall_metrics"], dtype=np.float64),
            np.asarray(validation_feedback["cross_run_validation_std"], dtype=np.float64),
            np.asarray(validation_feedback["per_cluster_metrics"], dtype=np.float64),
            cluster_ids,
        )
        features = torch.tensor(feedback["features"], dtype=x.dtype, device=device)
        reward = torch.tensor(feedback["target"], dtype=y.dtype, device=device)
        confidence = quality_agent(x, y, features).reshape(-1)
        loss, parts = paper_cbtg_agent_loss(confidence, reward)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(quality_agent.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        reward_losses.append(float(parts["expected_reward_loss"].detach().cpu()))
        mean_regularizers.append(float(parts["mean_regularizer"].detach().cpu()))
        entropies.append(float(parts["confidence_entropy"].detach().cpu()))
    if not losses:
        raise ValueError("CBTG-Agent requires a non-empty real training loader.")
    quality_agent.eval()

    def real_batch_weights(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cluster_ids = cluster_model.predict_tensor(x).numpy()
        feedback = build_paper_cbtg_feedback(
            np.asarray(validation_feedback["overall_metrics"], dtype=np.float64),
            np.asarray(validation_feedback["cross_run_validation_std"], dtype=np.float64),
            np.asarray(validation_feedback["per_cluster_metrics"], dtype=np.float64),
            cluster_ids,
        )
        features = torch.tensor(feedback["features"], dtype=x.dtype, device=x.device)
        with torch.no_grad():
            return quality_agent(x, y, features).reshape(-1, 1).detach()

    logs = {
        "real_cbtg_agent_epoch": float(epoch),
        "real_cbtg_agent_loss": float(np.mean(losses)),
        "real_cbtg_expected_reward_loss": float(np.mean(reward_losses)),
        "real_cbtg_mean_regularizer": float(np.mean(mean_regularizers)),
        "real_cbtg_confidence_entropy": float(np.mean(entropies)),
        "real_cbtg_validation_rmse": float(validation_feedback["validation_rmse"]),
        "real_cbtg_validation_mae": float(validation_feedback["validation_mae"]),
        "real_cbtg_validation_mape": float(validation_feedback["validation_mape"]),
        "real_cbtg_validation_one_minus_r2": float(validation_feedback["validation_one_minus_r2"]),
        "real_cbtg_validation_tail_mae": float(validation_feedback["validation_tail_mae"]),
    }
    return logs, real_batch_weights


def build_synthetic_pretrain_optimizer(
    model: torch.nn.Module,
    quality_agent: torch.nn.Module,
) -> AdamW:
    pretrain_lr = float(config.LR)
    agent_lr = float(getattr(config, "SYNTHETIC_AGENT_LR", pretrain_lr))
    if not np.isclose(agent_lr, pretrain_lr, rtol=0.0, atol=1.0e-12):
        raise ValueError("Synthetic model and CBTG-Agent must use the same pretraining learning rate.")
    return AdamW(
        [
            {"params": model.parameters(), "name": "synthetic_model"},
            {"params": quality_agent.parameters(), "name": "synthetic_quality_agent"},
        ],
        lr=pretrain_lr,
        weight_decay=config.WEIGHT_DECAY,
    )


def quality_agent_updates_enabled(epoch: int) -> bool:
    agent_epochs = int(getattr(config, "SYNTHETIC_AGENT_EPOCHS", 0))
    if agent_epochs < 1:
        raise ValueError("SYNTHETIC_AGENT_EPOCHS must be positive.")
    return int(epoch) <= agent_epochs


def dynamic_synthetic_refresh_due(
    epoch: int,
    refresh_epochs: int | None = None,
    warmup_epochs: int | None = None,
) -> bool:
    refresh = int(
        getattr(config, "DYNAMIC_SYNTHETIC_REFRESH_EPOCHS", 5)
        if refresh_epochs is None
        else refresh_epochs
    )
    warmup = int(
        getattr(config, "DYNAMIC_SYNTHETIC_WARMUP_EPOCHS", 0)
        if warmup_epochs is None
        else warmup_epochs
    )
    if refresh < 1 or warmup < 0:
        raise ValueError("Dynamic refresh must be positive and warmup must be non-negative.")
    current = int(epoch)
    return current > warmup and (current - warmup - 1) % refresh == 0


def pretrain_with_cbtg(
    model: torch.nn.Module,
    quality_agent: SyntheticQualityAgent,
    synthetic_bundle: SyntheticBundle,
    data_bundle: DataBundle,
    train_tensors: TrainTensorBundle,
    device: torch.device,
    cross_run_validation_std: np.ndarray,
    epochs: int | None = None,
    cluster_model: WorkingConditionCluster | None = None,
) -> Path:
    epochs = int(epochs or getattr(config, "SYNTHETIC_PRETRAIN_EPOCHS", 20))
    run_std = _validated_cross_run_std(cross_run_validation_std)
    pretrain_lr = float(config.LR)
    agent_epochs = int(getattr(config, "SYNTHETIC_AGENT_EPOCHS", epochs))
    if agent_epochs < 1:
        raise ValueError("SYNTHETIC_AGENT_EPOCHS must be positive.")
    optimizer = build_synthetic_pretrain_optimizer(model, quality_agent)
    refresh_epochs = int(getattr(config, "DYNAMIC_SYNTHETIC_REFRESH_EPOCHS", 5))
    warmup_epochs = int(getattr(config, "DYNAMIC_SYNTHETIC_WARMUP_EPOCHS", 0))
    dynamic_synthetic_refresh_due(1, refresh_epochs, warmup_epochs)
    log_path = config.LOG_DIR / "cbtg_pretrain_log.csv"
    if log_path.exists():
        log_path.unlink()
    dynamic_log_path = config.LOG_DIR / "dynamic_synthetic_agent_log.csv"
    if dynamic_log_path.exists():
        dynamic_log_path.unlink()
    dynamic_state = (
        build_dynamic_synthetic_state(
            synthetic_bundle,
            data_bundle,
            run_std,
            cluster_model=cluster_model,
        )
        if bool(getattr(config, "USE_DYNAMIC_SYNTHETIC_AGENT", False))
        else None
    )
    if dynamic_state is not None:
        rebuild_synthetic_loader(synthetic_bundle, dynamic_state)
    model.train()
    quality_agent.train()
    start = time.perf_counter()
    progress = tqdm(range(1, epochs + 1), desc="Synthetic pretrain", unit="epoch")
    for epoch in progress:
        update_quality_agent = quality_agent_updates_enabled(epoch)
        quality_agent.requires_grad_(update_quality_agent)
        quality_agent.train(update_quality_agent)
        refresh_logs: dict[str, float | str] = {}
        if dynamic_state is not None:
            should_refresh = dynamic_synthetic_refresh_due(epoch, refresh_epochs, warmup_epochs)
            if should_refresh:
                refresh_logs = refresh_dynamic_synthetic_weights(
                    model,
                    quality_agent,
                    synthetic_bundle,
                    data_bundle,
                    train_tensors,
                    dynamic_state,
                    device,
                    epoch,
                    run_std,
                    cluster_model=cluster_model,
                    optimizer=optimizer if update_quality_agent else None,
                )
                rebuild_synthetic_loader(synthetic_bundle, dynamic_state)
                append_csv(dynamic_log_path, refresh_logs)
                model.train()
                quality_agent.train()
        batch_logs: list[dict[str, float]] = []
        for batch in synthetic_bundle.loader:
            batch_logs.append(
                _synthetic_step(
                    model,
                    quality_agent,
                    batch,
                    optimizer,
                    synthetic_bundle,
                    train_tensors,
                    device,
                    dynamic_state=dynamic_state,
                    update_quality_agent=update_quality_agent,
                )
            )
        avg_logs = _average_logs(batch_logs)
        row = {
            "epoch": epoch,
            "elapsed_train_time": time.perf_counter() - start,
            "use_dynamic_synthetic_agent": bool(dynamic_state is not None),
            "dynamic_synthetic_refresh_epochs": refresh_epochs,
            "dynamic_synthetic_warmup_epochs": warmup_epochs,
            "dynamic_synthetic_use_sampler": bool(getattr(config, "DYNAMIC_SYNTHETIC_USE_SAMPLER", True)),
            "dynamic_synthetic_use_loss_weight": bool(getattr(config, "DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT", True)),
            "dynamic_synthetic_top_ratio": 0.60,
            "synthetic_pretrain_lr": pretrain_lr,
            "synthetic_agent_epochs": agent_epochs,
            "synthetic_agent_update_active": update_quality_agent,
            **avg_logs,
            **refresh_logs,
        }
        append_csv(log_path, row)
        loss_value = avg_logs.get("synthetic_total_loss", avg_logs.get("total_loss", float("nan")))
        progress.set_postfix(loss=f"{loss_value:.4f}")
    quality_agent.requires_grad_(True)
    ckpt_path = config.CHECKPOINT_DIR / "cbtg_pretrained_model.pth"
    torch.save(
        checkpoint_payload(
            model,
            data_bundle,
            epochs,
            node_names=getattr(model, "node_names", data_bundle_node_names(model)),
            synthetic_pretrain=True,
            synthetic_size=int(len(synthetic_bundle.frame)),
            synthetic_quality_agent_state_dict=quality_agent.state_dict(),
            dynamic_synthetic_agent=bool(dynamic_state is not None),
            dynamic_synthetic_refresh_epochs=refresh_epochs,
            dynamic_synthetic_warmup_epochs=warmup_epochs,
            dynamic_synthetic_use_sampler=bool(getattr(config, "DYNAMIC_SYNTHETIC_USE_SAMPLER", True)),
            dynamic_synthetic_use_loss_weight=bool(getattr(config, "DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT", True)),
            dynamic_synthetic_top_ratio=0.60,
            synthetic_agent_feedback_features=list(AGENT_FEEDBACK_FEATURE_NAMES),
            synthetic_agent_reward_formula=(
                "R_i = clip(-sum_l beta_l[z(vbar_l) + 0.1z(s_l) + "
                "0.3z(v_l,g(i)) + 0.3z(var_l)], -1, 1) on validation predictions"
            ),
            synthetic_pretrain_confidence_rule="Top-60% by S=c*p*m*b every 5 epochs",
        ),
        ckpt_path,
    )
    return ckpt_path


def data_bundle_node_names(model: torch.nn.Module) -> list[str]:
    return list(getattr(model, "node_names", config.active_node_names()))


def _config_sha256(*, required: bool) -> str:
    value = str(getattr(config, "CONFIG_SHA256", "") or "").lower()
    valid = len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    if required and not valid:
        raise RuntimeError("Formal main training requires a stable hashed YAML configuration.")
    return value if valid else ""


def _protocol_metrics(
    data_bundle: DataBundle,
    synthetic_bundle: SyntheticBundle,
    *,
    require_config: bool = False,
) -> dict[str, object]:
    return {
        "Split_Seed": int(data_bundle.split_seed),
        "Split_Method": str(data_bundle.split_method),
        "Combined_Split_SHA256": str(data_bundle.combined_split_hash),
        "Source_Data_SHA256": str(data_bundle.source_sha256),
        "Synthetic_SHA256": str(synthetic_bundle.synthetic_sha256),
        "Synthetic_Provenance_SHA256": str(synthetic_bundle.provenance_sha256),
        "Generation_Seed": int(getattr(config, "TABDIFF_GENERATION_SEED", 0)),
        "Config_SHA256": _config_sha256(required=require_config),
    }


def run_cbtg_pretraining(
    data_bundle: DataBundle,
    synthetic_data_path: str | Path | None = None,
    synthetic_epochs: int | None = None,
    finetune_real: bool = False,
    real_epochs: int | None = None,
    cross_run_validation_stats: CrossRunValidationStats | None = None,
) -> tuple[torch.nn.Module, dict[str, float]]:
    if finetune_real:
        _config_sha256(required=True)
    ensure_dirs(config.CHECKPOINT_DIR, config.LOG_DIR, config.RESULT_DIR)
    save_split_artifacts(data_bundle, config.RESULT_DIR)
    validation_stats = cross_run_validation_stats or load_cross_run_validation_stats(data_bundle)
    set_seed(config.SEED)
    device = resolve_device()
    model = build_experiment_model().to(device)
    quality_agent = build_synthetic_quality_agent(data_bundle, device)
    synthetic_bundle = _load_synthetic_bundle(data_bundle, synthetic_data_path)
    train_tensors = _train_tensor_bundle(data_bundle, device)
    x_train_np = data_bundle.train_loader.dataset.x.detach().cpu().numpy()
    cluster_model = WorkingConditionCluster()
    cluster_model.fit(x_train_np, data_bundle.y_train_raw)
    start = time.perf_counter()
    pretrain_ckpt = pretrain_with_cbtg(
        model,
        quality_agent,
        synthetic_bundle,
        data_bundle,
        train_tensors,
        device,
        validation_stats.metric_std,
        epochs=synthetic_epochs,
        cluster_model=cluster_model,
    )
    best_epoch = -1
    epochs_run = 0
    stopped_early = False
    if finetune_real:
        real_agent_optimizer = AdamW(
            quality_agent.parameters(),
            lr=float(getattr(config, "FINETUNE_QUALITY_AGENT_LR", 1.0e-3)),
            weight_decay=config.WEIGHT_DECAY,
        )
        real_agent_calibrator = partial(
            calibrate_quality_agent_on_real,
            optimizer=real_agent_optimizer,
            cross_run_validation_std=validation_stats.metric_std,
            cluster_model=cluster_model,
        )
        _, best_epoch, epochs_run, stopped_early = supervised_finetune(
            model,
            data_bundle,
            device,
            epochs=real_epochs,
            freeze_backbone=bool(getattr(config, "FREEZE_FINETUNE_BACKBONE", False)),
            quality_agent=quality_agent,
            quality_agent_calibration_fn=real_agent_calibrator,
        )
        ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
        if ckpt_path.exists():
            load_checkpoint(model, ckpt_path, device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            agent_state = checkpoint.get("synthetic_quality_agent_state_dict")
            if agent_state is None:
                raise RuntimeError("Best checkpoint is missing the calibrated CBTG-Agent state.")
            quality_agent.load_state_dict(agent_state)

    assert_cross_run_validation_stats_current(validation_stats, data_bundle)
    metrics, collected = evaluate_model(model, data_bundle.test_loader, device, data_bundle)
    assert_cross_run_validation_stats_current(validation_stats, data_bundle)
    metrics.update(
        {
            "total_params": count_trainable_parameters(model),
            "Params": count_trainable_parameters(model),
            "Total_Params_All": count_total_parameters(model),
            "Finetune_Freeze_Backbone": bool(finetune_real and getattr(config, "FREEZE_FINETUNE_BACKBONE", False)),
            "Train_Time": time.perf_counter() - start,
            "Best_Epoch": best_epoch,
            "Epochs_Run": epochs_run,
            "Stopped_Early": bool(stopped_early),
            "Model": "mtam_hg",
            "Experiment_Group": getattr(config, "EXPERIMENT_NAME", "synthetic_pretrain"),
            "Data_Split": data_bundle.split_method,
            "Train_Size": int(data_bundle.split_sizes.get("train", 0)),
            "Synthetic_Size": int(len(synthetic_bundle.frame)),
            "Synthetic_Pretrain_Checkpoint": str(pretrain_ckpt),
            "CBTG_Cross_Run_Validation_Runs": int(validation_stats.num_runs),
            "CBTG_Cross_Run_Validation_Stats": str(validation_stats.path),
            "CBTG_Cross_Run_Validation_STD_SHA256": validation_stats.sha256,
            "CBTG_Real_Calibrated": bool(finetune_real),
            "Seed": int(config.SEED),
            **_protocol_metrics(
                data_bundle,
                synthetic_bundle,
                require_config=finetune_real,
            ),
        }
    )
    save_evaluation_outputs(metrics, collected, config.RESULT_DIR)
    return model, metrics
