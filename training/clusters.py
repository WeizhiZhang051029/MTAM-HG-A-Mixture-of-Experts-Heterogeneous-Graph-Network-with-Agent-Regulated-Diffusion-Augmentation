"""Cluster working conditions for the CBTG reward."""

from __future__ import annotations

import numpy as np
import torch

import config


class WorkingConditionCluster:
    """Cluster CAPL samples into working-condition groups."""

    def __init__(
        self,
        n_clusters: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = int(
            n_clusters
            if n_clusters is not None
            else getattr(config, "NUM_WORKING_CONDITION_CLUSTERS", 5)
        )
        if self.n_clusters < 2:
            raise ValueError(
                f"NUM_WORKING_CONDITION_CLUSTERS must be >= 2, got {self.n_clusters}."
            )
        self.random_state = int(
            random_state
            if random_state is not None
            else getattr(config, "SPLIT_SEED", 42)
        )
        self._kmeans = None
        self._is_fitted = False
        self._label_remap: np.ndarray | None = None
        self.cluster_mean_y_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray | None = None,
    ) -> "WorkingConditionCluster":
        """Fit on standardized training inputs and order C1--C5 by mean YS."""
        from sklearn.cluster import KMeans

        x = np.asarray(x_train, dtype=np.float32)
        finite_mask = np.all(np.isfinite(x), axis=1)
        x_clean = x[finite_mask] if finite_mask.any() else x
        effective_k = min(self.n_clusters, max(2, x_clean.shape[0]))
        self._kmeans = KMeans(
            n_clusters=effective_k,
            random_state=self.random_state,
            n_init=10,
        )
        self._kmeans.fit(x_clean)
        raw_labels = self._kmeans.predict(x_clean).astype(np.int64)
        self._label_remap = np.arange(effective_k, dtype=np.int64)
        self.cluster_mean_y_ = None
        if y_train is not None:
            y = np.asarray(y_train, dtype=np.float64).reshape(-1)
            if len(y) != len(x):
                raise ValueError("y_train must have the same number of rows as x_train.")
            y_clean = y[finite_mask] if finite_mask.any() else y
            means = np.array(
                [float(np.nanmean(y_clean[raw_labels == g])) for g in range(effective_k)],
                dtype=np.float64,
            )
            order = np.argsort(means, kind="stable")
            remap = np.empty(effective_k, dtype=np.int64)
            remap[order] = np.arange(effective_k, dtype=np.int64)
            self._label_remap = remap
            self.cluster_mean_y_ = means[order]
        self._is_fitted = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict cluster ids for an array of samples (numpy in, int64 out)."""
        if not self._is_fitted or self._kmeans is None:
            raise RuntimeError(
                "WorkingConditionCluster.fit() must be called before predict()."
            )
        x_arr = np.asarray(x, dtype=np.float32)
        nan_rows = ~np.all(np.isfinite(x_arr), axis=1)
        x_safe = np.where(np.isfinite(x_arr), x_arr, 0.0).astype(np.float32)
        labels = self._kmeans.predict(x_safe).astype(np.int64)
        if self._label_remap is not None:
            labels = self._label_remap[labels]
        if nan_rows.any():
            labels[nan_rows] = 0
        return labels

    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Predict cluster ids for a batch tensor; returns a CPU long tensor."""
        labels_np = self.predict(x.detach().cpu().numpy())
        return torch.from_numpy(labels_np).long()

    @property
    def effective_n_clusters(self) -> int:
        if self._kmeans is None:
            return self.n_clusters
        return int(self._kmeans.n_clusters)


def compute_cluster_balance_stats(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    cluster_labels: torch.Tensor,
    num_clusters: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Compute per-cluster metrics and variances."""
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true.reshape(-1)
    labels = cluster_labels.reshape(-1).to(device=y_pred.device)

    cluster_rmse: list[torch.Tensor] = []
    cluster_mae: list[torch.Tensor] = []
    cluster_mape: list[torch.Tensor] = []
    cluster_r2: list[torch.Tensor] = []
    details: dict[str, object] = {}
    eps = y_pred.new_tensor(1.0e-6)

    for g in range(num_clusters):
        mask = labels == g
        if mask.sum() < 1:
            continue
        yp = y_pred_flat[mask]
        yt = y_true_flat[mask]
        residuals = yp - yt

        rmse_g = residuals.pow(2).mean().clamp_min(1.0e-12).sqrt()
        mae_g = residuals.abs().mean()
        mape_g = (residuals.abs() / yt.abs().clamp_min(eps)).mean() * 100.0
        ss_res = residuals.pow(2).sum()
        ss_tot = (yt - yt.mean()).pow(2).sum().clamp_min(1.0e-12)
        r2_g = 1.0 - ss_res / ss_tot
        r2_g = r2_g.clamp(
            min=float(getattr(config, "CLUSTER_BALANCE_R2_MIN", -1.0)),
            max=float(getattr(config, "CLUSTER_BALANCE_R2_MAX", 1.0)),
        )

        cluster_rmse.append(rmse_g)
        cluster_mae.append(mae_g)
        cluster_mape.append(mape_g)
        cluster_r2.append(r2_g)
        details[f"cluster_{g}_rmse"] = float(rmse_g.detach().cpu())
        details[f"cluster_{g}_mae"] = float(mae_g.detach().cpu())
        details[f"cluster_{g}_mape"] = float(mape_g.detach().cpu())
        details[f"cluster_{g}_r2"] = float(r2_g.detach().cpu())

    if not cluster_rmse:
        zero = y_pred.new_tensor(0.0)
        return zero, zero, details

    def _var(tensors: list[torch.Tensor]) -> torch.Tensor:
        stack = torch.stack(tensors)
        return ((stack - stack.mean()) ** 2).mean()

    rmse_stack = torch.stack(cluster_rmse)
    rmse_mean = rmse_stack.mean()
    var_rmse = _var(cluster_rmse)
    var_mae = _var(cluster_mae)
    var_mape = _var(cluster_mape)
    var_r2 = _var(cluster_r2)

    details["cluster_rmse_mean"] = float(rmse_mean.detach().cpu())
    details["cluster_mae_mean"] = float(torch.stack(cluster_mae).mean().detach().cpu())
    details["cluster_mape_mean"] = float(torch.stack(cluster_mape).mean().detach().cpu())
    details["cluster_r2_mean"] = float(torch.stack(cluster_r2).mean().detach().cpu())
    details["cluster_rmse_var"] = float(var_rmse.detach().cpu())
    details["cluster_mae_var"] = float(var_mae.detach().cpu())
    details["cluster_mape_var"] = float(var_mape.detach().cpu())
    details["cluster_r2_var"] = float(var_r2.detach().cpu())
    details["cluster_active_count"] = len(cluster_rmse)
    details["var_mae_tensor"] = var_mae.detach()
    details["var_mape_tensor"] = var_mape.detach()
    details["var_r2_tensor"] = var_r2.detach()
    return rmse_mean.detach(), var_rmse.detach(), details


def compute_cluster_balance_stats_numpy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    num_clusters: int,
) -> tuple[float, float, np.ndarray, dict[str, object]]:
    """Compute NumPy per-cluster metrics and variances."""
    y_pred_flat = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true_flat = np.asarray(y_true, dtype=np.float64).reshape(-1)
    labels = np.asarray(cluster_labels, dtype=np.int64).reshape(-1)

    per_cluster_rmse = np.full(num_clusters, np.nan, dtype=np.float64)
    details: dict[str, object] = {}
    active_rmse: list[float] = []
    active_mae: list[float] = []
    active_mape: list[float] = []
    active_r2: list[float] = []

    eps = 1.0e-6
    for g in range(num_clusters):
        mask = labels == g
        if mask.sum() < 1:
            continue
        yp = y_pred_flat[mask]
        yt = y_true_flat[mask]
        residuals = yp - yt

        rmse_g = float(np.sqrt(np.maximum(np.mean(residuals ** 2), 1.0e-12)))
        mae_g = float(np.mean(np.abs(residuals)))
        mape_g = float(100.0 * np.mean(np.abs(residuals) / np.maximum(np.abs(yt), eps)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        r2_g = float(1.0 - ss_res / max(ss_tot, 1.0e-12))
        r2_g = float(
            np.clip(
                r2_g,
                float(getattr(config, "CLUSTER_BALANCE_R2_MIN", -1.0)),
                float(getattr(config, "CLUSTER_BALANCE_R2_MAX", 1.0)),
            )
        )

        per_cluster_rmse[g] = rmse_g
        active_rmse.append(rmse_g)
        active_mae.append(mae_g)
        active_mape.append(mape_g)
        active_r2.append(r2_g)
        details[f"cluster_{g}_rmse"] = rmse_g
        details[f"cluster_{g}_mae"] = mae_g
        details[f"cluster_{g}_mape"] = mape_g
        details[f"cluster_{g}_r2"] = r2_g

    if not active_rmse:
        return 0.0, 0.0, per_cluster_rmse, details

    rmse_mean = float(np.mean(active_rmse))
    var_cluster = float(np.mean([(r - rmse_mean) ** 2 for r in active_rmse]))

    mae_mean = float(np.mean(active_mae))
    var_mae = float(np.mean([(m - mae_mean) ** 2 for m in active_mae]))

    mape_mean = float(np.mean(active_mape))
    var_mape = float(np.mean([(m - mape_mean) ** 2 for m in active_mape]))

    r2_mean = float(np.mean(active_r2))
    var_r2 = float(np.mean([(r - r2_mean) ** 2 for r in active_r2]))

    details["cluster_rmse_mean"] = rmse_mean
    details["cluster_mae_mean"] = mae_mean
    details["cluster_mape_mean"] = mape_mean
    details["cluster_r2_mean"] = r2_mean
    details["cluster_rmse_var"] = var_cluster
    details["cluster_mae_var"] = var_mae
    details["cluster_mape_var"] = var_mape
    details["cluster_r2_var"] = var_r2
    details["cluster_active_count"] = len(active_rmse)
    return rmse_mean, var_cluster, per_cluster_rmse, details


def cluster_balance_reward(
    rmse_all: float,
    var_cluster: float,
    lambda_cluster: float | None = None,
    var_mae: float = 0.0,
    var_mape: float = 0.0,
    var_r2: float = 0.0,
) -> float:
    """Compute the cluster-balance Agent reward."""
    lam = float(
        lambda_cluster
        if lambda_cluster is not None
        else getattr(config, "CLUSTER_BALANCE_LAMBDA", 0.3)
    )
    w_mae = float(getattr(config, "CLUSTER_BALANCE_MAE_WEIGHT", 0.3))
    w_mape = float(getattr(config, "CLUSTER_BALANCE_MAPE_WEIGHT", 0.1))
    w_r2 = float(getattr(config, "CLUSTER_BALANCE_R2_WEIGHT", 0.3))
    var_mape_normed = var_mape / (100.0 ** 2 + 1.0e-8)
    var_composite = var_cluster + w_mae * var_mae + w_mape * var_mape_normed + w_r2 * var_r2
    return float(1.0 / (1.0 + rmse_all + lam * var_composite))


def cluster_difficulty_scores(
    per_cluster_rmse: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """Map cluster RMSE to per-sample difficulty."""
    arr = np.asarray(per_cluster_rmse, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    global_mean = float(np.nanmean(finite)) if len(finite) else 1.0
    filled = np.where(np.isfinite(arr), arr, global_mean)
    safe_mean = max(global_mean, 1.0e-8)
    relative = filled / safe_mean
    ids = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    ids_safe = np.clip(ids, 0, len(relative) - 1)
    return np.clip(relative[ids_safe], 0.0, 10.0).astype(np.float64)
