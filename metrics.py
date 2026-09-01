"""Evaluation metrics for yield strength prediction."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0e-8) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + eps))


def tail_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    quantile: float = 0.10,
) -> float:
    """MAE on bottom and top quantile samples by true yield strength."""
    y_flat = y_true.reshape(-1)
    pred_flat = y_pred.reshape(-1)
    lower = np.quantile(y_flat, quantile) if lower_threshold is None else lower_threshold
    upper = np.quantile(y_flat, 1.0 - quantile) if upper_threshold is None else upper_threshold
    mask = (y_flat <= lower) | (y_flat >= upper)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_flat[mask] - pred_flat[mask])))


def laplace_nll_np(y_true: np.ndarray, mu: np.ndarray, b: np.ndarray, eps: float = 1.0e-8) -> float:
    scale = b + eps
    return float(np.mean(np.abs(y_true - mu) / scale + np.log(scale)))


def uncertainty_reliability(
    y_true: np.ndarray,
    mu: np.ndarray,
    b: np.ndarray,
    levels: tuple[float, ...] = (0.50, 0.80, 0.90),
) -> dict[str, float]:
    """Empirical coverage for central Laplace prediction intervals."""
    out: dict[str, float] = {}
    abs_err = np.abs(y_true - mu)
    for level in levels:
        radius = b * np.log(1.0 / max(1.0 - level, 1.0e-8))
        coverage = np.mean(abs_err <= radius)
        out[f"coverage_{int(level * 100)}"] = float(coverage)
    out["mean_scale_b"] = float(np.mean(b))
    out["error_scale_corr"] = float(np.corrcoef(abs_err.reshape(-1), b.reshape(-1))[0, 1]) if len(abs_err.reshape(-1)) > 1 else float("nan")
    return out


def compute_metrics(
    y_true: np.ndarray,
    mu: np.ndarray,
    b: np.ndarray | None = None,
    tail_thresholds: tuple[float, float] | None = None,
) -> dict[str, float]:
    lower, upper = tail_thresholds if tail_thresholds is not None else (None, None)
    metrics = {
        "RMSE": rmse(y_true, mu),
        "MAE": mae(y_true, mu),
        "MAPE": mape(y_true, mu),
        "R2": r2_score(y_true, mu),
        "TAIL_MAE": tail_mae(y_true, mu, lower, upper),
    }
    if b is not None:
        metrics["NLL"] = laplace_nll_np(y_true, mu, b)
        metrics.update(uncertainty_reliability(y_true, mu, b))
    return metrics
