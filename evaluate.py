"""Evaluation and output export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from metrics import compute_metrics
from utils.logger import save_json


def _inverse_y(y: np.ndarray, data_bundle) -> np.ndarray:
    if config.STANDARDIZE_Y:
        return data_bundle.y_scaler.inverse_transform(y)
    return y


def _inverse_b(b: np.ndarray, data_bundle) -> np.ndarray:
    if config.STANDARDIZE_Y:
        return b * data_bundle.y_scaler.std_
    return b


def _as_output_dict(outputs):
    if torch.is_tensor(outputs):
        return {"mu": outputs}
    return outputs


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def agent_selection_alignment_metrics(diag: pd.DataFrame) -> dict[str, float | int]:
    """Summarize whether Agent confidence aligns with real validation/test quality."""
    if diag.empty or "abs_error" not in diag or "sample_confidence" not in diag:
        return {}
    abs_error = diag["abs_error"].astype(float).to_numpy()
    confidence = diag["sample_confidence"].astype(float).to_numpy()
    y_true = diag["y_true"].astype(float).to_numpy() if "y_true" in diag else np.array([])
    tail_mask = np.zeros(len(diag), dtype=bool)
    if len(y_true):
        low = float(np.quantile(y_true, 0.10))
        high = float(np.quantile(y_true, 0.90))
        tail_mask = (y_true <= low) | (y_true >= high)
    high_error_cutoff = float(np.quantile(abs_error, 0.90)) if len(abs_error) else float("nan")
    high_error_mask = abs_error >= high_error_cutoff if np.isfinite(high_error_cutoff) else np.zeros(len(diag), dtype=bool)
    top_conf_cutoff = float(np.quantile(confidence, 0.80)) if len(confidence) else float("nan")
    top_conf_mask = confidence >= top_conf_cutoff if np.isfinite(top_conf_cutoff) else np.zeros(len(diag), dtype=bool)
    confidence_rank = pd.Series(confidence).rank(method="average").to_numpy()
    error_rank = pd.Series(abs_error).rank(method="average").to_numpy()
    out: dict[str, float | int] = {
        "samples": int(len(diag)),
        "confidence_abs_error_pearson": _safe_pearson(confidence, abs_error),
        "confidence_abs_error_spearman": _safe_pearson(confidence_rank, error_rank),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_std": float(np.std(confidence)),
        "abs_error_mean": float(np.mean(abs_error)),
        "abs_error_p90": high_error_cutoff,
        "top_confidence_abs_error_mean": float(np.mean(abs_error[top_conf_mask])) if bool(top_conf_mask.any()) else float("nan"),
        "tail_sample_count": int(tail_mask.sum()),
        "tail_abs_error_mean": float(np.mean(abs_error[tail_mask])) if bool(tail_mask.any()) else float("nan"),
        "tail_confidence_mean": float(np.mean(confidence[tail_mask])) if bool(tail_mask.any()) else float("nan"),
        "body_confidence_mean": float(np.mean(confidence[~tail_mask])) if bool((~tail_mask).any()) else float("nan"),
        "high_error_sample_count": int(high_error_mask.sum()),
        "high_error_confidence_mean": float(np.mean(confidence[high_error_mask])) if bool(high_error_mask.any()) else float("nan"),
        "non_high_error_confidence_mean": float(np.mean(confidence[~high_error_mask])) if bool((~high_error_mask).any()) else float("nan"),
    }
    if "expert_uncertainty" in diag:
        uncertainty = diag["expert_uncertainty"].astype(float).to_numpy()
        out["uncertainty_abs_error_pearson"] = _safe_pearson(uncertainty, abs_error)
        out["uncertainty_abs_error_spearman"] = _safe_pearson(
            pd.Series(uncertainty).rank(method="average").to_numpy(),
            error_rank,
        )
    return out


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device, data_bundle) -> dict[str, np.ndarray]:
    model.eval()
    ys, mus, bs = [], [], []
    gates_by_stage: list[list[np.ndarray]] | None = None
    last_A_kg = None
    last_A_kg_experts = None
    last_A_het = None
    expert_weights_all = []
    gate_probs_all = []
    topk_indices_all = []
    expert_preds_all = []
    sample_confidence_all = []
    synthetic_keep_score_all = []
    training_weight_all = []
    expert_reliability_all = []
    uncertainty_reason_all = []
    expert_uncertainty_all = []
    agent_gate_entropy_all = []

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)
        outputs = _as_output_dict(model(x))
        ys.append(y.cpu().numpy())
        mus.append(outputs["mu"].cpu().numpy())
        if "b" in outputs:
            bs.append(outputs["b"].cpu().numpy())
        if outputs.get("gate_weights"):
            if gates_by_stage is None:
                gates_by_stage = [[] for _ in outputs["gate_weights"]]
            for stage_idx, weights in enumerate(outputs["gate_weights"]):
                gates_by_stage[stage_idx].append(weights.cpu().numpy())
        elif "expert_weights" in outputs:
            if gates_by_stage is None:
                gates_by_stage = [[]]
            gates_by_stage[0].append(outputs["expert_weights"].detach().cpu().numpy())
        if "expert_weights" in outputs:
            expert_weights_all.append(outputs["expert_weights"].detach().cpu().numpy())
        if "gate_probs" in outputs:
            gate_probs_all.append(outputs["gate_probs"].detach().cpu().numpy())
        if "topk_indices" in outputs:
            topk_indices_all.append(outputs["topk_indices"].detach().cpu().numpy())
        if "expert_preds" in outputs:
            expert_preds_all.append(outputs["expert_preds"].detach().cpu().numpy())
        if "sample_confidence" in outputs:
            sample_confidence_all.append(outputs["sample_confidence"].detach().cpu().numpy())
        if "synthetic_keep_score" in outputs:
            synthetic_keep_score_all.append(outputs["synthetic_keep_score"].detach().cpu().numpy())
        if "training_weight" in outputs:
            training_weight_all.append(outputs["training_weight"].detach().cpu().numpy())
        if "expert_reliability" in outputs:
            expert_reliability_all.append(outputs["expert_reliability"].detach().cpu().numpy())
        if "uncertainty_reason_vector" in outputs:
            uncertainty_reason_all.append(outputs["uncertainty_reason_vector"].detach().cpu().numpy())
        if "expert_uncertainty" in outputs:
            expert_uncertainty_all.append(outputs["expert_uncertainty"].detach().cpu().numpy())
        if "agent_gate_entropy" in outputs:
            agent_gate_entropy_all.append(outputs["agent_gate_entropy"].detach().cpu().numpy())
        if "A_kg" in outputs:
            last_A_kg = outputs["A_kg"].detach().cpu().numpy()
        if "A_kg_experts" in outputs:
            last_A_kg_experts = outputs["A_kg_experts"].detach().cpu().numpy()
        if "A_het" in outputs:
            last_A_het = outputs["A_het"].detach().cpu().numpy()

    y = np.concatenate(ys, axis=0)
    mu = np.concatenate(mus, axis=0)
    b = np.concatenate(bs, axis=0) if bs else None
    result = {
        "y_scaled": y,
        "mu_scaled": mu,
        "y": _inverse_y(y, data_bundle),
        "mu": _inverse_y(mu, data_bundle),
    }
    if last_A_kg is not None:
        result["A_kg"] = last_A_kg
    if last_A_kg_experts is not None:
        result["A_kg_experts"] = last_A_kg_experts
    if last_A_het is not None:
        result["A_het"] = last_A_het
    if b is not None:
        result["b_scaled"] = b
        result["b"] = _inverse_b(b, data_bundle)
    if gates_by_stage is not None:
        result["gate_weights"] = np.stack([np.concatenate(stage, axis=0) for stage in gates_by_stage], axis=0)
    if expert_weights_all:
        result["expert_weights"] = np.concatenate(expert_weights_all, axis=0)
    if gate_probs_all:
        result["gate_probs"] = np.concatenate(gate_probs_all, axis=0)
    if topk_indices_all:
        result["topk_indices"] = np.concatenate(topk_indices_all, axis=0)
    if expert_preds_all:
        result["expert_preds"] = np.concatenate(expert_preds_all, axis=0)
    if sample_confidence_all:
        result["sample_confidence"] = np.concatenate(sample_confidence_all, axis=0)
    if synthetic_keep_score_all:
        result["synthetic_keep_score"] = np.concatenate(synthetic_keep_score_all, axis=0)
    if training_weight_all:
        result["training_weight"] = np.concatenate(training_weight_all, axis=0)
    if expert_reliability_all:
        result["expert_reliability"] = np.concatenate(expert_reliability_all, axis=0)
    if uncertainty_reason_all:
        result["uncertainty_reason_vector"] = np.concatenate(uncertainty_reason_all, axis=0)
    if expert_uncertainty_all:
        result["expert_uncertainty"] = np.concatenate(expert_uncertainty_all, axis=0)
    if agent_gate_entropy_all:
        result["agent_gate_entropy"] = np.concatenate(agent_gate_entropy_all, axis=0)
    return result


def evaluate_model(model, loader, device: torch.device, data_bundle) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    collected = collect_predictions(model, loader, device, data_bundle)
    b = collected.get("b")
    metrics = compute_metrics(
        collected["y"],
        collected["mu"],
        b=b,
        tail_thresholds=data_bundle.tail_thresholds,
    )
    return metrics, collected


def save_evaluation_outputs(
    metrics: dict[str, float],
    collected: dict[str, np.ndarray],
    output_dir: Path | None = None,
) -> None:
    output_dir = output_dir or config.RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "metrics.json", metrics)
    pred_df = pd.DataFrame(
        {
            "y_true": collected["y"].reshape(-1),
            "mu": collected["mu"].reshape(-1),
        }
    )
    if "b" in collected:
        pred_df["b"] = collected["b"].reshape(-1)
    pred_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")

    if "A_kg" in collected:
        np.save(output_dir / "learned_A_kg.npy", collected["A_kg"])
    if "A_kg_experts" in collected:
        np.save(output_dir / "learned_A_kg_experts.npy", collected["A_kg_experts"])
        edge_importance = np.abs(collected["A_kg_experts"]).mean(axis=0)
        np.fill_diagonal(edge_importance, 0.0)
        node_scores = edge_importance.sum(axis=0) + edge_importance.sum(axis=1)
        node_importance = node_scores / max(float(node_scores.max()), 1.0e-12)
        np.save(output_dir / "edge_importance.npy", edge_importance)
        np.save(output_dir / "node_importance.npy", node_importance)
    if "A_het" in collected:
        np.save(output_dir / "learned_A_het.npy", collected["A_het"])
    if "gate_weights" in collected:
        np.save(output_dir / "gate_weights.npy", collected["gate_weights"])
    if "expert_weights" in collected:
        np.save(output_dir / "mtam_hg_expert_weights.npy", collected["expert_weights"])
    if "gate_probs" in collected:
        np.save(output_dir / "mtam_hg_gate_probs.npy", collected["gate_probs"])
    if "topk_indices" in collected:
        np.save(output_dir / "mtam_hg_topk_indices.npy", collected["topk_indices"])
    if "expert_preds" in collected:
        np.save(output_dir / "mtam_hg_expert_preds.npy", collected["expert_preds"])
    if "sample_confidence" in collected:
        np.save(output_dir / "agent_sample_confidence.npy", collected["sample_confidence"])
    if "synthetic_keep_score" in collected:
        np.save(output_dir / "agent_synthetic_keep_score.npy", collected["synthetic_keep_score"])
    if "training_weight" in collected:
        np.save(output_dir / "agent_training_weight.npy", collected["training_weight"])
    if "expert_reliability" in collected:
        np.save(output_dir / "agent_expert_reliability.npy", collected["expert_reliability"])
    if "uncertainty_reason_vector" in collected:
        np.save(output_dir / "agent_uncertainty_reason_vector.npy", collected["uncertainty_reason_vector"])
    if "expert_uncertainty" in collected:
        np.save(output_dir / "agent_expert_uncertainty.npy", collected["expert_uncertainty"])
    if "agent_gate_entropy" in collected:
        np.save(output_dir / "agent_gate_entropy.npy", collected["agent_gate_entropy"])
    if all(key in collected for key in ("sample_confidence", "expert_uncertainty", "agent_gate_entropy", "topk_indices", "expert_weights", "expert_preds")):
        diag = pd.DataFrame(
            {
                "y_true": collected["y"].reshape(-1),
                "y_pred": collected["mu"].reshape(-1),
                "abs_error": np.abs(collected["y"].reshape(-1) - collected["mu"].reshape(-1)),
                "sample_confidence": collected["sample_confidence"].reshape(-1),
                "synthetic_keep_score": collected.get("synthetic_keep_score", collected["sample_confidence"]).reshape(-1),
                "training_weight": collected.get("training_weight", collected["sample_confidence"]).reshape(-1),
                "expert_uncertainty": collected["expert_uncertainty"].reshape(-1),
                "gate_entropy": collected["agent_gate_entropy"].reshape(-1),
            }
        )
        for idx in range(collected["topk_indices"].shape[1]):
            diag[f"top{idx + 1}_index"] = collected["topk_indices"][:, idx].reshape(-1)
        for idx in range(collected["expert_weights"].shape[1]):
            diag[f"expert_weight_{idx}"] = collected["expert_weights"][:, idx]
            diag[f"expert_pred_{idx}"] = collected["expert_preds"][:, idx, ...].reshape(collected["expert_preds"].shape[0], -1).mean(axis=1)
            if "expert_reliability" in collected:
                diag[f"expert_reliability_{idx}"] = collected["expert_reliability"][:, idx]
        if "uncertainty_reason_vector" in collected:
            for idx in range(collected["uncertainty_reason_vector"].shape[1]):
                diag[f"uncertainty_reason_{idx}"] = collected["uncertainty_reason_vector"][:, idx]
        diag.to_csv(output_dir / "agent_diagnostics.csv", index=False, encoding="utf-8")
        save_json(output_dir / "agent_selection_alignment.json", agent_selection_alignment_metrics(diag))
    if "b" in collected:
        pd.DataFrame(
            {
                "mu": collected["mu"].reshape(-1),
                "b": collected["b"].reshape(-1),
                "abs_error": np.abs(collected["y"].reshape(-1) - collected["mu"].reshape(-1)),
            }
        ).to_csv(output_dir / "uncertainty.csv", index=False, encoding="utf-8")
