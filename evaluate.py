from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from metrics import compute_metrics
from losses import prediction_loss
from utils.logger import save_json


_SAMPLE_OUTPUT_FILES = {'expert_weights': 'mtam_hg_expert_weights.npy',
 'gate_probs': 'mtam_hg_gate_probs.npy',
 'topk_indices': 'mtam_hg_topk_indices.npy',
 'expert_preds': 'mtam_hg_expert_preds.npy',
 'sample_confidence': 'agent_sample_confidence.npy',
 'synthetic_keep_score': 'agent_synthetic_keep_score.npy',
 'training_weight': 'agent_training_weight.npy',
 'expert_reliability': 'agent_expert_reliability.npy',
 'uncertainty_reason_vector': 'agent_uncertainty_reason_vector.npy',
 'expert_uncertainty': 'agent_expert_uncertainty.npy',
 'agent_gate_entropy': 'agent_gate_entropy.npy'}
_GRAPH_OUTPUT_FILES = {'A_kg': 'learned_A_kg.npy',
 'A_kg_experts': 'learned_A_kg_experts.npy',
 'A_het': 'learned_A_het.npy'}


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
def collect_predictions(model, loader, device: torch.device, data_bundle, *, batch_observer=None) -> dict[str, np.ndarray]:
    model.eval()
    ys, mus, bs = [], [], []
    gates_by_stage: list[list[np.ndarray]] | None = None
    sample_chunks = {key: [] for key in _SAMPLE_OUTPUT_FILES}
    last_graphs = {key: None for key in _GRAPH_OUTPUT_FILES}

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        y = y.to(device)
        raw_outputs = model(x)
        if batch_observer is not None:
            batch_observer(raw_outputs, y)
        outputs = _as_output_dict(raw_outputs)
        ys.append(y.cpu().numpy())
        mus.append(outputs["mu"].cpu().numpy())
        if "b" in outputs:
            bs.append(outputs["b"].cpu().numpy())
        sample_arrays = {
            key: outputs[key].detach().cpu().numpy()
            for key in sample_chunks if key in outputs
        }
        if outputs.get("gate_weights"):
            if gates_by_stage is None:
                gates_by_stage = [[] for _ in outputs["gate_weights"]]
            arrays_by_id = {id(outputs[key]): array for key, array in sample_arrays.items()}
            for stage_idx, weights in enumerate(outputs["gate_weights"]):
                array = arrays_by_id.get(id(weights))
                if array is None or weights.requires_grad:
                    array = weights.cpu().numpy()
                gates_by_stage[stage_idx].append(array)
        elif "expert_weights" in sample_arrays:
            if gates_by_stage is None:
                gates_by_stage = [[]]
            gates_by_stage[0].append(sample_arrays["expert_weights"])
        for key, array in sample_arrays.items():
            sample_chunks[key].append(array)
        for key in last_graphs:
            if key in outputs:
                last_graphs[key] = outputs[key].detach().cpu().numpy()

    y = np.concatenate(ys, axis=0)
    mu = np.concatenate(mus, axis=0)
    b = np.concatenate(bs, axis=0) if bs else None
    result = {
        "y_scaled": y,
        "mu_scaled": mu,
        "y": _inverse_y(y, data_bundle),
        "mu": _inverse_y(mu, data_bundle),
    }
    result.update({key: value for key, value in last_graphs.items() if value is not None})
    if b is not None:
        result["b_scaled"] = b
        result["b"] = _inverse_b(b, data_bundle)
    if gates_by_stage is not None:
        result["gate_weights"] = np.stack([np.concatenate(stage, axis=0) for stage in gates_by_stage], axis=0)
    result.update({
        key: np.concatenate(chunks, axis=0)
        for key, chunks in sample_chunks.items() if chunks
    })
    return result


def evaluate_model(model, loader, device: torch.device, data_bundle, *, batch_observer=None) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    collected = collect_predictions(model, loader, device, data_bundle, batch_observer=batch_observer)
    b = collected.get("b")
    metrics = compute_metrics(
        collected["y"],
        collected["mu"],
        b=b,
        tail_thresholds=data_bundle.tail_thresholds,
    )
    return metrics, collected


def evaluate_model_and_loss(model, loader, device: torch.device, data_bundle):

    loss_sum = 0.0
    count = 0

    def accumulate_loss(outputs, y):
        nonlocal loss_sum, count
        loss = prediction_loss(outputs, y, weights=None)
        loss_sum += float(loss.detach().cpu()) * y.shape[0]
        count += int(y.shape[0])

    metrics, collected = evaluate_model(
        model, loader, device, data_bundle, batch_observer=accumulate_loss,
    )
    return metrics, collected, loss_sum / max(count, 1)


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

    for key, filename in _GRAPH_OUTPUT_FILES.items():
        if key in collected:
            np.save(output_dir / filename, collected[key])
        if key == "A_kg_experts" and key in collected:
            edge_importance = np.abs(collected["A_kg_experts"]).mean(axis=0)
            np.fill_diagonal(edge_importance, 0.0)
            node_scores = edge_importance.sum(axis=0) + edge_importance.sum(axis=1)
            node_importance = node_scores / max(float(node_scores.max()), 1.0e-12)
            np.save(output_dir / "edge_importance.npy", edge_importance)
            np.save(output_dir / "node_importance.npy", node_importance)
    for key, filename in {"gate_weights": "gate_weights.npy", **_SAMPLE_OUTPUT_FILES}.items():
        if key in collected:
            np.save(output_dir / filename, collected[key])
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
