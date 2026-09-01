"""Loss functions for robust yield strength prediction."""

from __future__ import annotations

from typing import Iterable

import torch

import config


def weighted_mse_loss(mu: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted MSE for deterministic regression."""
    err = (mu - y) ** 2
    if weights is not None:
        err = err * weights
    return err.mean()


def laplace_nll_loss(
    mu: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    """Compute Laplace negative log likelihood."""
    eps = config.LAPLACE_EPS if eps is None else eps
    scale = b + eps
    loss = torch.abs(y - mu) / scale + torch.log(scale)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def moe_load_balance_loss(gate_weights_list: Iterable[torch.Tensor]) -> torch.Tensor:
    """Reference MoE auxiliary loss: importance balance + load balance."""
    weights_list = list(gate_weights_list)
    device = weights_list[0].device if weights_list else None
    losses = []
    for weights in weights_list:
        num_experts = weights.shape[-1]
        uniform = weights.new_full((num_experts,), 1.0 / num_experts)
        importance = weights.sum(dim=0)
        importance = importance / (importance.sum() + 1.0e-8)
        importance_loss = ((importance - uniform) ** 2).mean()
        load = weights.mean(dim=0)
        load_loss = ((load - uniform) ** 2).mean()
        losses.append(importance_loss + load_loss)
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).sum()


def graph_regularization_loss(A_kg: torch.Tensor, A0: torch.Tensor) -> torch.Tensor:
    """Paper L_graph = ||A_kg - A0||_F^2."""
    return torch.sum((A_kg - A0) ** 2)


def prediction_loss(
    outputs: dict[str, torch.Tensor] | torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return weighted_mse_loss(outputs, y, weights=weights)
    use_laplace = config.USE_LAPLACE and "b" in outputs
    if use_laplace:
        return laplace_nll_loss(outputs["mu"], outputs["b"], y, weights=weights)
    return weighted_mse_loss(outputs["mu"], y, weights=weights)


def expert_calibration_loss(
    expert_preds: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weakly supervise each expert so low-routed experts cannot drift."""
    target = y.unsqueeze(1).expand_as(expert_preds)
    err = (expert_preds - target) ** 2
    if weights is not None:
        err = err * weights.unsqueeze(1)
    per_expert = err.reshape(err.shape[0], err.shape[1], -1).mean(dim=(0, 2))
    quality_idx = int(getattr(config, "EXPERT_CALIBRATION_QUALITY_INDEX", 3))
    if 0 <= quality_idx < per_expert.numel():
        quality_lambda = float(getattr(config, "EXPERT_CALIBRATION_QUALITY_LAMBDA", 0.0))
        return per_expert.mean() + quality_lambda * per_expert[quality_idx]
    return per_expert.mean()


def _standardize_batch(values: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    flat = values.reshape(-1)
    return (values - flat.mean()) / (flat.std(unbiased=False) + eps)


def _batch_tail_indicator(y_true: torch.Tensor) -> torch.Tensor:
    """Batch-quantile tail marker; can be replaced by train-set thresholds later."""
    y_flat = y_true.detach().reshape(-1)
    low_q = float(getattr(config, "TAIL_QUANTILE_LOW", getattr(config, "TAIL_QUANTILE", 0.10)))
    high_q = float(getattr(config, "TAIL_QUANTILE_HIGH", 1.0 - getattr(config, "TAIL_QUANTILE", 0.10)))
    low = torch.quantile(y_flat, low_q)
    high = torch.quantile(y_flat, high_q)
    return ((y_flat <= low) | (y_flat >= high)).to(dtype=y_true.dtype, device=y_true.device)


def compute_agent_reward(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    expert_preds: torch.Tensor,
    gate_probs: torch.Tensor,
    tail_indicator: torch.Tensor | None = None,
    cluster_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per-sample Agent rewards."""
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    per_sample_error = ((y_pred_flat - y_true_flat) ** 2).mean(dim=-1)
    uncertainty = expert_preds.detach().var(dim=1, unbiased=False).reshape(expert_preds.shape[0], -1).mean(dim=-1)
    gate_entropy = -(gate_probs * torch.log(gate_probs + 1.0e-8)).sum(dim=-1)
    normalized_entropy = gate_entropy / torch.log(gate_probs.new_tensor(float(gate_probs.shape[-1])))
    if tail_indicator is None:
        tail_indicator = _batch_tail_indicator(y_true)
    else:
        tail_indicator = tail_indicator.detach().reshape(-1).to(dtype=y_true.dtype, device=y_true.device)

    norm_error = _standardize_batch(per_sample_error)
    norm_uncertainty = _standardize_batch(uncertainty)
    prediction_error_reward = -float(getattr(config, "REWARD_ALPHA_ERROR", 1.0)) * norm_error
    uncertainty_reward = -float(getattr(config, "REWARD_ALPHA_UNCERTAINTY", 0.5)) * norm_uncertainty
    entropy_reward = float(getattr(config, "REWARD_ALPHA_ENTROPY", 0.1)) * normalized_entropy
    tail_bonus = float(getattr(config, "REWARD_ALPHA_TAIL", 0.2)) * tail_indicator

    cluster_balance_penalty = y_pred.new_tensor(0.0)
    var_cluster_val = y_pred.new_tensor(0.0)
    cluster_details: dict[str, torch.Tensor] = {}
    if (
        bool(getattr(config, "USE_CLUSTER_BALANCE_REWARD", False))
        and cluster_labels is not None
    ):
        from training.clusters import (
            compute_cluster_balance_stats,
        )
        num_clusters = int(getattr(config, "NUM_WORKING_CONDITION_CLUSTERS", 5))
        alpha_cluster = float(getattr(config, "REWARD_ALPHA_CLUSTER", 0.3))
        labels_dev = cluster_labels.reshape(-1).to(device=y_pred.device)
        _, var_cluster, cb_details = compute_cluster_balance_stats(
            y_pred_flat.mean(dim=-1),
            y_true_flat.mean(dim=-1),
            labels_dev,
            num_clusters,
        )
        var_cluster_val = var_cluster
        var_mae = cb_details.get("var_mae_tensor", y_pred.new_tensor(0.0))
        var_mape = cb_details.get("var_mape_tensor", y_pred.new_tensor(0.0))
        var_r2 = cb_details.get("var_r2_tensor", y_pred.new_tensor(0.0))
        w_mae = float(getattr(config, "CLUSTER_BALANCE_MAE_WEIGHT", 0.3))
        w_mape = float(getattr(config, "CLUSTER_BALANCE_MAPE_WEIGHT", 0.1))
        w_r2 = float(getattr(config, "CLUSTER_BALANCE_R2_WEIGHT", 0.3))
        var_mape_normed = var_mape / (100.0 ** 2 + 1.0e-8)
        var_composite = (
            var_cluster + w_mae * var_mae + w_mape * var_mape_normed + w_r2 * var_r2
        )
        cluster_balance_penalty = (-alpha_cluster * var_composite).clamp(
            min=float(getattr(config, "CLUSTER_BALANCE_PENALTY_MIN", -1.0)),
            max=float(getattr(config, "CLUSTER_BALANCE_PENALTY_MAX", 0.0)),
        ).unsqueeze(0).expand(y_pred.shape[0])
        cluster_details = {
            "var_mae": var_mae.detach(),
            "var_mape": var_mape.detach(),
            "var_r2": var_r2.detach(),
        }

    reward = prediction_error_reward + uncertainty_reward + entropy_reward + tail_bonus + cluster_balance_penalty
    reward = reward.detach()
    reward = torch.clamp(
        reward,
        min=float(getattr(config, "REWARD_CLAMP_MIN", -3.0)),
        max=float(getattr(config, "REWARD_CLAMP_MAX", 3.0)),
    )

    components = {
        "reward": reward,
        "per_sample_error": per_sample_error.detach(),
        "expert_uncertainty": uncertainty.detach(),
        "gate_entropy": gate_entropy.detach(),
        "prediction_error_reward": prediction_error_reward.detach(),
        "uncertainty_reward": uncertainty_reward.detach(),
        "entropy_reward": entropy_reward.detach(),
        "tail_bonus": tail_bonus.detach(),
        "tail_indicator": tail_indicator.detach(),
        "cluster_balance_penalty": cluster_balance_penalty.detach() if torch.is_tensor(cluster_balance_penalty) else cluster_balance_penalty,
        "var_cluster": var_cluster_val.detach(),
        **cluster_details,
    }
    return reward, components


def compute_agent_reward_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    expert_preds: torch.Tensor,
    gate_probs: torch.Tensor,
    sample_confidence: torch.Tensor,
    tail_indicator: torch.Tensor | None = None,
    cluster_labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reward objective for Agent sample confidence."""
    reward, components = compute_agent_reward(
        y_pred,
        y_true,
        expert_preds,
        gate_probs,
        tail_indicator=tail_indicator,
        cluster_labels=cluster_labels,
    )
    confidence = sample_confidence.reshape(-1).clamp(1.0e-6, 1.0 - 1.0e-6)
    agent_reward_loss = -(confidence * reward).mean()
    conf_mean = confidence.mean()
    confidence_mean_reg = (conf_mean - float(getattr(config, "TARGET_CONFIDENCE_MEAN", 0.6))) ** 2
    confidence_entropy = -(
        confidence * torch.log(confidence + 1.0e-8)
        + (1.0 - confidence) * torch.log(1.0 - confidence + 1.0e-8)
    ).mean()
    agent_total_loss = (
        agent_reward_loss
        + float(getattr(config, "CONFIDENCE_MEAN_REG_LAMBDA", 0.01)) * confidence_mean_reg
        - float(getattr(config, "CONFIDENCE_ENTROPY_REG_LAMBDA", 0.001)) * confidence_entropy
    )
    info = {
        "agent_reward_loss": float(agent_reward_loss.detach().cpu()),
        "agent_total_loss": float(agent_total_loss.detach().cpu()),
        "reward_mean": float(reward.mean().detach().cpu()),
        "reward_std": float(reward.std(unbiased=False).detach().cpu()),
        "reward_min": float(reward.min().detach().cpu()),
        "reward_max": float(reward.max().detach().cpu()),
        "sample_confidence_min": float(confidence.detach().min().cpu()),
        "sample_confidence_max": float(confidence.detach().max().cpu()),
        "prediction_error_reward_mean": float(components["prediction_error_reward"].mean().detach().cpu()),
        "uncertainty_reward_mean": float(components["uncertainty_reward"].mean().detach().cpu()),
        "entropy_reward_mean": float(components["entropy_reward"].mean().detach().cpu()),
        "tail_bonus_mean": float(components["tail_bonus"].mean().detach().cpu()),
        "tail_sample_ratio": float(components["tail_indicator"].mean().detach().cpu()),
        "confidence_mean_reg": float(confidence_mean_reg.detach().cpu()),
        "confidence_entropy": float(confidence_entropy.detach().cpu()),
        "cluster_balance_penalty_mean": float(
            components["cluster_balance_penalty"].mean().detach().cpu()
            if torch.is_tensor(components["cluster_balance_penalty"])
            else 0.0
        ),
        "var_cluster": float(components["var_cluster"].detach().cpu()),
    }
    return agent_total_loss, info


def _as_output_dict(outputs: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
    if torch.is_tensor(outputs):
        return {"mu": outputs}
    return outputs


def _append_expert_logs(outputs: dict[str, torch.Tensor], logs: dict[str, float]) -> None:
    weights = outputs.get("expert_weights")
    if weights is None:
        gate_weights = outputs.get("gate_weights", [])
        if gate_weights:
            weights = gate_weights[0]
    if weights is None:
        return
    weights_detached = weights.detach()
    mean_weight = weights_detached.mean(dim=0).cpu()
    selected_rate = (weights_detached > 0).float().mean(dim=0).cpu()
    for idx in range(weights_detached.shape[-1]):
        logs[f"expert_weight_{idx}_mean"] = float(mean_weight[idx])
        logs[f"expert_usage_{idx}"] = float(mean_weight[idx])
        logs[f"expert_selected_rate_{idx}"] = float(selected_rate[idx])
    sample_confidence = outputs.get("sample_confidence")
    if sample_confidence is not None:
        confidence = sample_confidence.detach().reshape(-1).cpu()
        logs["sample_confidence_mean"] = float(confidence.mean())
        logs["sample_confidence_std"] = float(confidence.std(unbiased=False))
    synthetic_keep_score = outputs.get("synthetic_keep_score")
    if synthetic_keep_score is not None:
        keep_score = synthetic_keep_score.detach().reshape(-1).cpu()
        logs["synthetic_keep_score_mean"] = float(keep_score.mean())
        logs["synthetic_keep_score_std"] = float(keep_score.std(unbiased=False))
    training_weight = outputs.get("training_weight")
    if training_weight is not None:
        weight = training_weight.detach().reshape(-1).cpu()
        logs["agent_training_weight_mean"] = float(weight.mean())
        logs["agent_training_weight_std"] = float(weight.std(unbiased=False))
    expert_reliability = outputs.get("expert_reliability")
    if expert_reliability is not None:
        reliability = expert_reliability.detach().cpu()
        logs["expert_reliability_mean"] = float(reliability.mean())
        for idx in range(reliability.shape[-1]):
            logs[f"expert_reliability_{idx}_mean"] = float(reliability[:, idx].mean())
    uncertainty_reason_vector = outputs.get("uncertainty_reason_vector")
    if uncertainty_reason_vector is not None:
        reason = uncertainty_reason_vector.detach().cpu()
        for idx in range(reason.shape[-1]):
            logs[f"uncertainty_reason_{idx}_mean"] = float(reason[:, idx].mean())
    expert_uncertainty = outputs.get("expert_uncertainty")
    if expert_uncertainty is None and "expert_preds" in outputs:
        expert_uncertainty = outputs["expert_preds"].detach().var(dim=1, unbiased=False)
    if expert_uncertainty is not None:
        logs["expert_uncertainty_mean"] = float(expert_uncertainty.detach().mean().cpu())
    gate_probs = outputs.get("gate_probs")
    if gate_probs is not None:
        entropy = -(gate_probs.detach() * torch.log(gate_probs.detach() + 1.0e-8)).sum(dim=-1).mean()
        logs["agent_gate_entropy"] = float(entropy.cpu())


def total_loss(
    outputs: dict[str, torch.Tensor] | torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    edge_mask: torch.Tensor | None = None,
    batch_weights: torch.Tensor | None = None,
    cluster_labels: torch.Tensor | None = None,
    use_internal_agent_weight: bool | None = None,
    use_agent_reward: bool | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the supervised training objective."""
    outputs = _as_output_dict(outputs)
    configured_internal_agent_weight = (
        bool(getattr(config, "AGENT_USE_SAMPLE_WEIGHT_FOR_SUPERVISED_LOSS", False))
        or bool(getattr(config, "USE_CONFIDENCE_WEIGHTED_SUPERVISED_LOSS", False))
    )
    apply_internal_agent_weight = (
        configured_internal_agent_weight
        if use_internal_agent_weight is None
        else bool(use_internal_agent_weight)
    )
    if (
        apply_internal_agent_weight
        and "sample_confidence" in outputs
    ):
        agent_weight = outputs.get("training_weight", outputs["sample_confidence"])
        confidence_weight = agent_weight.detach().reshape(-1, 1).to(y.device, dtype=y.dtype)
        batch_weights = confidence_weight if batch_weights is None else batch_weights * confidence_weight
    pred = prediction_loss(outputs, y, weights=batch_weights)
    if "aux_loss" in outputs:
        moe = outputs["aux_loss"].to(y.device)
    else:
        moe = moe_load_balance_loss(outputs.get("gate_weights", [])).to(y.device)
    lambda_moe = float(getattr(config, "MOE_AUX_LAMBDA", getattr(config, "LAMBDA_MOE", 0.01)))
    total = pred + lambda_moe * moe
    expert_calib = torch.tensor(0.0, device=y.device)
    lambda_expert_calib = float(getattr(config, "EXPERT_CALIBRATION_LAMBDA", 0.0))
    if lambda_expert_calib > 0 and "expert_preds" in outputs:
        expert_calib = expert_calibration_loss(outputs["expert_preds"], y, weights=None).to(y.device)
        total = total + lambda_expert_calib * expert_calib
    diversity_loss = outputs.get("diversity_loss", torch.tensor(0.0, device=y.device))
    diversity_loss = diversity_loss.to(y.device) if torch.is_tensor(diversity_loss) else torch.tensor(0.0, device=y.device)
    lambda_diversity = float(getattr(config, "EXPERT_DIVERSITY_LAMBDA", 0.0))
    if lambda_diversity > 0:
        total = total + lambda_diversity * diversity_loss
    agent_reward_loss = torch.tensor(0.0, device=y.device)
    agent_logs: dict[str, float] = {}
    use_batch_agent_reward = (
        bool(getattr(config, "USE_AGENT_REWARD", False))
        if use_agent_reward is None
        else bool(use_agent_reward)
    )
    if (
        use_batch_agent_reward
        and "sample_confidence" in outputs
        and "expert_preds" in outputs
        and "gate_probs" in outputs
    ):
        agent_reward_loss, agent_logs = compute_agent_reward_loss(
            y_pred=outputs["mu"],
            y_true=y,
            expert_preds=outputs["expert_preds"],
            gate_probs=outputs["gate_probs"],
            sample_confidence=outputs["sample_confidence"],
            cluster_labels=cluster_labels,
        )
        total = total + float(getattr(config, "AGENT_REWARD_LAMBDA", 0.01)) * agent_reward_loss
    confidence_reg = torch.tensor(0.0, device=y.device)
    confidence_reg_lambda = float(getattr(config, "AGENT_CONFIDENCE_REG_LAMBDA", 0.0))
    if confidence_reg_lambda > 0 and "sample_confidence" in outputs:
        confidence_reg = (1.0 - outputs["sample_confidence"].reshape(-1)).mean()
        total = total + confidence_reg_lambda * confidence_reg

    graph_loss = torch.tensor(0.0, device=y.device)
    if "A0" in outputs:
        if "A_kg_experts" in outputs:
            A0 = outputs.get("A0_experts")
            if A0 is None:
                A0 = outputs["A0"].unsqueeze(0).expand_as(outputs["A_kg_experts"])
            graph_loss = graph_regularization_loss(outputs["A_kg_experts"], A0)
        elif "A_kg" in outputs:
            graph_loss = graph_regularization_loss(outputs["A_kg"], outputs["A0"])
        total = total + config.LAMBDA_GRAPH * graph_loss

    logs = {
        "pred_loss": float(pred.detach().cpu()),
        "moe_loss": float(moe.detach().cpu()),
        "moe_aux_loss": float(moe.detach().cpu()),
        "expert_calibration_loss": float(expert_calib.detach().cpu()),
        "expert_diversity_loss": float(diversity_loss.detach().cpu()),
        "mask_loss": 0.0,
        "edge_loss": 0.0,
        "graph_loss": float(graph_loss.detach().cpu()),
        "confidence_reg_loss": float(confidence_reg.detach().cpu()),
        "agent_reward_loss": agent_logs.get("agent_reward_loss", float(agent_reward_loss.detach().cpu())),
        "agent_total_loss": agent_logs.get("agent_total_loss", float(agent_reward_loss.detach().cpu())),
        "total_loss": float(total.detach().cpu()),
        "batch_weight_mean": float(batch_weights.detach().mean().cpu()) if batch_weights is not None else 1.0,
    }
    logs.update(agent_logs)
    _append_expert_logs(outputs, logs)
    return total, logs
