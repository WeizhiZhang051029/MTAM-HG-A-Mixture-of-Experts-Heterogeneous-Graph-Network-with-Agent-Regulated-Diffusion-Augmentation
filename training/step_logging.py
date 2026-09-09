from __future__ import annotations
import torch
from utils.tensor_logging import scalar_logs


@torch.no_grad()
def agent_step_logs(agent_components, agent_loss, reward, reward_parts, feedback_reward, mse_real, *, defer_logs=False):
    values = {
        "agent_reward_loss": agent_components["expected_reward_loss"].detach(),
        "agent_total_loss": agent_loss.detach(),
        "agent_confidence_mean_regularizer": agent_components["mean_regularizer"].detach(),
        "agent_confidence_entropy": agent_components["confidence_entropy"].detach(),
        "reward_mean": reward.detach().mean(),
        "reward_std": reward.detach().std(unbiased=False),
        "reward_min": reward.detach().min(),
        "reward_max": reward.detach().max(),
        "reward_mse_mean": reward_parts["reward_mse"].detach().mean(),
        "reward_process_mean": reward_parts["reward_process"].detach().mean(),
        "reward_mechanism_mean": reward_parts["reward_mechanism"].detach().mean(),
        "reward_feedback_mean": feedback_reward.detach().mean() if feedback_reward is not None else float("nan"),
        "mse_real_mean": mse_real.detach().mean(),
        "mse_real_std": mse_real.detach().std(unbiased=False),
    }
    return values if defer_logs else scalar_logs(values)


@torch.no_grad()
def synthetic_step_logs(
    *,
    outputs,
    base_loss_logs,
    agent_logs,
    agent_loss,
    total,
    quality_score,
    process_consistency,
    mechanism_consistency,
    selected,
    is_tail,
    dynamic_batch_weight,
    update_quality_agent,
    device,
):
    quality_mean = quality_score.detach().mean()
    quality_std = quality_score.detach().std(unbiased=False)
    gate_probs = outputs.get("gate_probs")
    entropy = (
        -(gate_probs.detach() * torch.log(gate_probs.detach() + 1e-08)).sum(dim=-1).mean()
        if gate_probs is not None
        else torch.tensor(float("nan"), device=device)
    )
    expert_uncertainty = outputs.get("expert_uncertainty")
    if expert_uncertainty is None and "expert_preds" in outputs:
        expert_uncertainty = outputs["expert_preds"].detach().var(dim=1, unbiased=False)
    reward_mean = agent_logs.get("reward_mean", float("nan"))
    reward_std = agent_logs.get("reward_std", float("nan"))
    values = {
        "synthetic_pred_loss": base_loss_logs.get("pred_loss", float("nan")),
        "synthetic_moe_aux_loss": base_loss_logs.get("moe_loss", float("nan")),
        "synthetic_expert_calibration_loss": base_loss_logs.get("expert_calibration_loss", 0.0),
        "synthetic_expert_diversity_loss": base_loss_logs.get("expert_diversity_loss", 0.0),
        "synthetic_graph_loss": base_loss_logs.get("graph_loss", 0.0),
        "synthetic_agent_reward_loss": agent_logs.get("agent_reward_loss", agent_loss.detach()),
        "synthetic_agent_total_loss": agent_logs.get("agent_total_loss", agent_loss.detach()),
        "synthetic_total_loss": total.detach(),
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
        "synthetic_confidence_mean": quality_mean,
        "synthetic_confidence_std": quality_std,
        "synthetic_keep_score_mean": quality_mean,
        "synthetic_keep_score_std": quality_std,
        "synthetic_quality_score_mean": quality_mean,
        "synthetic_quality_score_std": quality_std,
        "synthetic_process_consistency_mean": process_consistency.detach().mean(),
        "synthetic_process_consistency_std": process_consistency.detach().std(unbiased=False),
        "synthetic_mechanism_consistency_mean": mechanism_consistency.detach().mean(),
        "synthetic_mechanism_consistency_std": mechanism_consistency.detach().std(unbiased=False),
        "synthetic_final_weight_mean": quality_mean,
        "synthetic_final_weight_std": quality_std,
        "synthetic_selected_ratio": selected.detach().float().mean(),
        "synthetic_tail_ratio": is_tail.detach().float().mean(),
        "dynamic_synthetic_weight_mean": dynamic_batch_weight.detach().mean()
        if dynamic_batch_weight is not None
        else float("nan"),
        "dynamic_synthetic_weight_std": dynamic_batch_weight.detach().std(unbiased=False)
        if dynamic_batch_weight is not None
        else float("nan"),
        "synthetic_expert_uncertainty_mean": expert_uncertainty.detach().mean()
        if expert_uncertainty is not None
        else float("nan"),
        "synthetic_gate_entropy": entropy.detach(),
        "synthetic_agent_update_active": float(update_quality_agent),
    }
    return scalar_logs(values)
