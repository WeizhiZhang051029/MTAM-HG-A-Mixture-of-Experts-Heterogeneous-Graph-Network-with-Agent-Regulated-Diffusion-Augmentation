"""Training loops for pretraining and supervised fine-tuning."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm

import config
from evaluate import evaluate_model, save_evaluation_outputs
from losses import (
    prediction_loss,
    total_loss,
)
from models.mr_lora import inject_mr_lora, mr_lora_parameter_names, mr_lora_scope_families
from models.mtam_hg import MTAMHG
from training.clusters import WorkingConditionCluster
from utils.logger import append_csv, ensure_dirs, save_json
from utils.seed import set_seed


def resolve_device() -> torch.device:
    if config.DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.DEVICE)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def moe_parameter_breakdown(model: torch.nn.Module) -> dict[str, float | int | str]:
    """Estimate stored and per-sample routed parameters for MoE reporting."""
    total_params = count_total_parameters(model)
    trainable_params = count_trainable_parameters(model)
    if not hasattr(model, "experts"):
        return {
            "Params": int(trainable_params),
            "Total_Params_All": int(total_params),
            "Active_Params": int(total_params),
            "Activated_Params": int(total_params),
            "Executed_Params": int(total_params),
            "Active_Params_Ratio": 1.0,
        }

    experts = list(getattr(model, "experts"))
    expert_param_counts = [sum(p.numel() for p in expert.parameters()) for expert in experts]
    expert_trainable_counts = [sum(p.numel() for p in expert.parameters() if p.requires_grad) for expert in experts]
    expert_total = int(sum(expert_param_counts))
    expert_trainable = int(sum(expert_trainable_counts))
    non_expert_total = int(total_params - expert_total)
    non_expert_trainable = int(trainable_params - expert_trainable)
    num_experts = int(getattr(model, "num_experts", len(experts)))
    top_k = int(getattr(model, "top_k", num_experts))
    routed_experts = min(max(top_k, 1), max(num_experts, 1))
    avg_expert_params = float(np.mean(expert_param_counts)) if expert_param_counts else 0.0
    avg_expert_trainable = float(np.mean(expert_trainable_counts)) if expert_trainable_counts else 0.0
    active_params = int(round(non_expert_total + routed_experts * avg_expert_params))
    active_trainable = int(round(non_expert_trainable + routed_experts * avg_expert_trainable))
    return {
        "Params": int(trainable_params),
        "Total_Params_All": int(total_params),
        "Expert_Params_Total": int(expert_total),
        "Non_Expert_Params": int(non_expert_total),
        "Active_Params": int(active_params),
        "Activated_Params": int(active_params),
        "Active_Trainable_Params": int(active_trainable),
        "Executed_Params": int(total_params),
        "Active_Params_Ratio": float(active_params / max(total_params, 1)),
        "Num_Experts": int(num_experts),
        "Top_K": int(top_k),
        "Active_Expert_Ratio": float(routed_experts / max(num_experts, 1)),
        "MoE_Routing_Strategy": "topk",
    }


def _finetune_trainable_keywords() -> tuple[str, ...]:
    keywords = getattr(
        config,
        "FINETUNE_TRAINABLE_KEYWORDS",
        ("readout", "mu_head", "log_b_head", "gate_state_proj", "router"),
    )
    return tuple(str(keyword) for keyword in keywords if str(keyword))


def configure_finetune_trainability(
    model: torch.nn.Module,
    freeze_backbone: bool | None = None,
) -> dict[str, object]:
    """Freeze pretrained backbone parameters for the real-data calibration stage."""
    has_lora_params = any(".lora_" in name for name, _param in model.named_parameters())
    if bool(getattr(config, "USE_MR_LORA", False)) and has_lora_params:
        train_output_head = bool(getattr(config, "MR_LORA_TRAIN_OUTPUT_HEAD", False))
        trainable_names: list[str] = []
        frozen_names: list[str] = []
        for name, param in model.named_parameters():
            should_train = ".lora_" in name
            if train_output_head and any(token in name for token in ("readout", "mu_head", "log_b_head")):
                should_train = True
            param.requires_grad_(should_train)
            if should_train:
                trainable_names.append(name)
            else:
                frozen_names.append(name)
        trainable = count_trainable_parameters(model)
        total = count_total_parameters(model)
        return {
            "freeze_backbone": True,
            "mr_lora": True,
            "mr_lora_scope": str(getattr(config, "MR_LORA_SCOPE", "graph_attention_routing")),
            "mr_lora_train_output_head": train_output_head,
            "trainable_keywords": ["lora_A", "lora_B"],
            "trainable_params": int(trainable),
            "frozen_params": int(total - trainable),
            "total_params": int(total),
            "trainable_tensors": int(len(trainable_names)),
            "frozen_tensors": int(len(frozen_names)),
            "trainable_name_examples": trainable_names[:20],
            "frozen_name_examples": frozen_names[:20],
        }

    freeze = bool(getattr(config, "FREEZE_FINETUNE_BACKBONE", False) if freeze_backbone is None else freeze_backbone)
    keywords = _finetune_trainable_keywords()
    if not freeze:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = count_total_parameters(model)
        return {
            "freeze_backbone": False,
            "trainable_keywords": list(keywords),
            "trainable_params": int(trainable),
            "frozen_params": int(total - trainable),
            "total_params": int(total),
            "trainable_tensors": int(sum(1 for p in model.parameters() if p.requires_grad)),
            "frozen_tensors": int(sum(1 for p in model.parameters() if not p.requires_grad)),
        }

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, param in model.named_parameters():
        should_train = any(keyword in name for keyword in keywords)
        param.requires_grad_(should_train)
        if should_train:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    if not trainable_names:
        raise RuntimeError(
            "FREEZE_FINETUNE_BACKBONE left no trainable parameters. "
            f"Check FINETUNE_TRAINABLE_KEYWORDS={keywords}."
        )
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    return {
        "freeze_backbone": True,
        "trainable_keywords": list(keywords),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "total_params": int(total),
        "trainable_tensors": int(len(trainable_names)),
        "frozen_tensors": int(len(frozen_names)),
        "trainable_name_examples": trainable_names[:20],
        "frozen_name_examples": frozen_names[:20],
    }


def maybe_enable_mr_lora(model: torch.nn.Module, device: torch.device | None = None) -> dict[str, object]:
    """Inject MR-LoRA adapters only for explicitly requested real-domain fine-tuning."""
    if not bool(getattr(config, "USE_MR_LORA", False)):
        return {"enabled": False}

    scope = str(getattr(config, "MR_LORA_SCOPE", "graph_attention_routing")).lower()
    mr_lora_scope_families(scope)
    graph_rank = int(getattr(config, "MR_LORA_RANK_GRAPH", 8))
    routing_rank = int(getattr(config, "MR_LORA_RANK_ROUTING", 4))
    graph_alpha = float(getattr(config, "MR_LORA_ALPHA_GRAPH", 16.0))
    routing_alpha = float(getattr(config, "MR_LORA_ALPHA_ROUTING", 8.0))
    summary = inject_mr_lora(
        model,
        graph_rank=graph_rank,
        routing_rank=routing_rank,
        graph_alpha=graph_alpha,
        routing_alpha=routing_alpha,
        dropout=float(getattr(config, "MR_LORA_DROPOUT", 0.05)),
    )
    if device is not None:
        model.to(device)
    names = mr_lora_parameter_names(model)
    if not names:
        raise RuntimeError("USE_MR_LORA=True but no LoRA parameters were injected.")
    return {
        "enabled": True,
        "scope": scope,
        "graph_modules": int(summary.graph_modules),
        "attention_modules": int(summary.attention_modules),
        "routing_modules": int(summary.routing_modules),
        "total_modules": int(summary.total_modules),
        "lora_parameter_tensors": int(len(names)),
        "lora_parameter_examples": names[:20],
    }


def build_finetune_optimizer(model: torch.nn.Module) -> AdamW:
    """AdamW with lower LR for pretrained graph backbones and higher LR for heads/gates."""
    if not bool(getattr(config, "USE_LAYERWISE_FINETUNE_LR", False)):
        return AdamW((p for p in model.parameters() if p.requires_grad), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    backbone_lr = float(getattr(config, "FINETUNE_BACKBONE_LR", config.LR))
    head_lr = float(getattr(config, "FINETUNE_HEAD_LR", config.LR))
    agent_lr = float(getattr(config, "FINETUNE_AGENT_LR", head_lr))
    param_groups: dict[str, dict[str, object]] = {
        "backbone": {"params": [], "lr": backbone_lr, "name": "backbone"},
        "heads": {"params": [], "lr": head_lr, "name": "heads"},
        "agent": {"params": [], "lr": agent_lr, "name": "agent"},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "router" in name or "gate_state_proj" in name:
            param_groups["agent"]["params"].append(param)
        elif "experts." in name and any(token in name for token in ("readout.", "mu_head.", "log_b_head.")):
            param_groups["heads"]["params"].append(param)
        else:
            param_groups["backbone"]["params"].append(param)

    groups = [group for group in param_groups.values() if group["params"]]
    return AdamW(groups, weight_decay=config.WEIGHT_DECAY)


def _active_model_name() -> str:
    return "mtam_hg"


def _output_mu(outputs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(outputs):
        return outputs
    return outputs["mu"]


def build_experiment_model() -> torch.nn.Module:
    input_dim = len(config.input_node_names(config.USE_EL_AS_INPUT))
    use_laplace = bool(config.USE_LAPLACE)
    return MTAMHG(input_dim=input_dim, use_laplace=use_laplace)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device: torch.device,
    cluster_model: WorkingConditionCluster | None = None,
    batch_weight_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    use_internal_agent_weight: bool | None = None,
    use_agent_reward: bool | None = None,
) -> dict[str, float]:
    model.train()
    accum: dict[str, float] = {}

    for x, y, _sample_ids in train_loader:
        x = x.to(device)
        y = y.to(device)
        cluster_labels: torch.Tensor | None = None
        if cluster_model is not None and cluster_model.is_fitted:
            cluster_labels = cluster_model.predict_tensor(x).to(device)
        outputs = model(x)
        batch_weights: torch.Tensor | None = None
        if batch_weight_fn is not None:
            batch_weights = batch_weight_fn(x, y).detach().to(device=device, dtype=y.dtype)
            if batch_weights.numel() != y.shape[0]:
                raise ValueError("Batch weights must contain one value per training sample.")
            batch_weights = batch_weights.reshape(-1, 1)
            if not torch.isfinite(batch_weights).all() or torch.any(batch_weights < 0.0):
                raise ValueError("Batch weights must be finite and non-negative.")
        loss, logs = total_loss(
            outputs,
            y,
            x=x,
            mask=None,
            edge_mask=None,
            batch_weights=batch_weights,
            cluster_labels=cluster_labels,
            use_internal_agent_weight=use_internal_agent_weight,
            use_agent_reward=use_agent_reward,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()

        for key, value in logs.items():
            accum[key] = accum.get(key, 0.0) + value

    return {key: value / max(len(train_loader), 1) for key, value in accum.items()}


@torch.no_grad()
def evaluate_prediction_loss(model, loader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        outputs = model(x)
        loss = prediction_loss(outputs, y, weights=None)
        total += float(loss.detach().cpu()) * x.shape[0]
        count += int(x.shape[0])
    return total / max(count, 1)


def split_metadata(data_bundle) -> dict[str, object]:
    """Serializable split metadata for diagnostics and checkpoints."""
    return {
        "seed": int(data_bundle.model_seed),
        "model_seed": int(data_bundle.model_seed),
        "run_seed": int(data_bundle.run_seed),
        "split_seed": int(data_bundle.split_seed),
        "split_method": str(data_bundle.split_method),
        "split_id_sha256": dict(data_bundle.split_hashes),
        "combined_split_sha256": str(data_bundle.combined_split_hash),
        "source_data_sha256": str(data_bundle.source_sha256),
        "schema_sha256": str(data_bundle.schema_hash),
        "split_sizes": {key: int(value) for key, value in data_bundle.split_sizes.items()},
        "data_path": str(data_bundle.data_path),
        "label_column": str(data_bundle.label_column),
        "train_sample_ids": [int(x) for x in np.asarray(data_bundle.train_sample_ids).reshape(-1)],
        "val_sample_ids": [int(x) for x in np.asarray(data_bundle.val_sample_ids).reshape(-1)],
        "test_sample_ids": [int(x) for x in np.asarray(data_bundle.test_sample_ids).reshape(-1)],
    }


def save_split_artifacts(data_bundle, output_dir: Path) -> None:
    """Save train/val/test sample ids for run-level reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = split_metadata(data_bundle)
    save_json(output_dir / "split_indices.json", meta)
    rows = []
    for split_name, ids in (
        ("train", data_bundle.train_sample_ids),
        ("val", data_bundle.val_sample_ids),
        ("test", data_bundle.test_sample_ids),
    ):
        for position, sample_id in enumerate(np.asarray(ids).reshape(-1)):
            rows.append(
                {
                    "split": split_name,
                    "position": int(position),
                    "sample_id": int(sample_id),
                    "seed": int(data_bundle.model_seed),
                    "model_seed": int(data_bundle.model_seed),
                    "run_seed": int(data_bundle.run_seed),
                    "split_seed": int(data_bundle.split_seed),
                    "split_method": data_bundle.split_method,
                    "split_id_sha256": data_bundle.split_hashes[split_name],
                    "combined_split_sha256": data_bundle.combined_split_hash,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "split_indices.csv", index=False, encoding="utf-8")


def checkpoint_payload(
    model: torch.nn.Module,
    data_bundle,
    epoch: int,
    **extra: object,
) -> dict[str, object]:
    """Standard .pth checkpoint payload used across training modes."""
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "seed": int(data_bundle.model_seed),
        "model_seed": int(data_bundle.model_seed),
        "run_seed": int(data_bundle.run_seed),
        "node_names": getattr(model, "node_names", data_bundle.graph_node_names),
        "model_name": _active_model_name(),
        "split": split_metadata(data_bundle),
        "checkpoint_format": "mtam_hg_v1",
    }
    payload.update(extra)
    return payload


def checkpoint_selection_score(val_metrics: dict[str, float]) -> tuple[float, dict[str, float | str]]:
    """Return the scalar checkpoint score and its components for validation metrics."""
    metric = str(getattr(config, "CHECKPOINT_SELECTION_METRIC", "rmse") or "rmse").lower()
    rmse = float(val_metrics.get("RMSE", float("inf")))
    tail_mae = float(val_metrics.get("TAIL_MAE", val_metrics.get("Tail_MAE", float("nan"))))
    tail_lambda = float(getattr(config, "CHECKPOINT_TAIL_MAE_LAMBDA", 0.0) or 0.0)
    if metric in {"rmse", "val_rmse"}:
        score = rmse
    elif metric in {"rmse_tail", "rmse_plus_tail", "val_rmse_plus_tail"}:
        tail_term = 0.0 if not np.isfinite(tail_mae) else tail_lambda * tail_mae
        score = rmse + tail_term
    else:
        raise ValueError(
            "CHECKPOINT_SELECTION_METRIC must be 'rmse' or 'rmse_tail', "
            f"got {metric!r}."
        )
    return score, {
        "checkpoint_selection_metric": metric,
        "checkpoint_score": float(score),
        "checkpoint_score_rmse": rmse,
        "checkpoint_score_tail_mae": tail_mae,
        "checkpoint_tail_mae_lambda": tail_lambda,
    }


def supervised_finetune(
    model,
    data_bundle,
    device: torch.device,
    epochs: int | None = None,
    freeze_backbone: bool | None = None,
    quality_agent: torch.nn.Module | None = None,
    quality_agent_calibration_fn: Callable[
        [torch.nn.Module, torch.nn.Module, object, torch.device, int],
        tuple[
            dict[str, float],
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        ],
    ]
    | None = None,
) -> tuple[float, int, int, bool]:
    if (quality_agent is None) != (quality_agent_calibration_fn is None):
        raise ValueError("quality_agent and quality_agent_calibration_fn must be provided together.")
    save_split_artifacts(data_bundle, config.RESULT_DIR)
    epochs = config.EPOCHS if epochs is None else epochs
    mr_lora_summary = maybe_enable_mr_lora(model, device=device)
    finetune_policy = configure_finetune_trainability(model, freeze_backbone=freeze_backbone)
    finetune_policy["mr_lora_injection"] = mr_lora_summary
    finetune_policy["cbtg_quality_agent_calibration"] = quality_agent is not None
    finetune_policy["cbtg_external_prediction_weights"] = quality_agent is not None
    finetune_policy["internal_router_sample_weights"] = False if quality_agent is not None else None
    finetune_policy["internal_router_reward"] = False if quality_agent is not None else None
    save_json(config.RESULT_DIR / "finetune_policy.json", finetune_policy)
    optimizer = build_finetune_optimizer(model)

    cluster_model: WorkingConditionCluster | None = None
    if bool(getattr(config, "USE_CLUSTER_BALANCE_REWARD", False)):
        train_dataset = data_bundle.train_loader.dataset
        x_train_np = train_dataset.x.detach().cpu().numpy()
        cluster_model = WorkingConditionCluster()
        cluster_model.fit(x_train_np, data_bundle.y_train_raw)
        finetune_policy["cluster_balance_enabled"] = True
        finetune_policy["num_working_condition_clusters"] = cluster_model.effective_n_clusters
        save_json(config.RESULT_DIR / "finetune_policy.json", finetune_policy)
    best_score = float("inf")
    best_epoch = -1
    epochs_run = 0
    stopped_early = False
    early_stopping_patience = int(getattr(config, "EARLY_STOPPING_PATIENCE", 0) or 0)
    early_stopping_min_delta = float(getattr(config, "EARLY_STOPPING_MIN_DELTA", 0.0))
    epochs_without_improvement = 0
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    log_path = config.LOG_DIR / "train_log.csv"
    if log_path.exists():
        log_path.unlink()
    train_start = time.perf_counter()

    progress = tqdm(range(1, epochs + 1), desc="Supervised finetune", unit="epoch")
    for epoch in progress:
        epochs_run = epoch
        if quality_agent is not None and quality_agent_calibration_fn is not None:
            quality_agent_logs, batch_weight_fn = quality_agent_calibration_fn(
                model,
                quality_agent,
                data_bundle,
                device,
                epoch,
            )
        else:
            quality_agent_logs, batch_weight_fn = {}, None
        train_logs = train_one_epoch(
            model,
            data_bundle.train_loader,
            optimizer,
            device,
            cluster_model=cluster_model,
            batch_weight_fn=batch_weight_fn,
            use_internal_agent_weight=False if quality_agent is not None else None,
            use_agent_reward=False if quality_agent is not None else None,
        )
        train_metrics, _ = evaluate_model(model, data_bundle.train_loader, device, data_bundle)
        val_metrics, _ = evaluate_model(model, data_bundle.val_loader, device, data_bundle)
        checkpoint_score, checkpoint_score_logs = checkpoint_selection_score(val_metrics)
        val_loss = evaluate_prediction_loss(model, data_bundle.val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_logs.get("total_loss", float("nan")),
            "val_loss": val_loss,
            "train_rmse": train_metrics.get("RMSE", float("nan")),
            "val_rmse": val_metrics.get("RMSE", float("nan")),
            "train_mae": train_metrics.get("MAE", float("nan")),
            "val_mae": val_metrics.get("MAE", float("nan")),
            **checkpoint_score_logs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "weight_decay": optimizer.param_groups[0]["weight_decay"],
            "finetune_backbone_lr": float(getattr(config, "FINETUNE_BACKBONE_LR", config.LR)),
            "finetune_head_lr": float(getattr(config, "FINETUNE_HEAD_LR", config.LR)),
            "finetune_agent_lr": float(getattr(config, "FINETUNE_AGENT_LR", config.LR)),
            "use_layerwise_finetune_lr": bool(getattr(config, "USE_LAYERWISE_FINETUNE_LR", False)),
            "freeze_finetune_backbone": bool(finetune_policy["freeze_backbone"]),
            "trainable_params": int(finetune_policy["trainable_params"]),
            "frozen_params": int(finetune_policy["frozen_params"]),
            "dropout": config.DROPOUT,
            "model_params": count_trainable_parameters(model),
            "elapsed_train_time": time.perf_counter() - train_start,
            "pred_loss": train_logs.get("pred_loss", float("nan")),
            "moe_loss": train_logs.get("moe_loss", float("nan")),
            "graph_loss": train_logs.get("graph_loss", float("nan")),
            "mask_loss": train_logs.get("mask_loss", float("nan")),
            "edge_loss": train_logs.get("edge_loss", float("nan")),
            **train_logs,
            **quality_agent_logs,
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
                if f"val_{k}".lower() not in {"val_rmse", "val_mae"}
            },
        }
        append_csv(log_path, row)
        progress.set_postfix(
            train_rmse=f"{train_metrics.get('RMSE', float('nan')):.4f}",
            val_rmse=f"{val_metrics.get('RMSE', float('nan')):.4f}",
        )

        if checkpoint_score < best_score - early_stopping_min_delta:
            best_score = checkpoint_score
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = checkpoint_payload(
                model,
                data_bundle,
                epoch,
                best_rmse=float(val_metrics["RMSE"]),
                best_checkpoint_score=best_score,
                **checkpoint_score_logs,
            )
            if quality_agent is not None:
                checkpoint["synthetic_quality_agent_state_dict"] = quality_agent.state_dict()
                checkpoint["cbtg_real_calibrated"] = True
            torch.save(checkpoint, ckpt_path)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                stopped_early = True
                progress.set_postfix(
                    train_rmse=f"{train_metrics.get('RMSE', float('nan')):.4f}",
                    val_rmse=f"{val_metrics.get('RMSE', float('nan')):.4f}",
                    early_stop=f"patience={early_stopping_patience}",
                )
                break

    return best_score, best_epoch, epochs_run, stopped_early


def run_training(data_bundle, mode: str = "train", epochs: int | None = None) -> tuple[torch.nn.Module, dict[str, float]]:
    ensure_dirs(config.CHECKPOINT_DIR, config.LOG_DIR, config.RESULT_DIR)
    set_seed(config.SEED)
    device = resolve_device()
    model = build_experiment_model().to(device)
    start_time = time.perf_counter()

    best_epoch = -1
    _, best_epoch, epochs_run, stopped_early = supervised_finetune(
        model,
        data_bundle,
        device,
        epochs=epochs,
        freeze_backbone=False,
    )
    ckpt_path = config.CHECKPOINT_DIR / "best_model.pth"
    if ckpt_path.exists():
        load_checkpoint(model, ckpt_path, device)

    inference_start = time.perf_counter()
    metrics, collected = evaluate_model(model, data_bundle.test_loader, device, data_bundle)
    inference_time = time.perf_counter() - inference_start
    train_time = time.perf_counter() - start_time
    param_breakdown = moe_parameter_breakdown(model)
    active_model = _active_model_name()
    metrics.update(
        {
            "total_params": int(param_breakdown["Params"]),
            **param_breakdown,
            "Finetune_Freeze_Backbone": False,
            "Train_Time": train_time,
            "Inference_Time": inference_time,
            "Inference_Time_Per_Sample": inference_time / max(int(data_bundle.split_sizes.get("test", 0)), 1),
            "Best_Epoch": best_epoch,
            "Epochs_Run": epochs_run,
            "Stopped_Early": bool(stopped_early),
            "Checkpoint_Selection_Metric": str(getattr(config, "CHECKPOINT_SELECTION_METRIC", "rmse")),
            "Checkpoint_Tail_MAE_Lambda": float(getattr(config, "CHECKPOINT_TAIL_MAE_LAMBDA", 0.0) or 0.0),
            "Model": active_model,
            "Experiment_Group": getattr(config, "EXPERIMENT_NAME", "default"),
            "Data_Split": data_bundle.split_method,
            "Train_Size": int(data_bundle.split_sizes.get("train", 0)),
            "Seed": int(config.SEED),
            "MoE_Aux_Lambda": float(getattr(config, "MOE_AUX_LAMBDA", getattr(config, "LAMBDA_MOE", 0.01))),
        }
    )
    save_evaluation_outputs(metrics, collected, config.RESULT_DIR)
    return model, metrics


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    has_lora_state = any(".lora_" in name for name in state)
    has_lora_model = bool(mr_lora_parameter_names(model))
    if (has_lora_state or bool(getattr(config, "USE_MR_LORA", False))) and not has_lora_model:
        maybe_enable_mr_lora(model, device=device)
    model.load_state_dict(state)
    model.to(device)


def _inverse_y_for_bundle(y: np.ndarray, data_bundle) -> np.ndarray:
    if config.STANDARDIZE_Y:
        return data_bundle.y_scaler.inverse_transform(y)
    return y


def run_small_batch_overfit(data_bundle, epochs: int | None = None) -> dict[str, float | bool | str]:
    """Overfit a fixed 32-sample training subset for model sanity checks."""
    ensure_dirs(config.CHECKPOINT_DIR, config.LOG_DIR, config.RESULT_DIR)
    set_seed(config.SEED)
    device = resolve_device()
    model = build_experiment_model().to(device)
    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=0.0)

    epochs = epochs or config.SMALL_BATCH_OVERFIT_EPOCHS
    n = min(config.SMALL_BATCH_OVERFIT_SIZE, len(data_bundle.train_loader.dataset))
    x = data_bundle.train_loader.dataset.x[:n].to(device)
    y = data_bundle.train_loader.dataset.y[:n].to(device)
    log_path = config.LOG_DIR / "small_batch_overfit_rmse.csv"
    if log_path.exists():
        log_path.unlink()

    initial_rmse = None
    final_rmse = None
    model.train()
    for epoch in range(1, epochs + 1):
        outputs = model(x)
        loss, logs = total_loss(outputs, y, x=x)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()

        with torch.no_grad():
            mu = _output_mu(outputs).detach().cpu().numpy()
            target = y.detach().cpu().numpy()
            mu_real = _inverse_y_for_bundle(mu, data_bundle)
            target_real = _inverse_y_for_bundle(target, data_bundle)
            rmse = float(np.sqrt(np.mean((mu_real - target_real) ** 2)))
        if initial_rmse is None:
            initial_rmse = rmse
        final_rmse = rmse
        append_csv(
            log_path,
            {
                "epoch": epoch,
                "train_rmse": rmse,
                "loss": logs["total_loss"],
                "pred_loss": logs["pred_loss"],
                "moe_loss": logs["moe_loss"],
            },
        )

    assert initial_rmse is not None and final_rmse is not None
    success = final_rmse <= config.SMALL_BATCH_OVERFIT_SUCCESS_RMSE or final_rmse <= 0.5 * initial_rmse
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "initial_rmse": initial_rmse,
            "final_rmse": final_rmse,
            "success": success,
        },
        config.CHECKPOINT_DIR / "small_batch_overfit_model.pth",
    )
    return {
        "samples": n,
        "epochs": epochs,
        "initial_rmse": initial_rmse,
        "final_rmse": final_rmse,
        "success": success,
        "rmse_curve": str(log_path),
    }
