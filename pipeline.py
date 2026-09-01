"""Run MTAM-HG training, generation, or evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import re
import time
from datetime import datetime
from pathlib import Path

import config
from dataset import create_dataloaders
from models.mr_lora import MR_LORA_SCOPE_FAMILIES
from protocol import (
    DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS,
    DEFAULT_TABDIFF_NUM_SAMPLES,
    MAIN_PY_MODE_CHOICES,
    SUPERVISED_MAIN_TRAIN_MODE,
)


def _set_mapped_config(attr: str, value) -> None:
    setattr(config, attr, value)
    lower = attr.lower()
    if hasattr(config, lower):
        setattr(config, lower, value)
    if attr in {"MOE_AUX_LAMBDA", "LAMBDA_MOE"}:
        config.MOE_AUX_LAMBDA = value
        config.LAMBDA_MOE = value
        config.MOE_AUX_WEIGHT = value
        config.moe_aux_lambda = value


def set_output_dir(path: str) -> None:
    if not path:
        return
    output_dir = Path(path)
    if not output_dir.is_absolute():
        output_dir = config.PROJECT_ROOT / output_dir
    config.OUTPUT_DIR = output_dir
    config.CHECKPOINT_DIR = output_dir / "checkpoints"
    config.LOG_DIR = output_dir / "logs"
    config.RESULT_DIR = output_dir / "results"
    config.SCALER_PATH = output_dir / "scaler.pkl"


def append_run_timestamp_to_output_dir() -> None:
    """Create a timestamped run directory under the configured output root."""
    current = Path(config.OUTPUT_DIR)
    if re.fullmatch(r"\d{8}_\d{6}", current.name):
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_output_dir(str(current / timestamp))


def load_config_overrides(path: str | None) -> None:
    config.CONFIG_SHA256 = ""
    if not path:
        return
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = Path(config.PROJECT_ROOT) / cfg_path
    if not cfg_path.exists() or cfg_path.resolve() == Path(config.__file__).resolve():
        return
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        from config_loader import load_yaml_config_text
        from protocol_integrity import assert_file_snapshot_current, capture_file_snapshot

        config_snapshot = capture_file_snapshot(cfg_path, include_content=True)
        assert config_snapshot.content is not None
        loaded = load_yaml_config_text(config_snapshot.content.decode("utf-8"))
        if loaded.get("experiment_name"):
            config.EXPERIMENT_NAME = str(loaded["experiment_name"])
        if loaded.get("output_base"):
            set_output_dir(str(loaded["output_base"]))
        mapping = {
            "data_path": "DATA_PATH",
            "batch_size": "BATCH_SIZE",
            "epochs": "EPOCHS",
            "lr": "LR",
            "weight_decay": "WEIGHT_DECAY",
            "dropout": "DROPOUT",
            "seed": "SEED",
            "split_seed": "SPLIT_SEED",
            "split_method": "SPLIT_METHOD",
            "generation_seed": "TABDIFF_GENERATION_SEED",
            "graph_backbone_layers": "GRAPH_BACKBONE_LAYERS",
            "top_k": "TOP_K",
            "lambda_moe": "LAMBDA_MOE",
            "moe_aux_lambda": "MOE_AUX_LAMBDA",
            "moe_gate_temperature": "MOE_GATE_TEMPERATURE",
            "moe_balance_prob_lambda": "MOE_BALANCE_PROB_LAMBDA",
            "moe_balance_usage_lambda": "MOE_BALANCE_USAGE_LAMBDA",
            "moe_entropy_reg_lambda": "MOE_ENTROPY_REG_LAMBDA",
            "expert_calibration_lambda": "EXPERT_CALIBRATION_LAMBDA",
            "expert_diversity_lambda": "EXPERT_DIVERSITY_LAMBDA",
            "expert_calibration_quality_lambda": "EXPERT_CALIBRATION_QUALITY_LAMBDA",
            "expert_calibration_quality_index": "EXPERT_CALIBRATION_QUALITY_INDEX",
            "num_experts": "NUM_EXPERTS",
            "agent_hidden_dim": "AGENT_HIDDEN_DIM",
            "agent_dropout": "AGENT_DROPOUT",
            "agent_use_process_features": "AGENT_USE_PROCESS_FEATURES",
            "agent_reason_dim": "AGENT_REASON_DIM",
            "agent_reliability_routing_lambda": "AGENT_RELIABILITY_ROUTING_LAMBDA",
            "agent_confidence_reg_lambda": "AGENT_CONFIDENCE_REG_LAMBDA",
            "use_agent_reward": "USE_AGENT_REWARD",
            "agent_reward_lambda": "AGENT_REWARD_LAMBDA",
            "reward_alpha_error": "REWARD_ALPHA_ERROR",
            "reward_alpha_uncertainty": "REWARD_ALPHA_UNCERTAINTY",
            "reward_alpha_entropy": "REWARD_ALPHA_ENTROPY",
            "reward_alpha_tail": "REWARD_ALPHA_TAIL",
            "reward_clamp_min": "REWARD_CLAMP_MIN",
            "reward_clamp_max": "REWARD_CLAMP_MAX",
            "target_confidence_mean": "TARGET_CONFIDENCE_MEAN",
            "confidence_mean_reg_lambda": "CONFIDENCE_MEAN_REG_LAMBDA",
            "confidence_entropy_reg_lambda": "CONFIDENCE_ENTROPY_REG_LAMBDA",
            "tail_quantile_low": "TAIL_QUANTILE_LOW",
            "tail_quantile_high": "TAIL_QUANTILE_HIGH",
            "tail_threshold_mode": "TAIL_THRESHOLD_MODE",
            "use_tabdiff_generation": "USE_TABDIFF_GENERATION",
            "tabdiff_repo_path": "TABDIFF_REPO_PATH",
            "tabdiff_data_dir": "TABDIFF_DATA_DIR",
            "tabdiff_output_dir": "TABDIFF_OUTPUT_DIR",
            "tabdiff_dataname": "TABDIFF_DATANAME",
            "tabdiff_exp_name": "TABDIFF_EXP_NAME",
            "tabdiff_num_samples": "TABDIFF_NUM_SAMPLES",
            "tabdiff_train_epochs": "TABDIFF_TRAIN_EPOCHS",
            "tabdiff_low_tail_ratio": "TABDIFF_LOW_TAIL_RATIO",
            "tabdiff_high_tail_ratio": "TABDIFF_HIGH_TAIL_RATIO",
            "tabdiff_gpu": "TABDIFF_GPU",
            "tabdiff_ckpt_path": "TABDIFF_CKPT_PATH",
            "tabdiff_mechanism_constraint": "TABDIFF_MECHANISM_CONSTRAINT",
            "tabdiff_mechanism_lambda": "TABDIFF_MECHANISM_LAMBDA",
            "tabdiff_guidance_scale": "TABDIFF_GUIDANCE_SCALE",
            "tabdiff_mechanism_temperature_hold_tolerance": "TABDIFF_MECHANISM_TEMPERATURE_HOLD_TOLERANCE",
            "tabdiff_mechanism_yield_tolerance": "TABDIFF_MECHANISM_YIELD_TOLERANCE",
            "tabdiff_trainable_scope": "TABDIFF_TRAINABLE_SCOPE",
            "tabdiff_min_save_epoch": "TABDIFF_MIN_SAVE_EPOCH",
            "tabdiff_finetune_lr": "TABDIFF_FINETUNE_LR",
            "tabdiff_finetune_steps": "TABDIFF_FINETUNE_STEPS",
            "tabdiff_num_timesteps_override": "TABDIFF_NUM_TIMESTEPS_OVERRIDE",
            "tabdiff_stochastic_sampler": "TABDIFF_STOCHASTIC_SAMPLER",
            "early_stopping_patience": "EARLY_STOPPING_PATIENCE",
            "min_delta": "EARLY_STOPPING_MIN_DELTA",
            "use_synthetic_pretrain": "USE_SYNTHETIC_PRETRAIN",
            "synthetic_data_path": "SYNTHETIC_DATA_PATH",
            "synthetic_label_col": "SYNTHETIC_LABEL_COL",
            "synthetic_pretrain_epochs": "SYNTHETIC_PRETRAIN_EPOCHS",
            "synthetic_batch_size": "SYNTHETIC_BATCH_SIZE",
            "synthetic_use_agent_weight": "SYNTHETIC_USE_AGENT_WEIGHT",
            "synthetic_use_reward_loss": "SYNTHETIC_USE_REWARD_LOSS",
            "synthetic_agent_hidden_dim": "SYNTHETIC_AGENT_HIDDEN_DIM",
            "synthetic_agent_attention_dim": "SYNTHETIC_AGENT_ATTENTION_DIM",
            "synthetic_agent_attention_heads": "SYNTHETIC_AGENT_ATTENTION_HEADS",
            "synthetic_agent_dropout": "SYNTHETIC_AGENT_DROPOUT",
            "synthetic_agent_epochs": "SYNTHETIC_AGENT_EPOCHS",
            "synthetic_agent_lr": "SYNTHETIC_AGENT_LR",
            "synthetic_confidence_threshold": "SYNTHETIC_CONFIDENCE_THRESHOLD",
            "synthetic_pretrain_confidence_threshold": "SYNTHETIC_PRETRAIN_CONFIDENCE_THRESHOLD",
            "synthetic_save_diagnostics": "SYNTHETIC_SAVE_DIAGNOSTICS",
            "synthetic_use_process_consistency": "SYNTHETIC_USE_PROCESS_CONSISTENCY",
            "synthetic_process_consistency_threshold": "SYNTHETIC_PROCESS_CONSISTENCY_THRESHOLD",
            "synthetic_process_range_quantile_low": "SYNTHETIC_PROCESS_RANGE_QUANTILE_LOW",
            "synthetic_process_range_quantile_high": "SYNTHETIC_PROCESS_RANGE_QUANTILE_HIGH",
            "synthetic_process_range_margin": "SYNTHETIC_PROCESS_RANGE_MARGIN",
            "synthetic_process_knn_k": "SYNTHETIC_PROCESS_KNN_K",
            "synthetic_process_range_weight": "SYNTHETIC_PROCESS_RANGE_WEIGHT",
            "synthetic_process_manifold_weight": "SYNTHETIC_PROCESS_MANIFOLD_WEIGHT",
            "synthetic_process_label_weight": "SYNTHETIC_PROCESS_LABEL_WEIGHT",
            "synthetic_process_score_power": "SYNTHETIC_PROCESS_SCORE_POWER",
            "synthetic_use_mechanism_consistency": "SYNTHETIC_USE_MECHANISM_CONSISTENCY",
            "synthetic_mechanism_score_power": "SYNTHETIC_MECHANISM_SCORE_POWER",
            "synthetic_reward_mse_weight": "SYNTHETIC_REWARD_MSE_WEIGHT",
            "synthetic_reward_process_weight": "SYNTHETIC_REWARD_PROCESS_WEIGHT",
            "synthetic_reward_mechanism_weight": "SYNTHETIC_REWARD_MECHANISM_WEIGHT",
            "use_dynamic_synthetic_agent": "USE_DYNAMIC_SYNTHETIC_AGENT",
            "dynamic_synthetic_refresh_epochs": "DYNAMIC_SYNTHETIC_REFRESH_EPOCHS",
            "dynamic_synthetic_warmup_epochs": "DYNAMIC_SYNTHETIC_WARMUP_EPOCHS",
            "dynamic_synthetic_use_sampler": "DYNAMIC_SYNTHETIC_USE_SAMPLER",
            "dynamic_synthetic_use_loss_weight": "DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT",
            "dynamic_synthetic_top_ratio": "DYNAMIC_SYNTHETIC_TOP_RATIO",
            "dynamic_synthetic_weight_min": "DYNAMIC_SYNTHETIC_WEIGHT_MIN",
            "dynamic_synthetic_weight_max": "DYNAMIC_SYNTHETIC_WEIGHT_MAX",
            "dynamic_synthetic_ema": "DYNAMIC_SYNTHETIC_EMA",
            "dynamic_synthetic_error_weight": "DYNAMIC_SYNTHETIC_ERROR_WEIGHT",
            "dynamic_synthetic_train_region_weight": "DYNAMIC_SYNTHETIC_TRAIN_REGION_WEIGHT",
            "dynamic_synthetic_scarcity_weight": "DYNAMIC_SYNTHETIC_SCARCITY_WEIGHT",
            "dynamic_synthetic_real_feedback_weight": "DYNAMIC_SYNTHETIC_REAL_FEEDBACK_WEIGHT",
            "dynamic_synthetic_quota_strength": "DYNAMIC_SYNTHETIC_QUOTA_STRENGTH",
            "dynamic_synthetic_quota_min": "DYNAMIC_SYNTHETIC_QUOTA_MIN",
            "dynamic_synthetic_quota_max": "DYNAMIC_SYNTHETIC_QUOTA_MAX",
            "dynamic_synthetic_reliability_floor": "DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR",
            "dynamic_synthetic_scarcity_bins": "DYNAMIC_SYNTHETIC_SCARCITY_BINS",
            "dynamic_synthetic_process_power": "DYNAMIC_SYNTHETIC_PROCESS_POWER",
            "dynamic_synthetic_mechanism_power": "DYNAMIC_SYNTHETIC_MECHANISM_POWER",
            "dynamic_synthetic_train_reward_metric": "DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC",
            "dynamic_synthetic_train_tail_lambda": "DYNAMIC_SYNTHETIC_TRAIN_TAIL_LAMBDA",
            "use_layerwise_finetune_lr": "USE_LAYERWISE_FINETUNE_LR",
            "finetune_backbone_lr": "FINETUNE_BACKBONE_LR",
            "finetune_head_lr": "FINETUNE_HEAD_LR",
            "finetune_agent_lr": "FINETUNE_AGENT_LR",
            "finetune_quality_agent_lr": "FINETUNE_QUALITY_AGENT_LR",
            "freeze_finetune_backbone": "FREEZE_FINETUNE_BACKBONE",
            "finetune_trainable_keywords": "FINETUNE_TRAINABLE_KEYWORDS",
            "use_mr_lora": "USE_MR_LORA",
            "mr_lora_scope": "MR_LORA_SCOPE",
            "mr_lora_rank_graph": "MR_LORA_RANK_GRAPH",
            "mr_lora_rank_routing": "MR_LORA_RANK_ROUTING",
            "mr_lora_alpha_graph": "MR_LORA_ALPHA_GRAPH",
            "mr_lora_alpha_routing": "MR_LORA_ALPHA_ROUTING",
            "mr_lora_dropout": "MR_LORA_DROPOUT",
            "mr_lora_train_output_head": "MR_LORA_TRAIN_OUTPUT_HEAD",
            "use_cluster_balance_reward": "USE_CLUSTER_BALANCE_REWARD",
            "num_working_condition_clusters": "NUM_WORKING_CONDITION_CLUSTERS",
            "cluster_balance_lambda": "CLUSTER_BALANCE_LAMBDA",
            "reward_alpha_cluster": "REWARD_ALPHA_CLUSTER",
            "checkpoint_selection_metric": "CHECKPOINT_SELECTION_METRIC",
            "checkpoint_tail_mae_lambda": "CHECKPOINT_TAIL_MAE_LAMBDA",
        }
        for key, value in loaded.items():
            if key in {"cli", "experiment_name", "output_base"}:
                continue
            if key in mapping:
                _set_mapped_config(mapping[key], value)
        for key, value in loaded.get("cli", {}).items():
            if key == "hidden_dim":
                config.D_MODEL = int(value)
                config.GRAPH_EMBED_DIM = int(value)
            elif key == "expert_dim":
                config.EXPERT_DIM = int(value)
            elif key == "use_laplace" and value:
                config.USE_LAPLACE = True
            elif key == "no_laplace" and value:
                config.USE_LAPLACE = False
            elif key == "agent_use_expert_preds":
                config.AGENT_USE_EXPERT_PREDS = bool(value)
                config.agent_use_expert_preds = config.AGENT_USE_EXPERT_PREDS
            elif key == "agent_use_process_features":
                config.AGENT_USE_PROCESS_FEATURES = bool(value)
                config.agent_use_process_features = config.AGENT_USE_PROCESS_FEATURES
            elif key == "agent_use_uncertainty":
                config.AGENT_USE_UNCERTAINTY = bool(value)
                config.agent_use_uncertainty = config.AGENT_USE_UNCERTAINTY
            elif key == "agent_output_sample_confidence":
                config.AGENT_OUTPUT_SAMPLE_CONFIDENCE = bool(value)
                config.agent_output_sample_confidence = config.AGENT_OUTPUT_SAMPLE_CONFIDENCE
            elif key == "agent_use_sample_weight_for_supervised_loss":
                config.AGENT_USE_SAMPLE_WEIGHT_FOR_SUPERVISED_LOSS = bool(value)
                config.agent_use_sample_weight_for_supervised_loss = config.AGENT_USE_SAMPLE_WEIGHT_FOR_SUPERVISED_LOSS
            elif key == "use_confidence_weighted_supervised_loss":
                config.USE_CONFIDENCE_WEIGHTED_SUPERVISED_LOSS = bool(value)
                config.use_confidence_weighted_supervised_loss = config.USE_CONFIDENCE_WEIGHTED_SUPERVISED_LOSS
            elif key == "use_virtual_quality_node":
                config.USE_VIRTUAL_QUALITY_NODE = bool(value)
            elif key == "no_virtual_quality_node" and value:
                config.USE_VIRTUAL_QUALITY_NODE = False
            elif key in mapping:
                _set_mapped_config(mapping[key], value)
        assert_file_snapshot_current(config_snapshot, "YAML configuration")
        config.CONFIG_SHA256 = config_snapshot.sha256
        return
    spec = importlib.util.spec_from_file_location("user_config", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config file: {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in dir(module):
        if name.isupper():
            setattr(config, name, getattr(module, name))


def _has_value(value) -> bool:
    return value not in (None, "", [])


def apply_cli_overrides(args: argparse.Namespace) -> None:
    if args.experiment_name:
        config.EXPERIMENT_NAME = args.experiment_name
    if args.output_dir:
        set_output_dir(args.output_dir)
    if args.label_col:
        config.LABEL_COL = args.label_col
        config.LABEL_COL = args.label_col

    value_overrides = {
        "data_path": "DATA_PATH",
        "epochs": "EPOCHS",
        "batch_size": "BATCH_SIZE",
        "lr": "LR",
        "weight_decay": "WEIGHT_DECAY",
        "dropout": "DROPOUT",
        "seed": "SEED",
        "split_seed": "SPLIT_SEED",
        "generation_seed": "TABDIFF_GENERATION_SEED",
        "expert_calibration_lambda": "EXPERT_CALIBRATION_LAMBDA",
        "expert_calibration_quality_lambda": "EXPERT_CALIBRATION_QUALITY_LAMBDA",
        "expert_calibration_quality_index": "EXPERT_CALIBRATION_QUALITY_INDEX",
        "agent_hidden_dim": "AGENT_HIDDEN_DIM",
        "agent_dropout": "AGENT_DROPOUT",
        "agent_reason_dim": "AGENT_REASON_DIM",
        "agent_reliability_routing_lambda": "AGENT_RELIABILITY_ROUTING_LAMBDA",
        "agent_confidence_reg_lambda": "AGENT_CONFIDENCE_REG_LAMBDA",
        "agent_reward_lambda": "AGENT_REWARD_LAMBDA",
        "synthetic_data_path": "SYNTHETIC_DATA_PATH",
        "synthetic_pretrain_epochs": "SYNTHETIC_PRETRAIN_EPOCHS",
        "synthetic_agent_hidden_dim": "SYNTHETIC_AGENT_HIDDEN_DIM",
        "synthetic_agent_attention_dim": "SYNTHETIC_AGENT_ATTENTION_DIM",
        "synthetic_agent_attention_heads": "SYNTHETIC_AGENT_ATTENTION_HEADS",
        "synthetic_agent_dropout": "SYNTHETIC_AGENT_DROPOUT",
        "synthetic_agent_epochs": "SYNTHETIC_AGENT_EPOCHS",
        "synthetic_agent_lr": "SYNTHETIC_AGENT_LR",
        "synthetic_confidence_threshold": "SYNTHETIC_CONFIDENCE_THRESHOLD",
        "synthetic_pretrain_confidence_threshold": "SYNTHETIC_PRETRAIN_CONFIDENCE_THRESHOLD",
        "synthetic_process_consistency_threshold": "SYNTHETIC_PROCESS_CONSISTENCY_THRESHOLD",
        "synthetic_process_range_margin": "SYNTHETIC_PROCESS_RANGE_MARGIN",
        "synthetic_process_knn_k": "SYNTHETIC_PROCESS_KNN_K",
        "synthetic_process_score_power": "SYNTHETIC_PROCESS_SCORE_POWER",
        "synthetic_mechanism_score_power": "SYNTHETIC_MECHANISM_SCORE_POWER",
        "synthetic_reward_mse_weight": "SYNTHETIC_REWARD_MSE_WEIGHT",
        "synthetic_reward_process_weight": "SYNTHETIC_REWARD_PROCESS_WEIGHT",
        "synthetic_reward_mechanism_weight": "SYNTHETIC_REWARD_MECHANISM_WEIGHT",
        "use_dynamic_synthetic_agent": "USE_DYNAMIC_SYNTHETIC_AGENT",
        "dynamic_synthetic_refresh_epochs": "DYNAMIC_SYNTHETIC_REFRESH_EPOCHS",
        "dynamic_synthetic_warmup_epochs": "DYNAMIC_SYNTHETIC_WARMUP_EPOCHS",
        "dynamic_synthetic_use_sampler": "DYNAMIC_SYNTHETIC_USE_SAMPLER",
        "dynamic_synthetic_use_loss_weight": "DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT",
        "dynamic_synthetic_top_ratio": "DYNAMIC_SYNTHETIC_TOP_RATIO",
        "dynamic_synthetic_weight_min": "DYNAMIC_SYNTHETIC_WEIGHT_MIN",
        "dynamic_synthetic_weight_max": "DYNAMIC_SYNTHETIC_WEIGHT_MAX",
        "dynamic_synthetic_ema": "DYNAMIC_SYNTHETIC_EMA",
        "dynamic_synthetic_error_weight": "DYNAMIC_SYNTHETIC_ERROR_WEIGHT",
        "dynamic_synthetic_train_region_weight": "DYNAMIC_SYNTHETIC_TRAIN_REGION_WEIGHT",
        "dynamic_synthetic_scarcity_weight": "DYNAMIC_SYNTHETIC_SCARCITY_WEIGHT",
        "dynamic_synthetic_real_feedback_weight": "DYNAMIC_SYNTHETIC_REAL_FEEDBACK_WEIGHT",
        "dynamic_synthetic_quota_strength": "DYNAMIC_SYNTHETIC_QUOTA_STRENGTH",
        "dynamic_synthetic_quota_min": "DYNAMIC_SYNTHETIC_QUOTA_MIN",
        "dynamic_synthetic_quota_max": "DYNAMIC_SYNTHETIC_QUOTA_MAX",
        "dynamic_synthetic_reliability_floor": "DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR",
        "dynamic_synthetic_scarcity_bins": "DYNAMIC_SYNTHETIC_SCARCITY_BINS",
        "dynamic_synthetic_process_power": "DYNAMIC_SYNTHETIC_PROCESS_POWER",
        "dynamic_synthetic_mechanism_power": "DYNAMIC_SYNTHETIC_MECHANISM_POWER",
        "dynamic_synthetic_train_reward_metric": "DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC",
        "dynamic_synthetic_train_tail_lambda": "DYNAMIC_SYNTHETIC_TRAIN_TAIL_LAMBDA",
        "tabdiff_num_samples": "TABDIFF_NUM_SAMPLES",
        "tabdiff_gpu": "TABDIFF_GPU",
        "tabdiff_mechanism_constraint": "TABDIFF_MECHANISM_CONSTRAINT",
        "tabdiff_mechanism_lambda": "TABDIFF_MECHANISM_LAMBDA",
        "tabdiff_guidance_scale": "TABDIFF_GUIDANCE_SCALE",
        "tabdiff_mechanism_temperature_hold_tolerance": "TABDIFF_MECHANISM_TEMPERATURE_HOLD_TOLERANCE",
        "tabdiff_trainable_scope": "TABDIFF_TRAINABLE_SCOPE",
        "tabdiff_min_save_epoch": "TABDIFF_MIN_SAVE_EPOCH",
        "tabdiff_finetune_lr": "TABDIFF_FINETUNE_LR",
        "tabdiff_finetune_steps": "TABDIFF_FINETUNE_STEPS",
        "tabdiff_num_timesteps_override": "TABDIFF_NUM_TIMESTEPS_OVERRIDE",
        "tabdiff_stochastic_sampler": "TABDIFF_STOCHASTIC_SAMPLER",
        "split_method": "SPLIT_METHOD",
        "graph_backbone_layers": "GRAPH_BACKBONE_LAYERS",
        "early_stopping_patience": "EARLY_STOPPING_PATIENCE",
        "min_delta": "EARLY_STOPPING_MIN_DELTA",
        "finetune_backbone_lr": "FINETUNE_BACKBONE_LR",
        "finetune_head_lr": "FINETUNE_HEAD_LR",
        "finetune_agent_lr": "FINETUNE_AGENT_LR",
        "finetune_quality_agent_lr": "FINETUNE_QUALITY_AGENT_LR",
        "finetune_trainable_keywords": "FINETUNE_TRAINABLE_KEYWORDS",
        "mr_lora_scope": "MR_LORA_SCOPE",
        "mr_lora_rank_graph": "MR_LORA_RANK_GRAPH",
        "mr_lora_rank_routing": "MR_LORA_RANK_ROUTING",
        "mr_lora_alpha_graph": "MR_LORA_ALPHA_GRAPH",
        "mr_lora_alpha_routing": "MR_LORA_ALPHA_ROUTING",
        "mr_lora_dropout": "MR_LORA_DROPOUT",
        "num_working_condition_clusters": "NUM_WORKING_CONDITION_CLUSTERS",
        "cluster_balance_lambda": "CLUSTER_BALANCE_LAMBDA",
        "reward_alpha_cluster": "REWARD_ALPHA_CLUSTER",
        "checkpoint_selection_metric": "CHECKPOINT_SELECTION_METRIC",
        "checkpoint_tail_mae_lambda": "CHECKPOINT_TAIL_MAE_LAMBDA",
    }
    for arg_name, cfg_name in value_overrides.items():
        value = getattr(args, arg_name, None)
        if _has_value(value):
            _set_mapped_config(cfg_name, value)

    if args.hidden_dim is not None:
        config.D_MODEL = args.hidden_dim
        config.GRAPH_EMBED_DIM = args.hidden_dim
    if args.expert_dim is not None:
        config.EXPERT_DIM = args.expert_dim
        if args.hidden_dim is None:
            config.D_MODEL = args.expert_dim
    if args.lambda_moe is not None or args.moe_aux_lambda is not None:
        value = args.moe_aux_lambda if args.moe_aux_lambda is not None else args.lambda_moe
        config.LAMBDA_MOE = value
        config.MOE_AUX_LAMBDA = value
        config.moe_aux_lambda = value
    true_flags = {
        "agent_use_sample_weight_for_supervised_loss": "AGENT_USE_SAMPLE_WEIGHT_FOR_SUPERVISED_LOSS",
        "use_agent_reward": "USE_AGENT_REWARD",
        "use_confidence_weighted_supervised_loss": "USE_CONFIDENCE_WEIGHTED_SUPERVISED_LOSS",
        "use_synthetic_process_consistency": "SYNTHETIC_USE_PROCESS_CONSISTENCY",
        "use_layerwise_finetune_lr": "USE_LAYERWISE_FINETUNE_LR",
        "use_mr_lora": "USE_MR_LORA",
        "mr_lora_train_output_head": "MR_LORA_TRAIN_OUTPUT_HEAD",
        "use_cluster_balance_reward": "USE_CLUSTER_BALANCE_REWARD",
    }
    for arg_name, cfg_name in true_flags.items():
        if getattr(args, arg_name):
            _set_mapped_config(cfg_name, True)

    if args.no_agent_reward:
        _set_mapped_config("USE_AGENT_REWARD", False)
    if args.no_synthetic_process_consistency:
        _set_mapped_config("SYNTHETIC_USE_PROCESS_CONSISTENCY", False)
    if args.no_layerwise_finetune_lr:
        _set_mapped_config("USE_LAYERWISE_FINETUNE_LR", False)
    if args.no_mr_lora:
        _set_mapped_config("USE_MR_LORA", False)
    if args.no_mr_lora_train_output_head:
        _set_mapped_config("MR_LORA_TRAIN_OUTPUT_HEAD", False)
    if args.no_cluster_balance_reward:
        _set_mapped_config("USE_CLUSTER_BALANCE_REWARD", False)
    if args.freeze_finetune_backbone is not None:
        _set_mapped_config("FREEZE_FINETUNE_BACKBONE", bool(args.freeze_finetune_backbone))
    if args.no_el:
        config.USE_EL_AS_INPUT = False
    if args.no_laplace:
        config.USE_LAPLACE = False

def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MTAM-HG yield-strength prediction")
    parser.add_argument(
        "--mode",
        choices=[
            *MAIN_PY_MODE_CHOICES,
        ],
        default=SUPERVISED_MAIN_TRAIN_MODE,
    )
    parser.add_argument("--config", default="", help="Optional Python config file with uppercase overrides.")
    parser.add_argument("--experiment_name", default="", help="Name used in metrics summaries.")
    parser.add_argument("--output_dir", default="", help="Experiment output root; creates checkpoints/logs/results under it.")
    parser.add_argument("--data_path", default="", help="CAPL CSV/XLS/XLSX path. Required unless --allow_synthetic is set.")
    parser.add_argument("--label_col", default="", help="Yield strength label column name.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Fixed train/validation/test partition seed; independent of the model seed.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=None,
        help="TabDiff generation seed. Official deterministic mode requires 0.",
    )
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--expert_dim", type=int, default=None)
    parser.add_argument("--graph_backbone_layers", type=int, default=None)
    parser.add_argument("--lambda_moe", type=float, default=None, help="MoE load-balance auxiliary weight.")
    parser.add_argument("--moe_aux_lambda", type=float, default=None, help="Alias for the MoE-IPOHGN auxiliary weight.")
    parser.add_argument("--expert_calibration_lambda", type=float, default=None)
    parser.add_argument("--expert_calibration_quality_lambda", type=float, default=None)
    parser.add_argument("--expert_calibration_quality_index", type=int, default=None)
    parser.add_argument("--agent_hidden_dim", type=int, default=None)
    parser.add_argument("--agent_dropout", type=float, default=None)
    parser.add_argument("--agent_reason_dim", type=int, default=None)
    parser.add_argument("--agent_reliability_routing_lambda", type=float, default=None)
    parser.add_argument("--agent_use_sample_weight_for_supervised_loss", action="store_true")
    parser.add_argument("--agent_confidence_reg_lambda", type=float, default=None)
    parser.add_argument("--use_agent_reward", action="store_true", help="Enable reward-driven Agent confidence loss.")
    parser.add_argument("--no_agent_reward", action="store_true", help="Disable reward-driven Agent confidence loss.")
    parser.add_argument("--agent_reward_lambda", type=float, default=None)
    parser.add_argument("--synthetic_data_path", default="", help="Synthetic CAPL file for CBTG pretraining.")
    parser.add_argument("--synthetic_pretrain_epochs", type=int, default=None)
    parser.add_argument("--synthetic_agent_hidden_dim", type=int, default=None)
    parser.add_argument("--synthetic_agent_attention_dim", type=int, default=None)
    parser.add_argument("--synthetic_agent_attention_heads", type=int, default=None)
    parser.add_argument("--synthetic_agent_dropout", type=float, default=None)
    parser.add_argument("--synthetic_agent_epochs", type=int, default=None)
    parser.add_argument("--synthetic_agent_lr", type=float, default=None)
    parser.add_argument("--synthetic_confidence_threshold", type=float, default=None)
    parser.add_argument("--synthetic_pretrain_confidence_threshold", type=float, default=None)
    parser.add_argument("--synthetic_process_consistency_threshold", type=float, default=None)
    parser.add_argument("--synthetic_process_range_margin", type=float, default=None)
    parser.add_argument("--synthetic_process_knn_k", type=int, default=None)
    parser.add_argument("--synthetic_process_score_power", type=float, default=None)
    parser.add_argument("--synthetic_mechanism_score_power", type=float, default=None)
    parser.add_argument("--synthetic_reward_mse_weight", type=float, default=None)
    parser.add_argument("--synthetic_reward_process_weight", type=float, default=None)
    parser.add_argument("--synthetic_reward_mechanism_weight", type=float, default=None)
    parser.add_argument("--use_dynamic_synthetic_agent", dest="use_dynamic_synthetic_agent", action="store_true", default=None)
    parser.add_argument("--no_dynamic_synthetic_agent", dest="use_dynamic_synthetic_agent", action="store_false")
    parser.add_argument("--dynamic_synthetic_refresh_epochs", type=int, default=None)
    parser.add_argument("--dynamic_synthetic_warmup_epochs", type=int, default=None)
    parser.add_argument("--dynamic_synthetic_use_sampler", type=_str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--dynamic_synthetic_use_loss_weight", type=_str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--dynamic_synthetic_top_ratio", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_weight_min", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_weight_max", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_ema", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_error_weight", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_train_region_weight", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_scarcity_weight", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_real_feedback_weight", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_quota_strength", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_quota_min", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_quota_max", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_reliability_floor", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_scarcity_bins", type=int, default=None)
    parser.add_argument("--dynamic_synthetic_process_power", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_mechanism_power", type=float, default=None)
    parser.add_argument("--dynamic_synthetic_train_reward_metric", choices=["", "rmse", "mae", "rmse_tail"], default="")
    parser.add_argument("--dynamic_synthetic_train_tail_lambda", type=float, default=None)
    parser.add_argument("--use_cluster_balance_reward", action="store_true")
    parser.add_argument("--no_cluster_balance_reward", action="store_true")
    parser.add_argument("--num_working_condition_clusters", type=int, default=None)
    parser.add_argument("--cluster_balance_lambda", type=float, default=None)
    parser.add_argument("--reward_alpha_cluster", type=float, default=None)
    parser.add_argument("--finetune_backbone_lr", type=float, default=None)
    parser.add_argument("--finetune_head_lr", type=float, default=None)
    parser.add_argument("--finetune_agent_lr", type=float, default=None)
    parser.add_argument("--finetune_quality_agent_lr", type=float, default=None)
    parser.add_argument("--freeze_finetune_backbone", dest="freeze_finetune_backbone", action="store_true", default=None)
    parser.add_argument("--no_freeze_finetune_backbone", dest="freeze_finetune_backbone", action="store_false")
    parser.add_argument("--use_mr_lora", action="store_true", help="Enable MR-LoRA adapters only for real-domain fine-tuning.")
    parser.add_argument("--no_mr_lora", action="store_true", help="Force-disable MR-LoRA adapters.")
    parser.add_argument(
        "--mr_lora_scope",
        choices=list(MR_LORA_SCOPE_FAMILIES),
        default=None,
        help="MR-LoRA graph, process-order-attention, and routing adapter families to inject.",
    )
    parser.add_argument("--mr_lora_rank_graph", type=int, default=None)
    parser.add_argument("--mr_lora_rank_routing", type=int, default=None)
    parser.add_argument("--mr_lora_alpha_graph", type=float, default=None)
    parser.add_argument("--mr_lora_alpha_routing", type=float, default=None)
    parser.add_argument("--mr_lora_dropout", type=float, default=None)
    parser.add_argument("--mr_lora_train_output_head", action="store_true")
    parser.add_argument("--no_mr_lora_train_output_head", action="store_true")
    parser.add_argument("--checkpoint_selection_metric", choices=["rmse", "rmse_tail"], default=None)
    parser.add_argument("--checkpoint_tail_mae_lambda", type=float, default=None)
    parser.add_argument("--use_synthetic_process_consistency", action="store_true")
    parser.add_argument("--use_layerwise_finetune_lr", action="store_true")
    parser.add_argument("--no_layerwise_finetune_lr", action="store_true")
    parser.add_argument("--no_synthetic_process_consistency", action="store_true")
    parser.add_argument("--tabdiff_num_samples", type=int, default=None)
    parser.add_argument("--tabdiff_gpu", type=int, default=None)
    parser.add_argument("--dry_run", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--use_confidence_weighted_supervised_loss", action="store_true")
    parser.add_argument("--split_method", choices=["stratified_random", "chronological"], default="")
    parser.add_argument("--no_el", action="store_true", help="Exclude EL from input nodes.")
    parser.add_argument("--no_laplace", action="store_true", help="Use deterministic MSE regression.")
    parser.add_argument("--allow_synthetic", action="store_true", help="Allow synthetic data for development smoke tests only.")
    parser.add_argument("--debug_overfit_small_batch", action="store_true", help="Train on a fixed 32-sample batch for 300-epoch sanity check.")
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=None)
    return parser


def run_generate_synthetic_tabdiff(args: argparse.Namespace) -> dict[str, object]:
    from generation.postprocess import postprocess_tabdiff_samples
    from generation.prepare import prepare_tabdiff_data
    from generation.sample import run_tabdiff_sample
    from generation.train import run_tabdiff_train

    prepared = prepare_tabdiff_data(
        data_path=config.DATA_PATH,
        label_col=config.LABEL_COL,
        output_dir=config.TABDIFF_DATA_DIR,
        repo_path=config.TABDIFF_REPO_PATH,
        dataset_name=getattr(config, "TABDIFF_DATANAME", "capl"),
        split_seed=int(config.SPLIT_SEED),
        split_method=str(config.SPLIT_METHOD),
        generation_seed=int(getattr(config, "TABDIFF_GENERATION_SEED", 0)),
    )
    train_result = run_tabdiff_train(dry_run=bool(args.dry_run))
    sample_result = run_tabdiff_sample(
        dry_run=bool(args.dry_run),
        num_samples=int(getattr(config, "TABDIFF_NUM_SAMPLES", DEFAULT_TABDIFF_NUM_SAMPLES)),
        checkpoint_path=train_result.get("checkpoint_path"),
        expected_checkpoint_sha256=train_result.get("checkpoint_sha256"),
    )
    postprocess_result: dict[str, object] | None = None
    if not args.dry_run:
        postprocess_result = postprocess_tabdiff_samples(
            checkpoint_path=train_result.get("checkpoint_path"),
            expected_checkpoint_sha256=train_result.get("checkpoint_sha256"),
        )
    result = {
        "prepared": prepared,
        "tabdiff_train": train_result,
        "tabdiff_sample": sample_result,
        "postprocess": postprocess_result,
    }
    print(result)
    return result


def run_evaluate(data_bundle, checkpoint_path: str) -> dict[str, float]:
    from evaluate import evaluate_model, save_evaluation_outputs
    from train import (
        _active_model_name,
        build_experiment_model,
        load_checkpoint,
        moe_parameter_breakdown,
        resolve_device,
    )

    device = resolve_device()
    model = build_experiment_model().to(device)
    load_checkpoint(model, checkpoint_path, device)
    inference_start = time.perf_counter()
    metrics, collected = evaluate_model(model, data_bundle.test_loader, device, data_bundle)
    inference_time = time.perf_counter() - inference_start
    param_breakdown = moe_parameter_breakdown(model)
    metrics.update(
        {
            "total_params": int(param_breakdown["Params"]),
            **param_breakdown,
            "Inference_Time": inference_time,
            "Inference_Time_Per_Sample": inference_time / max(int(data_bundle.split_sizes.get("test", 0)), 1),
            "Model": _active_model_name(),
            "Experiment_Group": getattr(config, "EXPERIMENT_NAME", "default"),
            "Data_Split": data_bundle.split_method,
            "Train_Size": int(data_bundle.split_sizes.get("train", 0)),
            "Seed": int(config.SEED),
        }
    )
    save_evaluation_outputs(metrics, collected, config.RESULT_DIR)
    return metrics


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_config_overrides(args.config)
    apply_cli_overrides(args)
    if args.mode != "generate_synthetic_tabdiff":
        append_run_timestamp_to_output_dir()

    if args.mode == "generate_synthetic_tabdiff":
        run_generate_synthetic_tabdiff(args)
        return

    if args.mode == SUPERVISED_MAIN_TRAIN_MODE:
        from protocol_integrity import (
            SyntheticProvenanceError,
            validate_synthetic_provenance_for_runner,
        )

        synthetic_path = Path(config.SYNTHETIC_DATA_PATH)
        if not synthetic_path.is_absolute():
            synthetic_path = config.PROJECT_ROOT / synthetic_path
        if bool(getattr(config, "USE_TABDIFF_GENERATION", False)) and "smoke" not in synthetic_path.name.lower():
            regenerate = not synthetic_path.exists()
            if not regenerate:
                try:
                    validate_synthetic_provenance_for_runner(
                        synthetic_path,
                        config.DATA_PATH,
                        int(config.SPLIT_SEED),
                        str(config.SPLIT_METHOD),
                        int(getattr(config, "TABDIFF_GENERATION_SEED", 0)),
                        validate_current_generation_config=True,
                    )
                except SyntheticProvenanceError as exc:
                    print(f"[Protocol] Existing synthetic data cannot be reused: {exc}")
                    regenerate = True
            if regenerate:
                run_generate_synthetic_tabdiff(args)

    data_bundle = create_dataloaders(
        data_path=config.DATA_PATH,
        label_column=config.LABEL_COL,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        split_seed=int(config.SPLIT_SEED),
        use_el_as_input=config.USE_EL_AS_INPUT,
        split_method=config.SPLIT_METHOD,
        allow_synthetic=args.allow_synthetic,
    )

    if args.debug_overfit_small_batch:
        from train import run_small_batch_overfit

        result = run_small_batch_overfit(data_bundle, epochs=args.epochs or config.SMALL_BATCH_OVERFIT_EPOCHS)
        print(result)
        return

    if args.mode == "evaluate":
        metrics = run_evaluate(data_bundle, args.checkpoint or str(config.CHECKPOINT_DIR / "best_model.pth"))
    else:
        from protocol_integrity import validate_synthetic_provenance_for_data_bundle
        from training.cbtg import run_cbtg_pretraining

        validate_synthetic_provenance_for_data_bundle(
            config.SYNTHETIC_DATA_PATH,
            data_bundle,
            int(getattr(config, "TABDIFF_GENERATION_SEED", 0)),
        )

        _, metrics = run_cbtg_pretraining(
            data_bundle,
            synthetic_data_path=config.SYNTHETIC_DATA_PATH,
            synthetic_epochs=args.synthetic_pretrain_epochs
            or getattr(config, "SYNTHETIC_PRETRAIN_EPOCHS", DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS),
            finetune_real=args.mode == SUPERVISED_MAIN_TRAIN_MODE,
            real_epochs=args.epochs,
        )
    print(metrics)


if __name__ == "__main__":
    main()
