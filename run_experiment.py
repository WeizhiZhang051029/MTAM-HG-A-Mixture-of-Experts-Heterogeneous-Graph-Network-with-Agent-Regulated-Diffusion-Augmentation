"""Run the paper-aligned MTAM-HG main experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from models.mr_lora import MR_LORA_SCOPE_FAMILIES
from protocol import (
    BOOL_VALUE_FLAGS,
    CONFIRMATORY_LOCKED_ARGS,
    CONFIRMATORY_YAML_CANONICAL_SHA256,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CBTG_CROSS_RUN_VALIDATION_STD_PATH,
    DEFAULT_CLUSTER_BALANCE_LAMBDA,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DROPOUT,
    DEFAULT_DYNAMIC_SYNTHETIC_EMA,
    DEFAULT_DYNAMIC_SYNTHETIC_ERROR_WEIGHT,
    DEFAULT_DYNAMIC_SYNTHETIC_MECHANISM_POWER,
    DEFAULT_DYNAMIC_SYNTHETIC_PROCESS_POWER,
    DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_MAX,
    DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_MIN,
    DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_STRENGTH,
    DEFAULT_DYNAMIC_SYNTHETIC_REAL_FEEDBACK_WEIGHT,
    DEFAULT_DYNAMIC_SYNTHETIC_REFRESH_EPOCHS,
    DEFAULT_DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR,
    DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_BINS,
    DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_WEIGHT,
    DEFAULT_DYNAMIC_SYNTHETIC_TOP_RATIO,
    DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_REGION_WEIGHT,
    DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC,
    DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_TAIL_LAMBDA,
    DEFAULT_DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT,
    DEFAULT_DYNAMIC_SYNTHETIC_USE_SAMPLER,
    DEFAULT_DYNAMIC_SYNTHETIC_WARMUP_EPOCHS,
    DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MAX,
    DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MIN,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EPOCHS,
    DEFAULT_FINETUNE_AGENT_LR,
    DEFAULT_FINETUNE_BACKBONE_LR,
    DEFAULT_FINETUNE_HEAD_LR,
    DEFAULT_FINETUNE_QUALITY_AGENT_LR,
    DEFAULT_FREEZE_FINETUNE_BACKBONE,
    DEFAULT_GENERATION_SEED,
    DEFAULT_LABEL_COL,
    DEFAULT_LR,
    DEFAULT_MAIN_CHECKPOINT_SELECTION_METRIC,
    DEFAULT_MAIN_CHECKPOINT_TAIL_MAE_LAMBDA,
    DEFAULT_MAIN_OUTPUT_ROOT,
    DEFAULT_MR_LORA_ALPHA_GRAPH,
    DEFAULT_MR_LORA_ALPHA_ROUTING,
    DEFAULT_MR_LORA_DROPOUT,
    DEFAULT_MR_LORA_RANK_GRAPH,
    DEFAULT_MR_LORA_RANK_ROUTING,
    DEFAULT_MR_LORA_SCOPE,
    DEFAULT_MR_LORA_TRAIN_OUTPUT_HEAD,
    DEFAULT_NUM_WORKING_CONDITION_CLUSTERS,
    DEFAULT_REWARD_ALPHA_CLUSTER,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_METHOD,
    DEFAULT_SPLIT_SEED,
    DEFAULT_SYNTHETIC_AGENT_ATTENTION_DIM,
    DEFAULT_SYNTHETIC_AGENT_ATTENTION_HEADS,
    DEFAULT_SYNTHETIC_AGENT_EPOCHS,
    DEFAULT_SYNTHETIC_AGENT_HIDDEN_DIM,
    DEFAULT_SYNTHETIC_AGENT_LR,
    DEFAULT_SYNTHETIC_CONFIDENCE_THRESHOLD,
    DEFAULT_SYNTHETIC_DATA_PATH,
    DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS,
    DEFAULT_TABDIFF_NUM_SAMPLES,
    DEFAULT_USE_CLUSTER_BALANCE_REWARD,
    DEFAULT_USE_DYNAMIC_SYNTHETIC_AGENT,
    DEFAULT_USE_MR_LORA,
    DEFAULT_WEIGHT_DECAY,
    MAIN_EXPERIMENT_MODEL,
    MAIN_EXPERIMENT_NAMES,
    MAIN_TRAIN_ARG_SPECS,
    MR_LORA_ARG_SPECS,
    RUNNER_SUMMARY_NAME,
    SUPERVISED_MAIN_EXPERIMENT_NAME,
    SUPERVISED_MAIN_TRAIN_MODE,
)
from protocol_integrity import (
    ProtocolIntegrityError,
    SyntheticProvenanceError,
    assert_file_snapshot_current,
    canonical_sha256,
    capture_file_snapshot,
    file_sha256,
    scientific_code_sha256,
    validate_cbtg_cross_run_validation_std,
    validate_synthetic_provenance_for_runner,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_METRICS = ("RMSE", "MAE", "MAPE", "R2", "TAIL_MAE")
CONFIRMATORY_METRICS = PAPER_METRICS
CBTG_VALIDATION_METRICS = ("RMSE", "MAE", "MAPE", "ONE_MINUS_R2", "TAIL_MAE")
OPERATIONAL_YAML_KEYS = {
    "model_seeds",
    "output_base",
    "tabdiff_repo_path",
    "tabdiff_data_dir",
    "tabdiff_output_dir",
    "synthetic_data_path",
    "cbtg_cross_run_validation_std_path",
}
OPERATIONAL_RUNTIME_KEYS = {
    "PROJECT_ROOT",
    "DATA_PATH",
    "SYNTHETIC_DATA_PATH",
    "CBTG_CROSS_RUN_VALIDATION_STD_PATH",
    "OUTPUT_DIR",
    "CHECKPOINT_DIR",
    "LOG_DIR",
    "RESULT_DIR",
    "SCALER_PATH",
    "TABDIFF_REPO_PATH",
    "TABDIFF_DATA_DIR",
    "TABDIFF_OUTPUT_DIR",
    "DEVICE",
    "TABDIFF_GPU",
}
METRICS_INTEGRITY_FIELDS = (
    "Split_Seed",
    "Split_Method",
    "Combined_Split_SHA256",
    "Source_Data_SHA256",
    "Synthetic_SHA256",
    "Generation_Seed",
    "Config_SHA256",
    "Effective_Protocol_SHA256",
    "CBTG_Cross_Run_Validation_STD_SHA256",
    "Synthetic_Provenance_SHA256",
)


@dataclass
class ExperimentPhase:
    name: str
    command: list[str]
    run_dir: Path
    output_path: Path | None = None
    checkpoint_path: Path | None = None


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper-aligned MTAM-HG main experiment.")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--label_col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--synthetic_data_path", default=DEFAULT_SYNTHETIC_DATA_PATH)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output_root", default=DEFAULT_MAIN_OUTPUT_ROOT)
    parser.add_argument(
        "--main_experiment_name",
        default="",
        help="Optional final training experiment name; defaults to the configured model name.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--generation_seed", type=int, default=DEFAULT_GENERATION_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--synthetic_pretrain_epochs", type=int, default=DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS)
    parser.add_argument("--synthetic_agent_epochs", type=int, default=DEFAULT_SYNTHETIC_AGENT_EPOCHS)
    parser.add_argument("--synthetic_agent_lr", type=float, default=DEFAULT_SYNTHETIC_AGENT_LR)
    parser.add_argument("--synthetic_agent_hidden_dim", type=int, default=DEFAULT_SYNTHETIC_AGENT_HIDDEN_DIM)
    parser.add_argument("--synthetic_agent_attention_dim", type=int, default=DEFAULT_SYNTHETIC_AGENT_ATTENTION_DIM)
    parser.add_argument("--synthetic_agent_attention_heads", type=int, default=DEFAULT_SYNTHETIC_AGENT_ATTENTION_HEADS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--agent_dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--synthetic_agent_dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--synthetic_confidence_threshold", type=float, default=DEFAULT_SYNTHETIC_CONFIDENCE_THRESHOLD)
    parser.add_argument("--synthetic_pretrain_confidence_threshold", type=float, default=0.0)
    parser.add_argument("--use_dynamic_synthetic_agent", action="store_true", default=DEFAULT_USE_DYNAMIC_SYNTHETIC_AGENT)
    parser.add_argument("--no_dynamic_synthetic_agent", dest="use_dynamic_synthetic_agent", action="store_false")
    parser.add_argument("--dynamic_synthetic_refresh_epochs", type=int, default=DEFAULT_DYNAMIC_SYNTHETIC_REFRESH_EPOCHS)
    parser.add_argument("--dynamic_synthetic_warmup_epochs", type=int, default=DEFAULT_DYNAMIC_SYNTHETIC_WARMUP_EPOCHS)
    parser.add_argument("--dynamic_synthetic_use_sampler", type=_str_to_bool, nargs="?", const=True, default=DEFAULT_DYNAMIC_SYNTHETIC_USE_SAMPLER)
    parser.add_argument("--dynamic_synthetic_use_loss_weight", type=_str_to_bool, nargs="?", const=True, default=DEFAULT_DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT)
    parser.add_argument("--dynamic_synthetic_top_ratio", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_TOP_RATIO)
    parser.add_argument("--dynamic_synthetic_weight_min", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MIN)
    parser.add_argument("--dynamic_synthetic_weight_max", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MAX)
    parser.add_argument("--dynamic_synthetic_ema", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_EMA)
    parser.add_argument("--dynamic_synthetic_error_weight", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_ERROR_WEIGHT)
    parser.add_argument("--dynamic_synthetic_train_region_weight", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_REGION_WEIGHT)
    parser.add_argument("--dynamic_synthetic_scarcity_weight", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_WEIGHT)
    parser.add_argument("--dynamic_synthetic_real_feedback_weight", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_REAL_FEEDBACK_WEIGHT)
    parser.add_argument("--dynamic_synthetic_quota_strength", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_STRENGTH)
    parser.add_argument("--dynamic_synthetic_quota_min", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_MIN)
    parser.add_argument("--dynamic_synthetic_quota_max", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_MAX)
    parser.add_argument("--dynamic_synthetic_reliability_floor", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR)
    parser.add_argument("--dynamic_synthetic_scarcity_bins", type=int, default=DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_BINS)
    parser.add_argument("--dynamic_synthetic_process_power", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_PROCESS_POWER)
    parser.add_argument("--dynamic_synthetic_mechanism_power", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_MECHANISM_POWER)
    parser.add_argument("--dynamic_synthetic_train_reward_metric", choices=["rmse", "mae", "rmse_tail"], default=DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC)
    parser.add_argument("--dynamic_synthetic_train_tail_lambda", type=float, default=DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_TAIL_LAMBDA)
    parser.add_argument("--use_cluster_balance_reward", action="store_true", default=DEFAULT_USE_CLUSTER_BALANCE_REWARD)
    parser.add_argument("--no_cluster_balance_reward", dest="use_cluster_balance_reward", action="store_false")
    parser.add_argument("--num_working_condition_clusters", type=int, default=DEFAULT_NUM_WORKING_CONDITION_CLUSTERS)
    parser.add_argument("--cluster_balance_lambda", type=float, default=DEFAULT_CLUSTER_BALANCE_LAMBDA)
    parser.add_argument("--reward_alpha_cluster", type=float, default=DEFAULT_REWARD_ALPHA_CLUSTER)
    parser.add_argument("--finetune_backbone_lr", type=float, default=DEFAULT_FINETUNE_BACKBONE_LR)
    parser.add_argument("--finetune_head_lr", type=float, default=DEFAULT_FINETUNE_HEAD_LR)
    parser.add_argument("--finetune_agent_lr", type=float, default=DEFAULT_FINETUNE_AGENT_LR)
    parser.add_argument("--finetune_quality_agent_lr", type=float, default=DEFAULT_FINETUNE_QUALITY_AGENT_LR)
    parser.add_argument("--use_layerwise_finetune_lr", action="store_true", default=True)
    parser.add_argument("--no_layerwise_finetune_lr", dest="use_layerwise_finetune_lr", action="store_false")
    parser.add_argument("--freeze_finetune_backbone", action="store_true", default=DEFAULT_FREEZE_FINETUNE_BACKBONE)
    parser.add_argument("--no_freeze_finetune_backbone", dest="freeze_finetune_backbone", action="store_false")
    parser.add_argument("--use_mr_lora", dest="use_mr_lora", action="store_true", default=DEFAULT_USE_MR_LORA)
    parser.add_argument("--no_mr_lora", dest="use_mr_lora", action="store_false")
    parser.add_argument(
        "--mr_lora_scope",
        choices=list(MR_LORA_SCOPE_FAMILIES),
        default=DEFAULT_MR_LORA_SCOPE,
        help="MR-LoRA adapter families; the paper-main setting is graph_attention_routing.",
    )
    parser.add_argument("--mr_lora_rank_graph", type=int, default=DEFAULT_MR_LORA_RANK_GRAPH)
    parser.add_argument("--mr_lora_rank_routing", type=int, default=DEFAULT_MR_LORA_RANK_ROUTING)
    parser.add_argument("--mr_lora_alpha_graph", type=float, default=DEFAULT_MR_LORA_ALPHA_GRAPH)
    parser.add_argument("--mr_lora_alpha_routing", type=float, default=DEFAULT_MR_LORA_ALPHA_ROUTING)
    parser.add_argument("--mr_lora_dropout", type=float, default=DEFAULT_MR_LORA_DROPOUT)
    parser.add_argument(
        "--mr_lora_train_output_head",
        dest="mr_lora_train_output_head",
        action="store_true",
        default=DEFAULT_MR_LORA_TRAIN_OUTPUT_HEAD,
    )
    parser.add_argument("--no_mr_lora_train_output_head", dest="mr_lora_train_output_head", action="store_false")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--early_stopping_patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--checkpoint_selection_metric",
        choices=["rmse", "rmse_tail"],
        default=DEFAULT_MAIN_CHECKPOINT_SELECTION_METRIC,
    )
    parser.add_argument("--checkpoint_tail_mae_lambda", type=float, default=DEFAULT_MAIN_CHECKPOINT_TAIL_MAE_LAMBDA)
    parser.add_argument("--split_method", choices=["stratified_random", "chronological"], default=DEFAULT_SPLIT_METHOD)
    parser.add_argument("--tabdiff_num_samples", type=int, default=None)
    parser.add_argument("--tabdiff_gpu", type=int, default=None)
    parser.add_argument(
        "--cbtg_cross_run_validation_std_path",
        default=DEFAULT_CBTG_CROSS_RUN_VALIDATION_STD_PATH,
    )
    parser.add_argument(
        "--skip_tabdiff_generation",
        action="store_true",
        help="Skip the TabDiff generation phase even when --synthetic_data_path is missing.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    selected_seeds = [int(seed) for seed in args.seeds]
    if len(selected_seeds) != 10 or len(set(selected_seeds)) != 10:
        raise ValueError("The confirmatory run requires ten distinct model seeds.")
    if int(args.split_seed) != DEFAULT_SPLIT_SEED:
        raise ValueError(
            f"The confirmatory split seed is fixed at {DEFAULT_SPLIT_SEED}; got {args.split_seed}."
        )
    if args.split_method != DEFAULT_SPLIT_METHOD:
        raise ValueError(
            "The confirmatory split method is fixed at "
            f"{DEFAULT_SPLIT_METHOD!r}; got {args.split_method!r}."
        )
    if int(args.generation_seed) != DEFAULT_GENERATION_SEED:
        raise ValueError(
            "The confirmatory TabDiff generation seed is fixed independently at "
            f"{DEFAULT_GENERATION_SEED}; got {args.generation_seed}."
        )
    if args.main_experiment_name not in {"", SUPERVISED_MAIN_EXPERIMENT_NAME}:
        raise ValueError(f"--main_experiment_name must be {SUPERVISED_MAIN_EXPERIMENT_NAME!r}.")
    if args.label_col != DEFAULT_LABEL_COL:
        raise ValueError(f"The confirmatory label column is fixed at {DEFAULT_LABEL_COL!r}.")
    for name, expected in CONFIRMATORY_LOCKED_ARGS.items():
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(
                f"The confirmatory value for --{name} is fixed at {expected!r}; got {actual!r}."
            )
    if args.tabdiff_num_samples is not None and int(args.tabdiff_num_samples) != DEFAULT_TABDIFF_NUM_SAMPLES:
        raise ValueError(
            f"The confirmatory value for --tabdiff_num_samples is fixed at {DEFAULT_TABDIFF_NUM_SAMPLES}."
        )
    if not str(args.cbtg_cross_run_validation_std_path).strip():
        raise ValueError("--cbtg_cross_run_validation_std_path is required.")
    positive_ints = {
        "epochs": args.epochs,
        "synthetic_pretrain_epochs": args.synthetic_pretrain_epochs,
        "batch_size": args.batch_size,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"--{name} must be positive, got {value}.")
    if args.synthetic_agent_epochs is not None and int(args.synthetic_agent_epochs) <= 0:
        raise ValueError(f"--synthetic_agent_epochs must be positive, got {args.synthetic_agent_epochs}.")
    if args.synthetic_agent_hidden_dim is not None and int(args.synthetic_agent_hidden_dim) <= 0:
        raise ValueError(f"--synthetic_agent_hidden_dim must be positive, got {args.synthetic_agent_hidden_dim}.")
    if args.synthetic_agent_attention_dim is not None and int(args.synthetic_agent_attention_dim) <= 0:
        raise ValueError(
            f"--synthetic_agent_attention_dim must be positive, got {args.synthetic_agent_attention_dim}."
        )
    if args.synthetic_agent_attention_heads is not None and int(args.synthetic_agent_attention_heads) <= 0:
        raise ValueError(
            f"--synthetic_agent_attention_heads must be positive, got {args.synthetic_agent_attention_heads}."
        )
    if args.synthetic_agent_attention_dim is not None and args.synthetic_agent_attention_heads is not None:
        if int(args.synthetic_agent_attention_dim) % int(args.synthetic_agent_attention_heads) != 0:
            raise ValueError(
                "--synthetic_agent_attention_dim must be divisible by "
                "--synthetic_agent_attention_heads."
            )
    if args.tabdiff_num_samples is not None and int(args.tabdiff_num_samples) <= 0:
        raise ValueError(f"--tabdiff_num_samples must be positive, got {args.tabdiff_num_samples}.")
    if int(args.dynamic_synthetic_refresh_epochs) <= 0:
        raise ValueError(
            "--dynamic_synthetic_refresh_epochs must be positive, "
            f"got {args.dynamic_synthetic_refresh_epochs}."
        )
    if int(args.dynamic_synthetic_warmup_epochs) < 0:
        raise ValueError(
            "--dynamic_synthetic_warmup_epochs must be non-negative, "
            f"got {args.dynamic_synthetic_warmup_epochs}."
        )
    if int(args.dynamic_synthetic_scarcity_bins) < 2:
        raise ValueError(
            "--dynamic_synthetic_scarcity_bins must be at least 2, "
            f"got {args.dynamic_synthetic_scarcity_bins}."
        )
    if int(args.num_working_condition_clusters) < 2:
        raise ValueError(
            "--num_working_condition_clusters must be at least 2, "
            f"got {args.num_working_condition_clusters}."
        )
    if float(args.lr) <= 0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if float(args.weight_decay) < 0:
        raise ValueError(f"--weight_decay must be non-negative, got {args.weight_decay}.")
    for name in ("dropout", "agent_dropout", "synthetic_agent_dropout"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}.")
    if int(args.early_stopping_patience) <= 0:
        raise ValueError(f"--early_stopping_patience must be positive, got {args.early_stopping_patience}.")
    if float(args.checkpoint_tail_mae_lambda) < 0:
        raise ValueError(
            "--checkpoint_tail_mae_lambda must be non-negative, "
            f"got {args.checkpoint_tail_mae_lambda}."
        )
    finetune_lrs = {
        "finetune_backbone_lr": args.finetune_backbone_lr,
        "finetune_head_lr": args.finetune_head_lr,
        "finetune_agent_lr": args.finetune_agent_lr,
        "finetune_quality_agent_lr": args.finetune_quality_agent_lr,
    }
    for name, value in finetune_lrs.items():
        if value is not None and float(value) <= 0:
            raise ValueError(f"--{name} must be positive, got {value}.")
    bounded = {
        "synthetic_confidence_threshold": args.synthetic_confidence_threshold,
        "synthetic_pretrain_confidence_threshold": args.synthetic_pretrain_confidence_threshold,
        "dynamic_synthetic_top_ratio": args.dynamic_synthetic_top_ratio,
        "dynamic_synthetic_weight_min": args.dynamic_synthetic_weight_min,
        "dynamic_synthetic_ema": args.dynamic_synthetic_ema,
        "dynamic_synthetic_reliability_floor": args.dynamic_synthetic_reliability_floor,
        "dynamic_synthetic_quota_strength": args.dynamic_synthetic_quota_strength,
    }
    for name, value in bounded.items():
        if value is None:
            continue
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}.")
    if args.dynamic_synthetic_top_ratio is not None and float(args.dynamic_synthetic_top_ratio) <= 0.0:
        raise ValueError(f"--dynamic_synthetic_top_ratio must be in (0, 1], got {args.dynamic_synthetic_top_ratio}.")
    nonnegative_dynamic = {
        "dynamic_synthetic_weight_max": args.dynamic_synthetic_weight_max,
        "dynamic_synthetic_error_weight": args.dynamic_synthetic_error_weight,
        "dynamic_synthetic_train_region_weight": args.dynamic_synthetic_train_region_weight,
        "dynamic_synthetic_scarcity_weight": args.dynamic_synthetic_scarcity_weight,
        "dynamic_synthetic_real_feedback_weight": args.dynamic_synthetic_real_feedback_weight,
        "dynamic_synthetic_quota_min": args.dynamic_synthetic_quota_min,
        "dynamic_synthetic_quota_max": args.dynamic_synthetic_quota_max,
        "dynamic_synthetic_process_power": args.dynamic_synthetic_process_power,
        "dynamic_synthetic_mechanism_power": args.dynamic_synthetic_mechanism_power,
        "dynamic_synthetic_train_tail_lambda": args.dynamic_synthetic_train_tail_lambda,
        "cluster_balance_lambda": args.cluster_balance_lambda,
        "reward_alpha_cluster": args.reward_alpha_cluster,
    }
    for name, value in nonnegative_dynamic.items():
        if value is not None and float(value) < 0.0:
            raise ValueError(f"--{name} must be non-negative, got {value}.")
    if float(args.dynamic_synthetic_weight_max) < float(args.dynamic_synthetic_weight_min):
        raise ValueError(
            "--dynamic_synthetic_weight_max must be >= --dynamic_synthetic_weight_min, "
            f"got {args.dynamic_synthetic_weight_max} < {args.dynamic_synthetic_weight_min}."
        )
    if float(args.dynamic_synthetic_quota_max) < float(args.dynamic_synthetic_quota_min):
        raise ValueError(
            "--dynamic_synthetic_quota_max must be >= --dynamic_synthetic_quota_min, "
            f"got {args.dynamic_synthetic_quota_max} < {args.dynamic_synthetic_quota_min}."
        )


def append_optional_cli_args(
    cmd: list[str],
    args: argparse.Namespace,
    specs: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Append non-empty experiment values to a child pipeline command."""
    for attr_name, flag in (specs or ()):
        value = getattr(args, attr_name, None)
        if isinstance(value, bool):
            if flag in BOOL_VALUE_FLAGS:
                cmd.extend([flag, str(value)])
            elif value:
                cmd.append(flag)
        elif value is not None:
            cmd.extend([flag, str(value)])
    return cmd


PROGRESS_PATTERNS = [
    re.compile(r"(Supervised finetune|Synthetic pretrain|Semi-supervised finetune).*?(\d+)/(\d+)"),
    re.compile(r"(TabDiff train|TabDiff sample|TabDiff generation).*?(\d+)/(\d+)"),
    re.compile(r"Epoch\s+(\d+)/(\d+)"),
]


def _phase_fraction(record: str) -> tuple[str, int, int] | None:
    text = record.replace("\r", "").replace("\n", "")
    for pattern in PROGRESS_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if len(match.groups()) == 3:
            return match.group(1), int(match.group(2)), int(match.group(3))
        return "TabDiff train", int(match.group(1)), int(match.group(2))
    return None


def _set_phase_bar_progress(
    progress_bar: tqdm,
    child_desc: str | None = None,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if child_desc is not None and current is not None and total:
        previous_desc = getattr(progress_bar, "_capl_child_desc", None)
        if previous_desc != child_desc or progress_bar.total != total:
            progress_bar.total = total
            progress_bar.n = 0
            progress_bar._capl_child_desc = child_desc
        progress_bar.n = min(max(current, 0), total)
        progress_bar.set_postfix_str(f"{child_desc} {current}/{total}", refresh=False)
    else:
        progress_bar.set_postfix_str("starting", refresh=False)
    progress_bar.refresh()


def _progress_bar(desc: str) -> tqdm:
    return tqdm(
        total=1,
        desc=desc,
        unit="step",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=False,
    )


def mark_phase_done(progress_bar: tqdm, message: str) -> None:
    if progress_bar.total is None:
        progress_bar.total = 1
    progress_bar.n = progress_bar.total
    progress_bar.set_postfix_str("done", refresh=False)
    progress_bar.refresh()
    progress_bar.close()
    tqdm.write(message, file=sys.stdout)


def run_command(
    cmd: list[str],
    cwd: Path,
    run_dir: Path,
    phase: str,
    progress_bar: tqdm,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command_path = run_dir / "command.txt"
    command_path.write_text(" ".join(cmd), encoding="utf-8")
    _set_phase_bar_progress(progress_bar)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        progress_lock = threading.Lock()

        def update_single_progress(record: str) -> None:
            parsed = _phase_fraction(record)
            if parsed is None:
                return
            child_desc, current, total = parsed
            with progress_lock:
                _set_phase_bar_progress(progress_bar, child_desc, current, total)

        def tee_stream(stream, log_file) -> None:
            buffer = []
            try:
                while True:
                    chunk = stream.read(1)
                    if not chunk:
                        break
                    log_file.write(chunk)
                    log_file.flush()
                    buffer.append(chunk)
                    if chunk in {"\r", "\n"}:
                        record = "".join(buffer)
                        buffer.clear()
                        update_single_progress(record)
                if buffer:
                    record = "".join(buffer)
                    update_single_progress(record)
            finally:
                stream.close()

        threads = [
            threading.Thread(target=tee_stream, args=(process.stdout, stdout), daemon=True),
            threading.Thread(target=tee_stream, args=(process.stderr, stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        return_code = process.wait()
        for thread in threads:
            thread.join()
        if return_code != 0:
            progress_bar.write(f"[failed:{phase}] stdout={stdout_path} stderr={stderr_path}")
            raise subprocess.CalledProcessError(return_code, cmd)
    if progress_bar.total is None:
        progress_bar.total = 1
    progress_bar.n = progress_bar.total
    progress_bar.set_postfix_str("done", refresh=False)
    progress_bar.refresh()


def latest_checkpoint(stage_dir: Path) -> Path:
    candidates = sorted(
        stage_dir.glob("**/checkpoints/best_model.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best_model.pth found under {stage_dir}")
    return candidates[0]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_template_path(template: str, seed: int) -> str:
    if "{seed}" in template:
        return template.format(seed=seed)
    path = Path(template)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_seed_{seed}{path.suffix}"))
    return str(path / f"seed_{seed}")


def build_tabdiff_generation_command(
    args: argparse.Namespace,
    tabdiff_num_samples: int,
    *,
    scientific_code_hash: str | None = None,
    generation_protocol_hash: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "pipeline.py",
        "--mode",
        "generate_synthetic_tabdiff",
        "--config",
        args.config,
        "--experiment_name",
        "tabdiff_generate_capl",
        "--data_path",
        args.data_path,
        "--label_col",
        args.label_col,
        "--synthetic_data_path",
        args.synthetic_data_path,
        "--seed",
        str(args.generation_seed),
        "--split_seed",
        str(args.split_seed),
        "--split_method",
        args.split_method,
        "--generation_seed",
        str(args.generation_seed),
        "--tabdiff_num_samples",
        str(tabdiff_num_samples),
    ]
    if args.tabdiff_gpu is not None:
        cmd.extend(["--tabdiff_gpu", str(args.tabdiff_gpu)])
    if (scientific_code_hash is None) != (generation_protocol_hash is None):
        raise ValueError("Scientific code and generation protocol hashes must be provided together.")
    if scientific_code_hash is not None and generation_protocol_hash is not None:
        cmd.extend(
            [
                "--scientific_code_sha256",
                _sha256_value(scientific_code_hash, "scientific code"),
                "--generation_protocol_sha256",
                _sha256_value(generation_protocol_hash, "generation protocol"),
            ]
        )
    return cmd


def build_main_train_command(
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    synthetic_path: str,
    tabdiff_num_samples: int,
    experiment_name: str | None = None,
    scientific_code_hash: str | None = None,
    generation_protocol_hash: str | None = None,
) -> list[str]:
    resolved_experiment_name = experiment_name or args.main_experiment_name or SUPERVISED_MAIN_EXPERIMENT_NAME
    cmd = [
        sys.executable,
        "pipeline.py",
        "--mode",
        SUPERVISED_MAIN_TRAIN_MODE,
        "--config",
        args.config,
        "--experiment_name",
        resolved_experiment_name,
        "--output_dir",
        str(run_dir),
        "--data_path",
        args.data_path,
        "--label_col",
        args.label_col,
        "--synthetic_data_path",
        synthetic_path,
        "--seed",
        str(seed),
        "--split_seed",
        str(args.split_seed),
        "--generation_seed",
        str(args.generation_seed),
        "--epochs",
        str(args.epochs),
        "--tabdiff_num_samples",
        str(tabdiff_num_samples),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--split_method",
        args.split_method,
        "--no_el",
        "--no_laplace",
    ]
    cmd.extend(["--synthetic_pretrain_epochs", str(args.synthetic_pretrain_epochs)])
    cmd = append_optional_cli_args(cmd, args, MAIN_TRAIN_ARG_SPECS)
    if not args.use_layerwise_finetune_lr:
        cmd.append("--no_layerwise_finetune_lr")
    if not args.freeze_finetune_backbone:
        cmd.append("--no_freeze_finetune_backbone")
    if args.use_mr_lora:
        mr_lora_value_specs = [spec for spec in MR_LORA_ARG_SPECS if spec[0] not in {"use_mr_lora", "mr_lora_train_output_head"}]
        cmd.extend(
            [
                "--use_mr_lora",
            ]
        )
        cmd = append_optional_cli_args(cmd, args, mr_lora_value_specs)
        if args.mr_lora_train_output_head:
            cmd.append("--mr_lora_train_output_head")
    if (scientific_code_hash is None) != (generation_protocol_hash is None):
        raise ValueError("Scientific code and generation protocol hashes must be provided together.")
    if scientific_code_hash is not None and generation_protocol_hash is not None:
        cmd.extend(
            [
                "--scientific_code_sha256",
                _sha256_value(scientific_code_hash, "scientific code"),
                "--generation_protocol_sha256",
                _sha256_value(generation_protocol_hash, "generation protocol"),
            ]
        )
    return cmd


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def metrics_output_snapshot(run_dir: Path) -> dict[str, tuple[int, int, str]]:
    """Fingerprint every pre-existing metrics artifact under one seed directory."""
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in sorted(run_dir.glob("**/results/metrics.json")):
        resolved = path.resolve()
        artifact = capture_file_snapshot(resolved)
        snapshot[str(resolved)] = (
            int(artifact.signature[3]),
            int(artifact.signature[2]),
            artifact.sha256,
        )
    return snapshot


def _sha256_value(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"Invalid SHA-256 value for {field}.")
    return text.lower()


def effective_protocol_snapshot(
    config_values: dict[str, object],
    args: argparse.Namespace,
    tabdiff_num_samples: int,
    cross_run_statistics: dict[str, object],
    runtime_config_values: dict[str, object] | None = None,
    producer_protocol_hash: str | None = None,
    producer_protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    runner = dict(vars(args))
    runner["seeds"] = [int(seed) for seed in args.seeds]
    runner["tabdiff_num_samples"] = int(tabdiff_num_samples)
    runner["main_experiment_name"] = args.main_experiment_name or SUPERVISED_MAIN_EXPERIMENT_NAME
    return {
        "format": "mtam_hg_effective_protocol_v1",
        "yaml": config_values,
        "runner": runner,
        "runtime": {
            "mode": SUPERVISED_MAIN_TRAIN_MODE,
            "model": MAIN_EXPERIMENT_MODEL,
            "use_el_as_input": False,
            "use_laplace": False,
            "shared_split": True,
            "config": runtime_config_values or {},
        },
        "cbtg_producer_protocol_sha256": (
            _sha256_value(producer_protocol_hash, "CBTG producer protocol")
            if producer_protocol_hash is not None
            else None
        ),
        "cbtg_producer_protocol": producer_protocol,
        "cbtg_cross_run_validation_std": {
            "sha256": _sha256_value(cross_run_statistics.get("sha256"), "cross-run statistics"),
            "payload": cross_run_statistics.get("payload"),
        },
    }


def effective_protocol_sha256(snapshot: dict[str, object]) -> str:
    return canonical_sha256(snapshot)


def scientific_producer_protocol_snapshot(
    config_values: dict[str, object],
    args: argparse.Namespace,
    tabdiff_num_samples: int,
    runtime_config_values: dict[str, object],
    code_sha256: str,
) -> dict[str, object]:
    yaml_values = {
        key: value
        for key, value in config_values.items()
        if key not in OPERATIONAL_YAML_KEYS
    }
    runner_values = {
        name: getattr(args, name)
        for name in sorted(CONFIRMATORY_LOCKED_ARGS)
    }
    runner_values.update(
        {
            "split_seed": int(args.split_seed),
            "split_method": str(args.split_method),
            "generation_seed": int(args.generation_seed),
            "label_col": str(args.label_col),
            "tabdiff_num_samples": int(tabdiff_num_samples),
            "main_experiment_name": args.main_experiment_name or SUPERVISED_MAIN_EXPERIMENT_NAME,
        }
    )
    runtime_values = {
        key: value
        for key, value in runtime_config_values.items()
        if key not in OPERATIONAL_RUNTIME_KEYS
    }
    return {
        "format": "mtam_hg_cbtg_producer_protocol_v1",
        "yaml": yaml_values,
        "runner": runner_values,
        "runtime": runtime_values,
        "validation_metrics": list(CBTG_VALIDATION_METRICS),
        "validation_std_ddof": 1,
        "code_sha256": _sha256_value(code_sha256, "scientific code"),
    }


def scientific_producer_protocol_sha256(snapshot: dict[str, object]) -> str:
    return canonical_sha256(snapshot)


def require_scientific_code_sha256(expected_sha256: str) -> None:
    if scientific_code_sha256() != expected_sha256:
        raise RuntimeError("Scientific source code changed during the confirmatory run.")


def _protocol_value(value: object) -> object:
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(value)
    if isinstance(value, dict):
        return {str(key): _protocol_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_protocol_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError


def runtime_config_snapshot(runtime_config: object) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    for name in sorted(name for name in dir(runtime_config) if name.isupper()):
        if name == "CONFIG_SHA256":
            continue
        try:
            snapshot[name] = _protocol_value(getattr(runtime_config, name))
        except TypeError:
            continue
    return snapshot


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def bind_metrics_to_effective_protocol(
    metrics_path: Path,
    expected_seed: int,
    expected_integrity: dict[str, object],
) -> str:
    snapshot = capture_file_snapshot(metrics_path, include_content=True)
    assert snapshot.content is not None
    metrics = json.loads(snapshot.content.decode("utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics root must be an object: {metrics_path}")
    if metrics.get("Seed") != expected_seed:
        raise ValueError(f"Metrics seed mismatch: {metrics_path}")
    binding_fields = {"Effective_Protocol_SHA256"}
    mismatched = [
        field
        for field in METRICS_INTEGRITY_FIELDS
        if field not in binding_fields and metrics.get(field) != expected_integrity.get(field)
    ]
    if mismatched:
        raise ValueError(f"Metrics protocol integrity mismatch at {metrics_path}: {', '.join(mismatched)}.")
    for field in binding_fields:
        existing = metrics.get(field)
        if existing is not None and existing != expected_integrity.get(field):
            raise ValueError(f"Metrics protocol integrity mismatch at {metrics_path}: {field}.")
        metrics[field] = expected_integrity[field]
    assert_file_snapshot_current(snapshot, "metrics artifact")
    _write_json_atomic(metrics_path, metrics)
    return file_sha256(metrics_path)


def require_file_sha256(path: Path, expected_sha256: str, label: str) -> None:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} changed during the confirmatory run: {path}")


def require_loaded_config_sha256(expected_sha256: str) -> None:
    import config as runtime_config

    if runtime_config.CONFIG_SHA256 != expected_sha256:
        raise RuntimeError("Experiment config changed while loading overrides.")


def validate_declared_protocol(config_values: dict[str, object]) -> None:
    if canonical_sha256(config_values) != CONFIRMATORY_YAML_CANONICAL_SHA256:
        raise ValueError("YAML configuration does not match the locked confirmatory protocol.")


def metrics_integrity_fields(
    provenance: dict[str, object],
    split_seed: int,
    split_method: str,
    generation_seed: int,
    config_sha256: str,
    effective_protocol_hash: str,
    cross_run_statistics_sha256: str,
) -> dict[str, object]:
    return {
        "Split_Seed": int(split_seed),
        "Split_Method": str(split_method),
        "Combined_Split_SHA256": _sha256_value(
            provenance.get("combined_split_sha256"), "combined_split_sha256"
        ),
        "Source_Data_SHA256": _sha256_value(provenance.get("source_sha256"), "source_sha256"),
        "Synthetic_SHA256": _sha256_value(provenance.get("synthetic_sha256"), "synthetic_sha256"),
        "Generation_Seed": int(generation_seed),
        "Config_SHA256": _sha256_value(config_sha256, "config_sha256"),
        "Effective_Protocol_SHA256": _sha256_value(
            effective_protocol_hash, "effective_protocol_sha256"
        ),
        "CBTG_Cross_Run_Validation_STD_SHA256": _sha256_value(
            cross_run_statistics_sha256, "cross_run_statistics_sha256"
        ),
        "Synthetic_Provenance_SHA256": _sha256_value(
            provenance.get("provenance_sha256"), "synthetic_provenance_sha256"
        ),
    }


def validate_runner_integrity(
    synthetic_path: Path,
    data_path: Path,
    split_seed: int,
    split_method: str,
    generation_seed: int,
    config_sha256: str,
    effective_protocol_hash: str,
    producer_protocol_hash: str,
    cross_run_statistics_path: Path,
    cross_run_statistics_sha256: str,
    *,
    label_col: str,
    scientific_code_hash: str,
    expected_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    provenance = validate_synthetic_provenance_for_runner(
        synthetic_path,
        data_path,
        split_seed,
        split_method,
        generation_seed,
        label_col=label_col,
        use_el_as_input=False,
        validate_current_generation_config=True,
        expected_scientific_code_sha256=scientific_code_hash,
        expected_generation_protocol_sha256=producer_protocol_hash,
    )
    cross_run_statistics = validate_cbtg_cross_run_validation_std(
        cross_run_statistics_path,
        expected_split_sha256=str(provenance["combined_split_sha256"]),
        expected_source_data_sha256=str(provenance["source_sha256"]),
        expected_config_sha256=config_sha256,
        expected_producer_protocol_sha256=producer_protocol_hash,
    )
    if cross_run_statistics["sha256"] != cross_run_statistics_sha256:
        raise RuntimeError("CBTG cross-run statistics changed during execution.")
    current_fields = metrics_integrity_fields(
        provenance,
        split_seed,
        split_method,
        generation_seed,
        config_sha256,
        effective_protocol_hash,
        cross_run_statistics_sha256,
    )
    if expected_fields is not None:
        mismatched = [field for field in METRICS_INTEGRITY_FIELDS if current_fields.get(field) != expected_fields.get(field)]
        if mismatched:
            raise RuntimeError(f"Run protocol integrity changed during execution: {', '.join(mismatched)}.")
    return current_fields


def load_seed_metrics(
    metrics_path: Path,
    expected_seed: int,
    expected_integrity: dict[str, object],
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Load and validate one exact metrics artifact from the current seed phase."""
    metrics_path = metrics_path.resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics artifact was not found for requested seed {expected_seed}: {metrics_path}")
    snapshot = capture_file_snapshot(metrics_path, include_content=True)
    assert snapshot.content is not None
    if expected_sha256 is not None and snapshot.sha256 != expected_sha256:
        raise RuntimeError(f"Metrics artifact changed after validation: {metrics_path}")
    metrics = json.loads(snapshot.content.decode("utf-8"))
    is_confirmatory = (
        isinstance(metrics, dict)
        and metrics.get("Model") == MAIN_EXPERIMENT_MODEL
        and metrics.get("Experiment_Group") in MAIN_EXPERIMENT_NAMES
        and metrics.get("Seed") == expected_seed
        and all(_is_finite_number(metrics.get(metric)) for metric in CONFIRMATORY_METRICS)
    )
    if not is_confirmatory:
        raise ValueError(
            "No confirmatory MTAM-HG metrics matched "
            f"seed={expected_seed}, model={MAIN_EXPERIMENT_MODEL!r}, "
            f"experiment={sorted(MAIN_EXPERIMENT_NAMES)}, and finite metrics={CONFIRMATORY_METRICS} "
            f"at {metrics_path}."
        )
    mismatched = [
        field
        for field in METRICS_INTEGRITY_FIELDS
        if metrics.get(field) != expected_integrity.get(field)
    ]
    if mismatched:
        raise ValueError(f"Metrics protocol integrity mismatch at {metrics_path}: {', '.join(mismatched)}.")
    assert_file_snapshot_current(snapshot, "metrics artifact")
    return {
        "metrics_path": str(metrics_path),
        "metrics_sha256": snapshot.sha256,
        "metrics": metrics,
    }


def require_fresh_seed_metrics(
    run_dir: Path,
    previous_snapshot: dict[str, tuple[int, int, str]],
    expected_seed: int,
    expected_integrity: dict[str, object],
) -> tuple[Path, str]:
    """Return the sole metrics file created or changed by the current phase."""
    current_snapshot = metrics_output_snapshot(run_dir)
    fresh_paths = [Path(path) for path, signature in current_snapshot.items() if previous_snapshot.get(path) != signature]
    if len(fresh_paths) != 1:
        raise RuntimeError(
            "Each confirmatory seed phase must produce exactly one fresh metrics.json; "
            f"seed {expected_seed} produced {len(fresh_paths)} under {run_dir}."
        )
    metrics_path = fresh_paths[0]
    bound_sha256 = bind_metrics_to_effective_protocol(
        metrics_path,
        expected_seed,
        expected_integrity,
    )
    loaded = load_seed_metrics(
        metrics_path,
        expected_seed=expected_seed,
        expected_integrity=expected_integrity,
        expected_sha256=bound_sha256,
    )
    return metrics_path, str(loaded["metrics_sha256"])


def _summary_seed_entry(
    seed: int,
    run_dir: Path,
    dry_run: bool,
    *,
    metrics_path: Path | None = None,
    metrics_sha256: str | None = None,
    expected_integrity: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    if not dry_run and (metrics_path is None or metrics_sha256 is None or expected_integrity is None):
        raise ValueError(f"Current-run metrics and integrity fields are required for confirmatory seed {seed}.")
    if dry_run:
        loaded = {"metrics_path": "", "metrics_sha256": "", "metrics": {}}
    else:
        assert metrics_path is not None
        assert metrics_sha256 is not None
        assert expected_integrity is not None
        loaded = load_seed_metrics(
            metrics_path,
            expected_seed=seed,
            expected_integrity=expected_integrity,
            expected_sha256=metrics_sha256,
        )
    metrics = loaded["metrics"]
    tail_mae = None
    if isinstance(metrics, dict):
        tail_mae = metrics.get("TAIL_MAE", metrics.get("Tail_MAE"))
    entry: dict[str, object] = {
        "seed": int(seed),
        "run_dir": str(run_dir),
        "metrics_path": loaded["metrics_path"],
        "metrics_sha256": loaded["metrics_sha256"],
        "metrics_seed": metrics.get("Seed") if isinstance(metrics, dict) else None,
        "RMSE": metrics.get("RMSE") if isinstance(metrics, dict) else None,
        "MAE": metrics.get("MAE") if isinstance(metrics, dict) else None,
        "MAPE": metrics.get("MAPE") if isinstance(metrics, dict) else None,
        "R2": metrics.get("R2") if isinstance(metrics, dict) else None,
        "TAIL_MAE": tail_mae,
        "Best_Epoch": metrics.get("Best_Epoch") if isinstance(metrics, dict) else None,
        "Epochs_Run": metrics.get("Epochs_Run") if isinstance(metrics, dict) else None,
    }
    if extra:
        entry.update(extra)
    return entry


def write_runner_summary(
    output_root: Path,
    phases: list[dict[str, object]],
    seed_run_dirs: dict[int, Path],
    dry_run: bool,
    use_dynamic_synthetic_agent: bool = False,
    split_seed: int = DEFAULT_SPLIT_SEED,
    split_method: str = DEFAULT_SPLIT_METHOD,
    generation_seed: int = DEFAULT_GENERATION_SEED,
    seed_metrics_paths: dict[int, Path] | None = None,
    seed_metrics_sha256: dict[int, str] | None = None,
    integrity_fields: dict[str, object] | None = None,
    effective_protocol: dict[str, object] | None = None,
    effective_protocol_hash: str | None = None,
) -> Path:
    if int(split_seed) != DEFAULT_SPLIT_SEED or split_method != DEFAULT_SPLIT_METHOD:
        raise ValueError("Summary protocol does not match the locked confirmatory split.")
    if int(generation_seed) != DEFAULT_GENERATION_SEED:
        raise ValueError("Summary protocol does not match the locked confirmatory generation seed.")
    ordered_seeds = list(seed_run_dirs)
    if not dry_run and len(ordered_seeds) != 10:
        raise ValueError("A non-dry-run confirmatory summary requires ten distinct model seeds.")
    if not dry_run and (seed_metrics_paths is None or list(seed_metrics_paths) != ordered_seeds):
        received = [] if seed_metrics_paths is None else list(seed_metrics_paths)
        raise ValueError(
            "A non-dry-run confirmatory summary requires one exact current-run metrics path for each seed "
            f"{ordered_seeds}; got {received}."
        )
    if not dry_run and (seed_metrics_sha256 is None or list(seed_metrics_sha256) != ordered_seeds):
        received = [] if seed_metrics_sha256 is None else list(seed_metrics_sha256)
        raise ValueError(
            "A non-dry-run confirmatory summary requires one locked metrics hash for each seed "
            f"{ordered_seeds}; got {received}."
        )
    if not dry_run and integrity_fields is None:
        raise ValueError("A non-dry-run confirmatory summary requires validated integrity fields.")
    if not dry_run and (effective_protocol is None or effective_protocol_hash is None):
        raise ValueError("A non-dry-run confirmatory summary requires the effective protocol.")
    if effective_protocol is not None:
        calculated_hash = effective_protocol_sha256(effective_protocol)
        if calculated_hash != effective_protocol_hash:
            raise ValueError("Effective protocol hash mismatch.")
    if integrity_fields is not None:
        missing_fields = [field for field in METRICS_INTEGRITY_FIELDS if field not in integrity_fields]
        if missing_fields:
            raise ValueError(f"Missing summary integrity fields: {', '.join(missing_fields)}.")
        normalized_integrity = metrics_integrity_fields(
            {
                "combined_split_sha256": integrity_fields["Combined_Split_SHA256"],
                "source_sha256": integrity_fields["Source_Data_SHA256"],
                "synthetic_sha256": integrity_fields["Synthetic_SHA256"],
                "provenance_sha256": integrity_fields["Synthetic_Provenance_SHA256"],
            },
            split_seed,
            split_method,
            generation_seed,
            str(integrity_fields["Config_SHA256"]),
            str(integrity_fields["Effective_Protocol_SHA256"]),
            str(integrity_fields["CBTG_Cross_Run_Validation_STD_SHA256"]),
        )
        if any(integrity_fields[field] != normalized_integrity[field] for field in METRICS_INTEGRITY_FIELDS):
            raise ValueError("Summary integrity fields do not match the locked run protocol.")
        if effective_protocol_hash != integrity_fields["Effective_Protocol_SHA256"]:
            raise ValueError("Summary effective protocol does not match the seed metrics.")
        integrity_fields = normalized_integrity
    seeds = [
        _summary_seed_entry(
            seed,
            seed_run_dirs[seed],
            dry_run,
            metrics_path=None if seed_metrics_paths is None else seed_metrics_paths.get(seed),
            metrics_sha256=None if seed_metrics_sha256 is None else seed_metrics_sha256.get(seed),
            expected_integrity=integrity_fields,
        )
        for seed in ordered_seeds
    ]
    aggregate: dict[str, dict[str, float | int]] = {}
    for metric in PAPER_METRICS:
        values = [
            float(entry[metric])
            for entry in seeds
            if isinstance(entry.get(metric), (int, float)) and math.isfinite(float(entry[metric]))
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        std = (
            math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
            if len(values) > 1
            else 0.0
        )
        aggregate[metric] = {"mean": mean, "std": std, "n": len(values), "ddof": 1}
    if not dry_run:
        for metric in CONFIRMATORY_METRICS:
            if aggregate.get(metric, {}).get("n") != 10:
                raise ValueError(
                    f"Confirmatory metric {metric} must contain ten finite values; got {aggregate.get(metric)}."
                )
    pipeline = [
        "mechanism-aware TabDiff generation when synthetic file is missing",
        "mechanism-aware TabDiff rows are used directly as the synthetic pretraining table",
        (
            "MoE-IPOHGN pretraining on synthetic samples with the K-means cluster-balanced CBTG-Agent policy"
            if use_dynamic_synthetic_agent
            else "MoE-IPOHGN pretraining on synthetic samples"
        ),
        "final supervised MTAM-HG MR-LoRA calibration with validation checkpoint selection",
        "test-set metric export",
    ]
    summary = {
        "dry_run": bool(dry_run),
        "main_train_mode": SUPERVISED_MAIN_TRAIN_MODE,
        "evaluation_protocol": "confirmatory_dry_run" if dry_run else "confirmatory_fixed_seed",
        "model_seeds": ordered_seeds,
        "split_seed": int(split_seed),
        "split_method": split_method,
        "generation_seed": int(generation_seed),
        "effective_protocol_sha256": effective_protocol_hash,
        "effective_protocol": effective_protocol,
        "confirmatory_protocol": {
            "validated": bool(not dry_run and integrity_fields is not None),
            "model_seeds": ordered_seeds,
            "split_seed": int(split_seed),
            "split_method": split_method,
            "generation_seed": int(generation_seed),
            "shared_main_split": bool(not dry_run and integrity_fields is not None),
            "metrics_integrity": (
                {}
                if integrity_fields is None
                else {field: integrity_fields[field] for field in METRICS_INTEGRITY_FIELDS}
            ),
        },
        "use_dynamic_synthetic_agent": bool(use_dynamic_synthetic_agent),
        "dynamic_synthetic_contract": (
            "When enabled, the Agent receives dynamic main-model feedback features during synthetic pretraining "
            "including K-means working-condition cluster difficulty, and its feedback-conditioned policy score drives synthetic-sample loss weights and expected sampling quota; "
            "the predictor is calibrated on real training rows and the Agent is calibrated from real validation feedback."
        ),
        "phases": phases,
        "seeds": seeds,
        "aggregate_mean_std": aggregate,
        "pipeline": pipeline,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / RUNNER_SUMMARY_NAME
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    aggregate_path = output_root / "metrics_mean_std.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n", "ddof"])
        writer.writeheader()
        for metric, values in aggregate.items():
            writer.writerow({"metric": metric, **values})
    return summary_path


def dry_run_command(cmd: list[str], progress_bar: tqdm) -> None:
    progress_bar.total = 1
    progress_bar.n = 1
    progress_bar.set_postfix_str("dry-run", refresh=False)
    progress_bar.refresh()


def run_phase(phase: ExperimentPhase, dry_run: bool) -> None:
    phase_bar = _progress_bar(phase.name)
    try:
        if dry_run:
            dry_run_command(phase.command, phase_bar)
        else:
            run_command(phase.command, PROJECT_ROOT, phase.run_dir, phase=phase.name, progress_bar=phase_bar)
        mark_phase_done(phase_bar, f"[done:{phase.name}]")
    except Exception:
        phase_bar.close()
        raise


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    config_path = PROJECT_ROOT / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config was not found: {config_path}")
    from config_loader import load_yaml_config_text

    config_snapshot = capture_file_snapshot(config_path, include_content=True)
    assert config_snapshot.content is not None
    config_values = load_yaml_config_text(config_snapshot.content.decode("utf-8"))
    assert_file_snapshot_current(config_snapshot, "Experiment config")
    config_sha256 = config_snapshot.sha256
    validate_declared_protocol(config_values)
    tabdiff_num_samples = int(
        args.tabdiff_num_samples
        if args.tabdiff_num_samples is not None
        else config_values.get("tabdiff_num_samples", DEFAULT_TABDIFF_NUM_SAMPLES)
    )
    import config as runtime_config
    from pipeline import load_config_overrides

    load_config_overrides(str(config_path))
    require_loaded_config_sha256(config_sha256)
    runtime_config.SPLIT_SEED = int(args.split_seed)
    runtime_config.SPLIT_METHOD = str(args.split_method)
    runtime_config.TABDIFF_GENERATION_SEED = int(args.generation_seed)
    runtime_config.TABDIFF_NUM_SAMPLES = tabdiff_num_samples
    runtime_config.LABEL_COL = args.label_col
    runtime_config.USE_EL_AS_INPUT = False
    runtime_config.USE_LAPLACE = False
    runtime_config.CBTG_CROSS_RUN_VALIDATION_STD_PATH = args.cbtg_cross_run_validation_std_path
    runtime_config.CONFIG_SHA256 = config_sha256
    runtime_values = runtime_config_snapshot(runtime_config)
    code_sha256 = scientific_code_sha256()
    producer_protocol = scientific_producer_protocol_snapshot(
        config_values,
        args,
        tabdiff_num_samples,
        runtime_values,
        code_sha256,
    )
    producer_protocol_hash = scientific_producer_protocol_sha256(producer_protocol)
    cross_run_statistics_path = project_path(args.cbtg_cross_run_validation_std_path)
    try:
        cross_run_statistics = validate_cbtg_cross_run_validation_std(
            cross_run_statistics_path,
            expected_config_sha256=config_sha256,
            expected_producer_protocol_sha256=producer_protocol_hash,
        )
    except ProtocolIntegrityError:
        if not args.dry_run:
            raise
        cross_run_statistics = {
            "path": str(cross_run_statistics_path),
            "sha256": "0" * 64,
            "payload": {"status": "required_for_confirmatory_execution"},
        }
    cross_run_statistics_sha256 = str(cross_run_statistics["sha256"])
    protocol_snapshot = effective_protocol_snapshot(
        config_values,
        args,
        tabdiff_num_samples,
        cross_run_statistics,
        runtime_values,
        producer_protocol_hash,
        producer_protocol,
    )
    protocol_hash = effective_protocol_sha256(protocol_snapshot)
    synthetic_path = PROJECT_ROOT / args.synthetic_data_path
    if "smoke" in synthetic_path.name.lower():
        raise ValueError("The released experiment must not use smoke synthetic data.")

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    phase_records: list[dict[str, object]] = []
    seed_run_dirs: dict[int, Path] = {}
    seed_metrics_paths: dict[int, Path] = {}
    seed_metrics_sha256: dict[int, str] = {}
    initial_integrity: dict[str, object] | None = None
    print("[fairness]", flush=True)
    print(f"  seeds={args.seeds}", flush=True)
    print(f"  main_train_mode={SUPERVISED_MAIN_TRAIN_MODE}", flush=True)
    print(f"  real_finetune_epochs={args.epochs}", flush=True)
    print(f"  synthetic_pretrain_epochs={args.synthetic_pretrain_epochs}", flush=True)
    if args.synthetic_agent_epochs is not None:
        print(f"  synthetic_agent_epochs={args.synthetic_agent_epochs}", flush=True)
    if args.synthetic_agent_lr is not None:
        print(f"  synthetic_agent_lr={args.synthetic_agent_lr}", flush=True)
    if args.synthetic_agent_attention_dim is not None:
        print(
            f"  synthetic_agent_attention_dim={args.synthetic_agent_attention_dim}, "
            f"heads={args.synthetic_agent_attention_heads}",
            flush=True,
        )
    if args.synthetic_confidence_threshold is not None:
        print(f"  synthetic_confidence_threshold={args.synthetic_confidence_threshold}", flush=True)
    print(f"  tabdiff_num_samples={tabdiff_num_samples}", flush=True)
    print(f"  batch_size={args.batch_size}, lr={args.lr}", flush=True)
    print(f"  weight_decay={args.weight_decay}", flush=True)
    print(
        f"  dropout={args.dropout}, agent_dropout={args.agent_dropout}, "
        f"synthetic_agent_dropout={args.synthetic_agent_dropout}",
        flush=True,
    )
    print(f"  early_stopping_patience={args.early_stopping_patience}", flush=True)
    print(
        "  main_checkpoint_policy="
        f"checkpoint={args.checkpoint_selection_metric}+{args.checkpoint_tail_mae_lambda}*TAIL_MAE",
        flush=True,
    )
    print(
        "  "
        f"use_layerwise_finetune_lr={args.use_layerwise_finetune_lr}, "
        f"freeze_finetune_backbone={args.freeze_finetune_backbone}, "
        f"finetune_backbone_lr={args.finetune_backbone_lr}, "
        f"finetune_head_lr={args.finetune_head_lr}, "
        f"finetune_agent_lr={args.finetune_agent_lr}",
        flush=True,
    )
    print(
        f"  split_seed={args.split_seed}, split_method={args.split_method}, no_laplace=True, "
        "use_el=False",
        flush=True,
    )
    print("  synthetic_governance=direct_tabdiff_dynamic_sampler", flush=True)
    print(
        f"  dynamic_synthetic_agent={args.use_dynamic_synthetic_agent}, "
        f"refresh_epochs={args.dynamic_synthetic_refresh_epochs}, "
        f"warmup_epochs={args.dynamic_synthetic_warmup_epochs}, "
        f"sampler={args.dynamic_synthetic_use_sampler}, "
        f"top_ratio={args.dynamic_synthetic_top_ratio}, "
        f"loss_weight={args.dynamic_synthetic_use_loss_weight}, "
        f"real_feedback_weight={args.dynamic_synthetic_real_feedback_weight}, "
        f"quota_strength={args.dynamic_synthetic_quota_strength}",
        flush=True,
    )
    print(
        f"  cluster_balance_reward={args.use_cluster_balance_reward}, "
        f"working_condition_clusters={args.num_working_condition_clusters}, "
        f"cluster_lambda={args.cluster_balance_lambda}, "
        f"reward_alpha_cluster={args.reward_alpha_cluster}",
        flush=True,
    )
    print(f"  tabdiff_generation={not args.skip_tabdiff_generation}", flush=True)
    print(f"  generation_seed={args.generation_seed}", flush=True)
    require_file_sha256(config_path, config_sha256, "Experiment config")
    require_scientific_code_sha256(code_sha256)

    needs_generation = not synthetic_path.exists()
    if synthetic_path.exists() and not args.dry_run:
        try:
            initial_integrity = validate_runner_integrity(
                synthetic_path,
                project_path(args.data_path),
                args.split_seed,
                args.split_method,
                args.generation_seed,
                config_sha256,
                protocol_hash,
                producer_protocol_hash,
                cross_run_statistics_path,
                cross_run_statistics_sha256,
                label_col=args.label_col,
                scientific_code_hash=code_sha256,
            )
        except SyntheticProvenanceError:
            if args.skip_tabdiff_generation:
                raise
            needs_generation = True

    if needs_generation and not args.skip_tabdiff_generation:
        phase = ExperimentPhase(
            name="tabdiff-generate",
            command=build_tabdiff_generation_command(
                args,
                tabdiff_num_samples,
                scientific_code_hash=code_sha256,
                generation_protocol_hash=producer_protocol_hash,
            ),
            run_dir=output_root / "tabdiff_generation",
            output_path=synthetic_path,
        )
        phase_records.append({"name": phase.name, "command": phase.command, "run_dir": str(phase.run_dir)})
        run_phase(phase, args.dry_run)
        if not args.dry_run:
            require_file_sha256(config_path, config_sha256, "Experiment config")
            require_scientific_code_sha256(code_sha256)
            initial_integrity = validate_runner_integrity(
                synthetic_path,
                project_path(args.data_path),
                args.split_seed,
                args.split_method,
                args.generation_seed,
                config_sha256,
                protocol_hash,
                producer_protocol_hash,
                cross_run_statistics_path,
                cross_run_statistics_sha256,
                label_col=args.label_col,
                scientific_code_hash=code_sha256,
            )
    elif needs_generation and args.skip_tabdiff_generation:
        if args.dry_run:
            tqdm.write(f"[dry-run:skip-tabdiff-generate] missing synthetic file: {synthetic_path}", file=sys.stdout)
        else:
            raise FileNotFoundError(
                f"Synthetic data file is missing and --skip_tabdiff_generation was set: {synthetic_path}"
            )
    else:
        tqdm.write(f"[skip:tabdiff-generate] existing synthetic file: {synthetic_path}", file=sys.stdout)

    for seed in args.seeds:
        run_dir = output_root / f"seed_{seed}"
        seed_run_dirs[int(seed)] = run_dir
        final_synthetic_path = args.synthetic_data_path
        if args.dry_run:
            previous_metrics = {}
        else:
            require_file_sha256(config_path, config_sha256, "Experiment config")
            require_scientific_code_sha256(code_sha256)
            if initial_integrity is None:
                raise RuntimeError("Initial run integrity was not validated.")
            validate_runner_integrity(
                synthetic_path,
                project_path(args.data_path),
                args.split_seed,
                args.split_method,
                args.generation_seed,
                config_sha256,
                protocol_hash,
                producer_protocol_hash,
                cross_run_statistics_path,
                cross_run_statistics_sha256,
                label_col=args.label_col,
                scientific_code_hash=code_sha256,
                expected_fields=initial_integrity,
            )
            previous_metrics = metrics_output_snapshot(run_dir)

        phase = ExperimentPhase(
            name=f"seed {seed} main-train",
            command=build_main_train_command(
                args,
                seed,
                run_dir,
                final_synthetic_path,
                tabdiff_num_samples,
                scientific_code_hash=code_sha256,
                generation_protocol_hash=producer_protocol_hash,
            ),
            run_dir=run_dir,
            output_path=run_dir / "results" / "metrics.json",
        )
        phase_record = {"name": phase.name, "command": phase.command, "run_dir": str(phase.run_dir)}
        phase_records.append(phase_record)
        run_phase(phase, args.dry_run)
        if not args.dry_run:
            require_file_sha256(config_path, config_sha256, "Experiment config")
            require_scientific_code_sha256(code_sha256)
            assert initial_integrity is not None
            validate_runner_integrity(
                synthetic_path,
                project_path(args.data_path),
                args.split_seed,
                args.split_method,
                args.generation_seed,
                config_sha256,
                protocol_hash,
                producer_protocol_hash,
                cross_run_statistics_path,
                cross_run_statistics_sha256,
                label_col=args.label_col,
                scientific_code_hash=code_sha256,
                expected_fields=initial_integrity,
            )
            fresh_metrics, fresh_metrics_sha256 = require_fresh_seed_metrics(
                run_dir,
                previous_metrics,
                expected_seed=int(seed),
                expected_integrity=initial_integrity,
            )
            phase.output_path = fresh_metrics
            seed_metrics_paths[int(seed)] = fresh_metrics
            seed_metrics_sha256[int(seed)] = fresh_metrics_sha256
            phase_record["output_path"] = str(fresh_metrics)

    summary_path = write_runner_summary(
        output_root,
        phase_records,
        seed_run_dirs,
        args.dry_run,
        use_dynamic_synthetic_agent=args.use_dynamic_synthetic_agent,
        split_seed=args.split_seed,
        split_method=args.split_method,
        generation_seed=args.generation_seed,
        seed_metrics_paths=None if args.dry_run else seed_metrics_paths,
        seed_metrics_sha256=None if args.dry_run else seed_metrics_sha256,
        integrity_fields=initial_integrity,
        effective_protocol=protocol_snapshot,
        effective_protocol_hash=protocol_hash,
    )
    print(f"\n[done] Main experiment outputs: {output_root}", flush=True)
    print(f"[done] Runner summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
