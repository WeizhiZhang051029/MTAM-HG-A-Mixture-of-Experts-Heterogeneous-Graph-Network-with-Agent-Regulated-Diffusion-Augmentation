"""Run the paper-aligned MTAM-HG main experiment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from models.mr_lora import MR_LORA_SCOPE_FAMILIES
from protocol import (
    BOOL_VALUE_FLAGS,
    DEFAULT_BATCH_SIZE,
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
    MAIN_TRAIN_ARG_SPECS,
    MR_LORA_ARG_SPECS,
    RUNNER_SUMMARY_NAME,
    SUPERVISED_MAIN_EXPERIMENT_NAME,
    SUPERVISED_MAIN_TRAIN_MODE,
)

PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_METRICS = ("RMSE", "MAE", "MAPE", "R2")

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
        "--skip_tabdiff_generation",
        action="store_true",
        help="Skip the TabDiff generation phase even when --synthetic_data_path is missing.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Choose one or more distinct run seeds.")
    if args.generation_seed != 0:
        raise ValueError("TabDiff deterministic generation uses seed 0.")
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
        str(seed),
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



def seed_template_path(template: str, seed: int) -> str:
    if "{seed}" in template:
        return template.format(seed=seed)
    path = Path(template)
    return str(path.with_name(f"{path.stem}_seed_{seed}{path.suffix}"))


def _sha256_value(value: object, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"Invalid {field}.")
    return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from config_loader import load_yaml_config

    parser = build_parser()
    initial, _ = parser.parse_known_args(argv)
    values = load_yaml_config(PROJECT_ROOT / initial.config)
    defaults = dict(values)
    defaults.update(values.get("cli", {}))
    defaults["seeds"] = values.get("model_seeds", DEFAULT_SEEDS)
    defaults["output_root"] = values.get("output_base", DEFAULT_MAIN_OUTPUT_ROOT)
    destinations = {action.dest for action in parser._actions}
    parser.set_defaults(**{k: v for k, v in defaults.items() if k in destinations})
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def run_experiments(args: argparse.Namespace) -> dict[str, object]:
    from datetime import datetime

    import numpy as np

    from config_loader import load_yaml_config

    root = (PROJECT_ROOT / args.output_root).resolve()
    root = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    count = int(args.tabdiff_num_samples or DEFAULT_TABDIFF_NUM_SAMPLES)
    results = []
    commands = []
    for seed in args.seeds:
        run_dir = root / f"seed_{seed}"
        synthetic = str((PROJECT_ROOT / seed_template_path(args.synthetic_data_path, seed)).resolve())
        if args.skip_tabdiff_generation and not Path(synthetic).is_file():
            raise FileNotFoundError(synthetic)
        command = build_main_train_command(args, seed, run_dir, synthetic, count)
        if args.tabdiff_gpu is not None:
            command.extend(["--tabdiff_gpu", str(args.tabdiff_gpu)])
        if args.skip_tabdiff_generation:
            command.append("--require_existing_synthetic")
        commands.append(command)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        paths = list(run_dir.rglob("metrics.json"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one fresh metrics file in {run_dir}, found {len(paths)}.")
        metrics = json.loads(paths[0].read_text(encoding="utf-8"))
        if metrics.get("Seed") != seed or metrics.get("Split_Seed") != seed:
            raise RuntimeError("Run and split seeds do not match.")
        if not all(math.isfinite(float(metrics[m])) for m in PAPER_METRICS):
            raise RuntimeError(f"Non-finite evaluation metric for seed {seed}.")
        results.append({"seed": seed, "metrics": metrics, "metrics_path": str(paths[0])})
    summary = {
        "protocol": "per_run_split",
        "feedback_source": "real_training_set",
        "config": load_yaml_config(PROJECT_ROOT / args.config),
        "arguments": vars(args),
        "commands": commands,
        "runs": results,
        "aggregate": {
            metric: {
                "mean": float(np.mean([run["metrics"][metric] for run in results])),
                "std": float(np.std([run["metrics"][metric] for run in results], ddof=1)) if len(results) > 1 else None,
            }
            for metric in PAPER_METRICS
        } if results else {},
    }
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        (root / RUNNER_SUMMARY_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Results: {root}")
    return summary


def main() -> None:
    run_experiments(parse_args())


if __name__ == "__main__":
    main()
