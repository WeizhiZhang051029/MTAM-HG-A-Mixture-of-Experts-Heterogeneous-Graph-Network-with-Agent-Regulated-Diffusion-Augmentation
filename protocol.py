"""Canonical constants for the released MTAM-HG experiment."""

from __future__ import annotations

DEFAULT_DATA_PATH = "data/CAPL.xlsx"
DEFAULT_LABEL_COL = "屈服强度"
DEFAULT_CONFIG_PATH = "configs/mtam_hg.yaml"
DEFAULT_SYNTHETIC_DATA_PATH = "data/synthetic_CAPL_ma_tabdiff.xlsx"

DEFAULT_SPLIT_SEED = 42
DEFAULT_SEEDS = list(range(42, 52))
DEFAULT_SPLIT_METHOD = "stratified_random"
DEFAULT_GENERATION_SEED = 0
DEFAULT_MAIN_OUTPUT_ROOT = "outputs/mtam_hg"

DEFAULT_EPOCHS = 50
DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS = 100
DEFAULT_SYNTHETIC_AGENT_EPOCHS = 100
DEFAULT_SYNTHETIC_AGENT_LR = 1.0e-3
DEFAULT_SYNTHETIC_AGENT_HIDDEN_DIM = 128
DEFAULT_SYNTHETIC_AGENT_ATTENTION_DIM = 64
DEFAULT_SYNTHETIC_AGENT_ATTENTION_HEADS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1.0e-3
DEFAULT_WEIGHT_DECAY = 1.0e-3
DEFAULT_DROPOUT = 0.20
DEFAULT_EARLY_STOPPING_PATIENCE = 5

DEFAULT_SYNTHETIC_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_USE_DYNAMIC_SYNTHETIC_AGENT = True
DEFAULT_DYNAMIC_SYNTHETIC_REFRESH_EPOCHS = 5
DEFAULT_DYNAMIC_SYNTHETIC_WARMUP_EPOCHS = 5
DEFAULT_DYNAMIC_SYNTHETIC_USE_SAMPLER = True
DEFAULT_DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT = True
DEFAULT_DYNAMIC_SYNTHETIC_TOP_RATIO = 0.60
DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MIN = 0.05
DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MAX = 3.0
DEFAULT_DYNAMIC_SYNTHETIC_EMA = 0.70
DEFAULT_DYNAMIC_SYNTHETIC_ERROR_WEIGHT = 0.45
DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_REGION_WEIGHT = 0.30
DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_WEIGHT = 0.25
DEFAULT_DYNAMIC_SYNTHETIC_REAL_FEEDBACK_WEIGHT = 0.35
DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_STRENGTH = 0.50
DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_MIN = 0.50
DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_MAX = 1.75
DEFAULT_DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR = 0.20
DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_BINS = 10
DEFAULT_DYNAMIC_SYNTHETIC_PROCESS_POWER = 1.0
DEFAULT_DYNAMIC_SYNTHETIC_MECHANISM_POWER = 1.0
DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC = "rmse_tail"
DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_TAIL_LAMBDA = 0.25

DEFAULT_USE_CLUSTER_BALANCE_REWARD = True
DEFAULT_NUM_WORKING_CONDITION_CLUSTERS = 5
DEFAULT_CLUSTER_BALANCE_LAMBDA = 0.3
DEFAULT_REWARD_ALPHA_CLUSTER = 0.3

DEFAULT_FINETUNE_BACKBONE_LR = 1.0e-4
DEFAULT_FINETUNE_HEAD_LR = 5.0e-4
DEFAULT_FINETUNE_AGENT_LR = 1.0e-3
DEFAULT_FINETUNE_QUALITY_AGENT_LR = 1.0e-3
DEFAULT_FREEZE_FINETUNE_BACKBONE = True
DEFAULT_USE_MR_LORA = True
DEFAULT_MR_LORA_SCOPE = "graph_attention_routing"
DEFAULT_MR_LORA_RANK_GRAPH = 8
DEFAULT_MR_LORA_RANK_ROUTING = 4
DEFAULT_MR_LORA_ALPHA_GRAPH = 16.0
DEFAULT_MR_LORA_ALPHA_ROUTING = 8.0
DEFAULT_MR_LORA_DROPOUT = 0.05
DEFAULT_MR_LORA_TRAIN_OUTPUT_HEAD = False
DEFAULT_MAIN_CHECKPOINT_SELECTION_METRIC = "rmse"
DEFAULT_MAIN_CHECKPOINT_TAIL_MAE_LAMBDA = 0.0
DEFAULT_TABDIFF_NUM_SAMPLES = 5000
DEFAULT_CBTG_CROSS_RUN_VALIDATION_STD_PATH = "data/cbtg_cross_run_validation_std.json"
CONFIRMATORY_YAML_CANONICAL_SHA256 = "c4b306f732f61c999b3f1d254685cc44ae110bd04294117f34a89bc730eb6bd2"

CONFIRMATORY_LOCKED_ARGS = {
    "epochs": DEFAULT_EPOCHS,
    "synthetic_pretrain_epochs": DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS,
    "synthetic_agent_epochs": DEFAULT_SYNTHETIC_AGENT_EPOCHS,
    "synthetic_agent_lr": DEFAULT_SYNTHETIC_AGENT_LR,
    "synthetic_agent_hidden_dim": DEFAULT_SYNTHETIC_AGENT_HIDDEN_DIM,
    "synthetic_agent_attention_dim": DEFAULT_SYNTHETIC_AGENT_ATTENTION_DIM,
    "synthetic_agent_attention_heads": DEFAULT_SYNTHETIC_AGENT_ATTENTION_HEADS,
    "dropout": DEFAULT_DROPOUT,
    "agent_dropout": DEFAULT_DROPOUT,
    "synthetic_agent_dropout": DEFAULT_DROPOUT,
    "synthetic_confidence_threshold": DEFAULT_SYNTHETIC_CONFIDENCE_THRESHOLD,
    "synthetic_pretrain_confidence_threshold": 0.0,
    "use_dynamic_synthetic_agent": DEFAULT_USE_DYNAMIC_SYNTHETIC_AGENT,
    "dynamic_synthetic_refresh_epochs": DEFAULT_DYNAMIC_SYNTHETIC_REFRESH_EPOCHS,
    "dynamic_synthetic_warmup_epochs": DEFAULT_DYNAMIC_SYNTHETIC_WARMUP_EPOCHS,
    "dynamic_synthetic_use_sampler": DEFAULT_DYNAMIC_SYNTHETIC_USE_SAMPLER,
    "dynamic_synthetic_use_loss_weight": DEFAULT_DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT,
    "dynamic_synthetic_top_ratio": DEFAULT_DYNAMIC_SYNTHETIC_TOP_RATIO,
    "dynamic_synthetic_weight_min": DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MIN,
    "dynamic_synthetic_weight_max": DEFAULT_DYNAMIC_SYNTHETIC_WEIGHT_MAX,
    "dynamic_synthetic_ema": DEFAULT_DYNAMIC_SYNTHETIC_EMA,
    "dynamic_synthetic_error_weight": DEFAULT_DYNAMIC_SYNTHETIC_ERROR_WEIGHT,
    "dynamic_synthetic_train_region_weight": DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_REGION_WEIGHT,
    "dynamic_synthetic_scarcity_weight": DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_WEIGHT,
    "dynamic_synthetic_real_feedback_weight": DEFAULT_DYNAMIC_SYNTHETIC_REAL_FEEDBACK_WEIGHT,
    "dynamic_synthetic_quota_strength": DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_STRENGTH,
    "dynamic_synthetic_quota_min": DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_MIN,
    "dynamic_synthetic_quota_max": DEFAULT_DYNAMIC_SYNTHETIC_QUOTA_MAX,
    "dynamic_synthetic_reliability_floor": DEFAULT_DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR,
    "dynamic_synthetic_scarcity_bins": DEFAULT_DYNAMIC_SYNTHETIC_SCARCITY_BINS,
    "dynamic_synthetic_process_power": DEFAULT_DYNAMIC_SYNTHETIC_PROCESS_POWER,
    "dynamic_synthetic_mechanism_power": DEFAULT_DYNAMIC_SYNTHETIC_MECHANISM_POWER,
    "dynamic_synthetic_train_reward_metric": DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC,
    "dynamic_synthetic_train_tail_lambda": DEFAULT_DYNAMIC_SYNTHETIC_TRAIN_TAIL_LAMBDA,
    "use_cluster_balance_reward": DEFAULT_USE_CLUSTER_BALANCE_REWARD,
    "num_working_condition_clusters": DEFAULT_NUM_WORKING_CONDITION_CLUSTERS,
    "cluster_balance_lambda": DEFAULT_CLUSTER_BALANCE_LAMBDA,
    "reward_alpha_cluster": DEFAULT_REWARD_ALPHA_CLUSTER,
    "finetune_backbone_lr": DEFAULT_FINETUNE_BACKBONE_LR,
    "finetune_head_lr": DEFAULT_FINETUNE_HEAD_LR,
    "finetune_agent_lr": DEFAULT_FINETUNE_AGENT_LR,
    "finetune_quality_agent_lr": DEFAULT_FINETUNE_QUALITY_AGENT_LR,
    "use_layerwise_finetune_lr": True,
    "freeze_finetune_backbone": DEFAULT_FREEZE_FINETUNE_BACKBONE,
    "use_mr_lora": DEFAULT_USE_MR_LORA,
    "mr_lora_scope": DEFAULT_MR_LORA_SCOPE,
    "mr_lora_rank_graph": DEFAULT_MR_LORA_RANK_GRAPH,
    "mr_lora_rank_routing": DEFAULT_MR_LORA_RANK_ROUTING,
    "mr_lora_alpha_graph": DEFAULT_MR_LORA_ALPHA_GRAPH,
    "mr_lora_alpha_routing": DEFAULT_MR_LORA_ALPHA_ROUTING,
    "mr_lora_dropout": DEFAULT_MR_LORA_DROPOUT,
    "mr_lora_train_output_head": DEFAULT_MR_LORA_TRAIN_OUTPUT_HEAD,
    "batch_size": DEFAULT_BATCH_SIZE,
    "lr": DEFAULT_LR,
    "weight_decay": DEFAULT_WEIGHT_DECAY,
    "early_stopping_patience": DEFAULT_EARLY_STOPPING_PATIENCE,
    "checkpoint_selection_metric": DEFAULT_MAIN_CHECKPOINT_SELECTION_METRIC,
    "checkpoint_tail_mae_lambda": DEFAULT_MAIN_CHECKPOINT_TAIL_MAE_LAMBDA,
}

RUNNER_SUMMARY_NAME = "experiment_summary.json"
SUPERVISED_MAIN_TRAIN_MODE = "train_with_tabdiff_pretrain"
DEFAULT_MAIN_TRAIN_MODE = SUPERVISED_MAIN_TRAIN_MODE
SUPERVISED_MAIN_EXPERIMENT_NAME = "mtam_hg_paper"
MAIN_EXPERIMENT_NAMES = {SUPERVISED_MAIN_EXPERIMENT_NAME}
MAIN_EXPERIMENT_MODEL = "mtam_hg"
MAIN_MODEL_ALIAS = "mtam_hg"

MAIN_PY_MODE_CHOICES = (
    "evaluate",
    "generate_synthetic_tabdiff",
    "pretrain_synthetic",
    SUPERVISED_MAIN_TRAIN_MODE,
)

DYNAMIC_SYNTHETIC_RUNNER_ARG_SPECS = [
    ("use_dynamic_synthetic_agent", "--use_dynamic_synthetic_agent"),
    ("dynamic_synthetic_refresh_epochs", "--dynamic_synthetic_refresh_epochs"),
    ("dynamic_synthetic_warmup_epochs", "--dynamic_synthetic_warmup_epochs"),
    ("dynamic_synthetic_use_sampler", "--dynamic_synthetic_use_sampler"),
    ("dynamic_synthetic_use_loss_weight", "--dynamic_synthetic_use_loss_weight"),
    ("dynamic_synthetic_top_ratio", "--dynamic_synthetic_top_ratio"),
    ("dynamic_synthetic_weight_min", "--dynamic_synthetic_weight_min"),
    ("dynamic_synthetic_weight_max", "--dynamic_synthetic_weight_max"),
    ("dynamic_synthetic_ema", "--dynamic_synthetic_ema"),
    ("dynamic_synthetic_error_weight", "--dynamic_synthetic_error_weight"),
    ("dynamic_synthetic_train_region_weight", "--dynamic_synthetic_train_region_weight"),
    ("dynamic_synthetic_scarcity_weight", "--dynamic_synthetic_scarcity_weight"),
    ("dynamic_synthetic_reliability_floor", "--dynamic_synthetic_reliability_floor"),
    ("dynamic_synthetic_scarcity_bins", "--dynamic_synthetic_scarcity_bins"),
    ("dynamic_synthetic_process_power", "--dynamic_synthetic_process_power"),
    ("dynamic_synthetic_mechanism_power", "--dynamic_synthetic_mechanism_power"),
    ("dynamic_synthetic_train_reward_metric", "--dynamic_synthetic_train_reward_metric"),
    ("dynamic_synthetic_train_tail_lambda", "--dynamic_synthetic_train_tail_lambda"),
]

CLUSTER_BALANCE_ARG_SPECS = [
    ("use_cluster_balance_reward", "--use_cluster_balance_reward"),
    ("num_working_condition_clusters", "--num_working_condition_clusters"),
    ("cluster_balance_lambda", "--cluster_balance_lambda"),
    ("reward_alpha_cluster", "--reward_alpha_cluster"),
]

MAIN_TRAIN_ARG_SPECS = [
    ("cbtg_cross_run_validation_std_path", "--cbtg_cross_run_validation_std_path"),
    ("dropout", "--dropout"),
    ("agent_dropout", "--agent_dropout"),
    ("synthetic_agent_epochs", "--synthetic_agent_epochs"),
    ("synthetic_agent_lr", "--synthetic_agent_lr"),
    ("synthetic_agent_hidden_dim", "--synthetic_agent_hidden_dim"),
    ("synthetic_agent_attention_dim", "--synthetic_agent_attention_dim"),
    ("synthetic_agent_attention_heads", "--synthetic_agent_attention_heads"),
    ("synthetic_agent_dropout", "--synthetic_agent_dropout"),
    ("synthetic_confidence_threshold", "--synthetic_confidence_threshold"),
    ("synthetic_pretrain_confidence_threshold", "--synthetic_pretrain_confidence_threshold"),
    *DYNAMIC_SYNTHETIC_RUNNER_ARG_SPECS,
    ("dynamic_synthetic_real_feedback_weight", "--dynamic_synthetic_real_feedback_weight"),
    ("dynamic_synthetic_quota_strength", "--dynamic_synthetic_quota_strength"),
    ("dynamic_synthetic_quota_min", "--dynamic_synthetic_quota_min"),
    ("dynamic_synthetic_quota_max", "--dynamic_synthetic_quota_max"),
    ("finetune_backbone_lr", "--finetune_backbone_lr"),
    ("finetune_head_lr", "--finetune_head_lr"),
    ("finetune_agent_lr", "--finetune_agent_lr"),
    ("finetune_quality_agent_lr", "--finetune_quality_agent_lr"),
    ("use_layerwise_finetune_lr", "--use_layerwise_finetune_lr"),
    ("freeze_finetune_backbone", "--freeze_finetune_backbone"),
    ("early_stopping_patience", "--early_stopping_patience"),
    ("checkpoint_selection_metric", "--checkpoint_selection_metric"),
    ("checkpoint_tail_mae_lambda", "--checkpoint_tail_mae_lambda"),
    *CLUSTER_BALANCE_ARG_SPECS,
]

MR_LORA_ARG_SPECS = [
    ("use_mr_lora", "--use_mr_lora"),
    ("mr_lora_scope", "--mr_lora_scope"),
    ("mr_lora_rank_graph", "--mr_lora_rank_graph"),
    ("mr_lora_rank_routing", "--mr_lora_rank_routing"),
    ("mr_lora_alpha_graph", "--mr_lora_alpha_graph"),
    ("mr_lora_alpha_routing", "--mr_lora_alpha_routing"),
    ("mr_lora_dropout", "--mr_lora_dropout"),
    ("mr_lora_train_output_head", "--mr_lora_train_output_head"),
]

BOOL_VALUE_FLAGS = {
    "--dynamic_synthetic_use_sampler",
    "--dynamic_synthetic_use_loss_weight",
}
