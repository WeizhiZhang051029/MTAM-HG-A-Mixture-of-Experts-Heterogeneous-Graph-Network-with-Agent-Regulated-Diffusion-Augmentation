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
DEFAULT_SYNTHETIC_AGENT_EPOCHS = 20
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
DEFAULT_DYNAMIC_SYNTHETIC_WARMUP_EPOCHS = 0
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
