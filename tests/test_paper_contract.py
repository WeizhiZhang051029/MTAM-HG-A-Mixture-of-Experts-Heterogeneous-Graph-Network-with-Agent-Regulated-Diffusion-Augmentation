"""Regression tests for the paper-facing MTAM-HG implementation contract."""

from __future__ import annotations

import sys
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from dataset import _split_indices, _split_target_counts  # noqa: E402
from generation.prepare import PAPER_FEATURE_KEYS  # noqa: E402
from models.mr_lora import mr_lora_scope_families  # noqa: E402
from protocol import (  # noqa: E402
    DEFAULT_EPOCHS,
    DEFAULT_FINETUNE_AGENT_LR,
    DEFAULT_FINETUNE_BACKBONE_LR,
    DEFAULT_FINETUNE_HEAD_LR,
    DEFAULT_GENERATION_SEED,
    DEFAULT_MR_LORA_ALPHA_GRAPH,
    DEFAULT_MR_LORA_ALPHA_ROUTING,
    DEFAULT_MR_LORA_DROPOUT,
    DEFAULT_MR_LORA_RANK_GRAPH,
    DEFAULT_MR_LORA_RANK_ROUTING,
    DEFAULT_MR_LORA_SCOPE,
    DEFAULT_NUM_WORKING_CONDITION_CLUSTERS,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_METHOD,
    DEFAULT_SPLIT_SEED,
    DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS,
    DEFAULT_TABDIFF_NUM_SAMPLES,
)
from run_experiment import PAPER_METRICS, build_parser  # noqa: E402
from third_party.TabDiff.tabdiff.models.capl_mechanism import (  # noqa: E402
    CAPLMechanismConstraint,
    CAPLMechanismWeights,
)
from training.cbtg import (  # noqa: E402
    PAPER_CBTG_LAMBDA_C,
    PAPER_CBTG_LAMBDA_H,
    PAPER_CBTG_LAMBDA_M,
    PAPER_CBTG_LAMBDA_S,
    PAPER_CBTG_LAMBDA_V,
    PAPER_CBTG_METRIC_NAMES,
    PAPER_CBTG_METRIC_WEIGHTS,
    PAPER_CBTG_REWARD_MAX,
    PAPER_CBTG_REWARD_MIN,
    PAPER_CBTG_STATE_COMPONENTS,
    PAPER_CBTG_TARGET_CONFIDENCE,
)

EXPECTED_INPUTS = (
    "FS",
    "JPF_PT",
    "HF_T",
    "SF_T",
    "SC_T",
    "FC1_T",
    "OA_T",
    "FC2_T",
    "Q_T",
    "RF",
    "BF",
    "HT",
    "FRT",
    "CT",
    "ATh",
    "AWd",
    "CRR",
    "C",
    "Mn",
    "S",
    "P",
)


class PaperContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = PROJECT_ROOT / "configs" / "mtam_hg.yaml"
        cls.paper_yaml = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def test_exact_21_input_schema_is_shared_by_model_and_tabdiff(self) -> None:
        self.assertEqual(tuple(config.active_node_names()), EXPECTED_INPUTS)
        self.assertEqual(tuple(PAPER_FEATURE_KEYS), EXPECTED_INPUTS)
        self.assertEqual(len(EXPECTED_INPUTS), 21)
        self.assertNotIn("EL", EXPECTED_INPUTS)
        self.assertFalse(config.USE_VIRTUAL_QUALITY_NODE)

    def test_fixed_stratified_70_15_15_split_for_600_records(self) -> None:
        self.assertEqual((config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO), (0.70, 0.15, 0.15))
        self.assertEqual(DEFAULT_SPLIT_METHOD, "stratified_random")
        self.assertEqual(config.SPLIT_METHOD, "stratified_random")
        self.assertEqual(config.SPLIT_SEED, DEFAULT_SPLIT_SEED)
        self.assertEqual(self.paper_yaml["split_seed"], DEFAULT_SPLIT_SEED)
        self.assertEqual(self.paper_yaml["split_method"], DEFAULT_SPLIT_METHOD)
        self.assertEqual(self.paper_yaml["generation_seed"], DEFAULT_GENERATION_SEED)
        self.assertEqual(self.paper_yaml["model_seeds"], DEFAULT_SEEDS)
        self.assertEqual(_split_target_counts(600), (420, 90, 90))

        y = np.linspace(200.0, 800.0, 600, dtype=np.float32).reshape(-1, 1)
        first = _split_indices(y, config.SPLIT_SEED, config.SPLIT_METHOD)
        second = _split_indices(y, config.SPLIT_SEED, config.SPLIT_METHOD)
        for first_part, second_part in zip(first, second):
            np.testing.assert_array_equal(first_part, second_part)

    def test_mp_tabdiff_uses_exact_three_mechanism_energies_and_paper_values(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(CAPLMechanismWeights)),
            ("temperature_path", "production_window", "yield_residual"),
        )
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(64, 1 + len(EXPECTED_INPUTS))).astype(np.float32)
        constraint = CAPLMechanismConstraint(
            [config.LABEL_COL, *EXPECTED_INPUTS],
            transformed_train=raw,
            raw_train=raw,
            temperature_hold_tolerance=10.0,
            yield_tolerance=0.0,
        )
        parts = constraint.energy_parts_from_physical(torch.from_numpy(raw[:4]))
        self.assertEqual(set(parts), {"temperature_path", "production_window", "yield_residual"})

        expected = {
            "tabdiff_num_samples": 5000,
            "tabdiff_mechanism_lambda": 0.05,
            "tabdiff_guidance_scale": 0.05,
            "tabdiff_mechanism_temperature_hold_tolerance": 10.0,
            "tabdiff_finetune_lr": 1.0e-4,
            "tabdiff_finetune_steps": 500,
            "tabdiff_num_timesteps_override": 50,
        }
        for key, value in expected.items():
            self.assertEqual(self.paper_yaml[key], value)
        self.assertEqual(config.TABDIFF_NUM_SAMPLES, DEFAULT_TABDIFF_NUM_SAMPLES)
        self.assertTrue(config.USE_TABDIFF_GENERATION)
        self.assertTrue(config.TABDIFF_MECHANISM_CONSTRAINT)

    def test_cbtg_agent_state_reward_and_selection_match_the_paper(self) -> None:
        self.assertEqual(
            PAPER_CBTG_METRIC_NAMES,
            ("RMSE", "MAE", "MAPE", "ONE_MINUS_R2", "TAIL_MAE"),
        )
        np.testing.assert_allclose(PAPER_CBTG_METRIC_WEIGHTS, (1.0, 0.3, 0.1, 0.3, 0.2))
        self.assertEqual(
            PAPER_CBTG_STATE_COMPONENTS,
            ("overall_mean", "run_std", "cluster_value", "cluster_variance"),
        )
        self.assertEqual((PAPER_CBTG_LAMBDA_S, PAPER_CBTG_LAMBDA_C, PAPER_CBTG_LAMBDA_V), (0.1, 0.3, 0.3))
        self.assertEqual((PAPER_CBTG_LAMBDA_M, PAPER_CBTG_LAMBDA_H), (0.01, 0.001))
        self.assertEqual((PAPER_CBTG_REWARD_MIN, PAPER_CBTG_REWARD_MAX), (-1.0, 1.0))
        self.assertEqual(PAPER_CBTG_TARGET_CONFIDENCE, 0.6)
        self.assertEqual(config.DYNAMIC_SYNTHETIC_REFRESH_EPOCHS, 5)
        self.assertEqual(config.DYNAMIC_SYNTHETIC_TOP_RATIO, 0.60)
        self.assertTrue(config.DYNAMIC_SYNTHETIC_USE_SAMPLER)
        self.assertTrue(config.DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT)
        self.assertEqual(config.DYNAMIC_SYNTHETIC_PROCESS_POWER, 1.0)
        self.assertEqual(config.DYNAMIC_SYNTHETIC_MECHANISM_POWER, 1.0)
        self.assertEqual(config.SYNTHETIC_AGENT_EPOCHS, 20)
        self.assertEqual(config.SYNTHETIC_AGENT_LR, 1.0e-3)

    def test_moe_ipohgn_hsg_and_mr_lora_hyperparameters_match_the_paper(self) -> None:
        self.assertEqual(config.GRAPH_BACKBONE_LAYERS, 2)
        self.assertEqual(config.NUM_EXPERTS, 4)
        self.assertEqual(config.TOP_K, 2)
        self.assertEqual(DEFAULT_NUM_WORKING_CONDITION_CLUSTERS, 5)
        self.assertEqual(config.NUM_WORKING_CONDITION_CLUSTERS, 5)
        self.assertEqual(config.MOE_AUX_LAMBDA, 0.2)
        self.assertEqual(config.EXPERT_CALIBRATION_LAMBDA, 0.03)
        self.assertEqual(config.EXPERT_DIVERSITY_LAMBDA, 0.001)
        self.assertEqual(config.GRAD_CLIP_NORM, 2.0)

        self.assertEqual(DEFAULT_MR_LORA_SCOPE, "graph_attention_routing")
        self.assertEqual(mr_lora_scope_families(DEFAULT_MR_LORA_SCOPE), {"graph", "attention", "routing"})
        self.assertEqual((DEFAULT_MR_LORA_RANK_GRAPH, DEFAULT_MR_LORA_RANK_ROUTING), (8, 4))
        self.assertEqual((DEFAULT_MR_LORA_ALPHA_GRAPH, DEFAULT_MR_LORA_ALPHA_ROUTING), (16.0, 8.0))
        self.assertEqual(DEFAULT_MR_LORA_DROPOUT, 0.05)

    def test_training_schedule_learning_rates_metrics_and_runs_match_the_paper(self) -> None:
        self.assertEqual(DEFAULT_SYNTHETIC_PRETRAIN_EPOCHS, 100)
        self.assertEqual(DEFAULT_EPOCHS, 50)
        self.assertEqual(config.WEIGHT_DECAY, 1.0e-3)
        self.assertEqual(DEFAULT_FINETUNE_BACKBONE_LR, 1.0e-4)
        self.assertEqual(DEFAULT_FINETUNE_HEAD_LR, 5.0e-4)
        self.assertEqual(DEFAULT_FINETUNE_AGENT_LR, 1.0e-3)
        self.assertEqual(PAPER_METRICS, ("RMSE", "MAE", "MAPE", "R2", "TAIL_MAE"))
        self.assertEqual(DEFAULT_SEEDS, list(range(42, 52)))
        self.assertEqual(len(DEFAULT_SEEDS), 10)

        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.synthetic_pretrain_epochs, 100)
        self.assertEqual(defaults.epochs, 50)
        self.assertEqual(defaults.mr_lora_scope, "graph_attention_routing")
        self.assertEqual(defaults.seeds, DEFAULT_SEEDS)
        self.assertEqual(defaults.split_seed, DEFAULT_SPLIT_SEED)
        self.assertEqual(defaults.split_method, DEFAULT_SPLIT_METHOD)
        self.assertEqual(defaults.generation_seed, DEFAULT_GENERATION_SEED)


if __name__ == "__main__":
    unittest.main()
