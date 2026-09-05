from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from dataset import (
    CAPLDataset,
    DataBundle,
    StandardScaler,
    _chronological_split_indices,
    _stratified_random_split_indices,
)
from evaluate import agent_selection_alignment_metrics
from losses import compute_agent_reward, moe_load_balance_loss, total_loss
from models.mtam_hg import (
    _topk_sparse_weights,
    router_load_balance_loss,
)
from pipeline import apply_cli_overrides, build_parser
from train import build_experiment_model, checkpoint_selection_score
from training.cbtg import (
    AGENT_FEEDBACK_FEATURE_NAMES,
    DynamicSyntheticState,
    SyntheticQualityAgent,
    agent_policy_quota_multiplier,
    build_paper_cbtg_feedback,
    compute_dynamic_synthetic_weights,
    compute_mechanism_consistency_scores,
    compute_process_consistency_scores,
    rebuild_synthetic_loader,
    select_synthetic_by_quality_score,
    select_top_synthetic_indices,
    synthetic_agent_reward_components,
    synthetic_agent_reward_from_real_mse,
    synthetic_scarcity_bonus,
)
from training.clusters import (
    cluster_balance_reward,
    compute_cluster_balance_stats,
)


class MTAMHGTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original = {
            "USE_EL_AS_INPUT": config.USE_EL_AS_INPUT,
            "USE_LAPLACE": config.USE_LAPLACE,
            "D_MODEL": config.D_MODEL,
            "GRAPH_EMBED_DIM": config.GRAPH_EMBED_DIM,
            "GRAPH_BACKBONE_LAYERS": config.GRAPH_BACKBONE_LAYERS,
            "DROPOUT": config.DROPOUT,
            "NUM_EXPERTS": config.NUM_EXPERTS,
            "TOP_K": config.TOP_K,
            "CHECKPOINT_SELECTION_METRIC": config.CHECKPOINT_SELECTION_METRIC,
            "CHECKPOINT_TAIL_MAE_LAMBDA": config.CHECKPOINT_TAIL_MAE_LAMBDA,
            "MOE_AUX_LAMBDA": config.MOE_AUX_LAMBDA,
            "EXPERT_CALIBRATION_LAMBDA": getattr(config, "EXPERT_CALIBRATION_LAMBDA", 0.0),
            "EXPERT_CALIBRATION_QUALITY_LAMBDA": getattr(config, "EXPERT_CALIBRATION_QUALITY_LAMBDA", 0.0),
            "EXPERT_CALIBRATION_QUALITY_INDEX": getattr(config, "EXPERT_CALIBRATION_QUALITY_INDEX", 3),
            "AGENT_HIDDEN_DIM": config.AGENT_HIDDEN_DIM,
            "AGENT_DROPOUT": config.AGENT_DROPOUT,
            "AGENT_USE_PROCESS_FEATURES": config.AGENT_USE_PROCESS_FEATURES,
            "AGENT_USE_EXPERT_PREDS": config.AGENT_USE_EXPERT_PREDS,
            "AGENT_USE_UNCERTAINTY": config.AGENT_USE_UNCERTAINTY,
            "AGENT_OUTPUT_SAMPLE_CONFIDENCE": config.AGENT_OUTPUT_SAMPLE_CONFIDENCE,
            "AGENT_REASON_DIM": config.AGENT_REASON_DIM,
            "AGENT_RELIABILITY_ROUTING_LAMBDA": config.AGENT_RELIABILITY_ROUTING_LAMBDA,
            "AGENT_USE_SAMPLE_WEIGHT_FOR_SUPERVISED_LOSS": config.AGENT_USE_SAMPLE_WEIGHT_FOR_SUPERVISED_LOSS,
            "AGENT_CONFIDENCE_REG_LAMBDA": config.AGENT_CONFIDENCE_REG_LAMBDA,
            "USE_AGENT_REWARD": config.USE_AGENT_REWARD,
            "AGENT_REWARD_LAMBDA": config.AGENT_REWARD_LAMBDA,
            "REWARD_ALPHA_ERROR": config.REWARD_ALPHA_ERROR,
            "REWARD_ALPHA_UNCERTAINTY": config.REWARD_ALPHA_UNCERTAINTY,
            "REWARD_ALPHA_ENTROPY": config.REWARD_ALPHA_ENTROPY,
            "REWARD_ALPHA_TAIL": config.REWARD_ALPHA_TAIL,
            "REWARD_CLAMP_MIN": config.REWARD_CLAMP_MIN,
            "REWARD_CLAMP_MAX": config.REWARD_CLAMP_MAX,
            "TARGET_CONFIDENCE_MEAN": config.TARGET_CONFIDENCE_MEAN,
            "CONFIDENCE_MEAN_REG_LAMBDA": config.CONFIDENCE_MEAN_REG_LAMBDA,
            "CONFIDENCE_ENTROPY_REG_LAMBDA": config.CONFIDENCE_ENTROPY_REG_LAMBDA,
            "USE_CONFIDENCE_WEIGHTED_SUPERVISED_LOSS": config.USE_CONFIDENCE_WEIGHTED_SUPERVISED_LOSS,
            "TAIL_QUANTILE_LOW": config.TAIL_QUANTILE_LOW,
            "TAIL_QUANTILE_HIGH": config.TAIL_QUANTILE_HIGH,
            "SYNTHETIC_PRETRAIN_CONFIDENCE_THRESHOLD": getattr(config, "SYNTHETIC_PRETRAIN_CONFIDENCE_THRESHOLD", 0.0),
            "SYNTHETIC_USE_PROCESS_CONSISTENCY": config.SYNTHETIC_USE_PROCESS_CONSISTENCY,
            "SYNTHETIC_PROCESS_CONSISTENCY_THRESHOLD": config.SYNTHETIC_PROCESS_CONSISTENCY_THRESHOLD,
            "SYNTHETIC_PROCESS_RANGE_QUANTILE_LOW": config.SYNTHETIC_PROCESS_RANGE_QUANTILE_LOW,
            "SYNTHETIC_PROCESS_RANGE_QUANTILE_HIGH": config.SYNTHETIC_PROCESS_RANGE_QUANTILE_HIGH,
            "SYNTHETIC_PROCESS_RANGE_MARGIN": config.SYNTHETIC_PROCESS_RANGE_MARGIN,
            "SYNTHETIC_PROCESS_KNN_K": config.SYNTHETIC_PROCESS_KNN_K,
            "SYNTHETIC_PROCESS_RANGE_WEIGHT": config.SYNTHETIC_PROCESS_RANGE_WEIGHT,
            "SYNTHETIC_PROCESS_MANIFOLD_WEIGHT": config.SYNTHETIC_PROCESS_MANIFOLD_WEIGHT,
            "SYNTHETIC_PROCESS_LABEL_WEIGHT": config.SYNTHETIC_PROCESS_LABEL_WEIGHT,
            "SYNTHETIC_PROCESS_SCORE_POWER": config.SYNTHETIC_PROCESS_SCORE_POWER,
            "SYNTHETIC_REWARD_MSE_WEIGHT": config.SYNTHETIC_REWARD_MSE_WEIGHT,
            "SYNTHETIC_REWARD_PROCESS_WEIGHT": config.SYNTHETIC_REWARD_PROCESS_WEIGHT,
            "SYNTHETIC_REWARD_MECHANISM_WEIGHT": config.SYNTHETIC_REWARD_MECHANISM_WEIGHT,
            "USE_LAYERWISE_FINETUNE_LR": getattr(config, "USE_LAYERWISE_FINETUNE_LR", False),
            "FINETUNE_BACKBONE_LR": getattr(config, "FINETUNE_BACKBONE_LR", 1.0e-3),
            "FINETUNE_HEAD_LR": getattr(config, "FINETUNE_HEAD_LR", 1.0e-3),
            "FINETUNE_AGENT_LR": getattr(config, "FINETUNE_AGENT_LR", 1.0e-3),
            "FREEZE_FINETUNE_BACKBONE": getattr(config, "FREEZE_FINETUNE_BACKBONE", False),
            "FINETUNE_TRAINABLE_KEYWORDS": list(getattr(config, "FINETUNE_TRAINABLE_KEYWORDS", [])),
            "USE_MR_LORA": getattr(config, "USE_MR_LORA", False),
            "MR_LORA_SCOPE": getattr(config, "MR_LORA_SCOPE", "graph_routing"),
            "MR_LORA_RANK_GRAPH": getattr(config, "MR_LORA_RANK_GRAPH", 4),
            "MR_LORA_RANK_ROUTING": getattr(config, "MR_LORA_RANK_ROUTING", 2),
            "MR_LORA_ALPHA_GRAPH": getattr(config, "MR_LORA_ALPHA_GRAPH", 8.0),
            "MR_LORA_ALPHA_ROUTING": getattr(config, "MR_LORA_ALPHA_ROUTING", 4.0),
            "MR_LORA_DROPOUT": getattr(config, "MR_LORA_DROPOUT", 0.05),
            "MR_LORA_TRAIN_OUTPUT_HEAD": getattr(config, "MR_LORA_TRAIN_OUTPUT_HEAD", False),
            "USE_CLUSTER_BALANCE_REWARD": getattr(config, "USE_CLUSTER_BALANCE_REWARD", False),
            "NUM_WORKING_CONDITION_CLUSTERS": getattr(config, "NUM_WORKING_CONDITION_CLUSTERS", 5),
            "CLUSTER_BALANCE_LAMBDA": getattr(config, "CLUSTER_BALANCE_LAMBDA", 0.3),
            "REWARD_ALPHA_CLUSTER": getattr(config, "REWARD_ALPHA_CLUSTER", 0.3),
            "ALLOW_UNTRAINED_AGENT_FILTER": getattr(config, "ALLOW_UNTRAINED_AGENT_FILTER", False),
            "SYNTHETIC_AGENT_HIDDEN_DIM": getattr(config, "SYNTHETIC_AGENT_HIDDEN_DIM", 128),
            "SYNTHETIC_AGENT_DROPOUT": getattr(config, "SYNTHETIC_AGENT_DROPOUT", 0.1),
            "SYNTHETIC_AGENT_EPOCHS": getattr(config, "SYNTHETIC_AGENT_EPOCHS", 20),
            "SYNTHETIC_AGENT_LR": getattr(config, "SYNTHETIC_AGENT_LR", 1.0e-3),
            "USE_DYNAMIC_SYNTHETIC_AGENT": getattr(config, "USE_DYNAMIC_SYNTHETIC_AGENT", False),
            "DYNAMIC_SYNTHETIC_WEIGHT_MIN": getattr(config, "DYNAMIC_SYNTHETIC_WEIGHT_MIN", 0.05),
            "DYNAMIC_SYNTHETIC_WEIGHT_MAX": getattr(config, "DYNAMIC_SYNTHETIC_WEIGHT_MAX", 3.0),
            "DYNAMIC_SYNTHETIC_TOP_RATIO": getattr(config, "DYNAMIC_SYNTHETIC_TOP_RATIO", 0.60),
            "DYNAMIC_SYNTHETIC_EMA": getattr(config, "DYNAMIC_SYNTHETIC_EMA", 0.70),
            "DYNAMIC_SYNTHETIC_ERROR_WEIGHT": getattr(config, "DYNAMIC_SYNTHETIC_ERROR_WEIGHT", 0.45),
            "DYNAMIC_SYNTHETIC_TRAIN_REGION_WEIGHT": getattr(config, "DYNAMIC_SYNTHETIC_TRAIN_REGION_WEIGHT", 0.30),
            "DYNAMIC_SYNTHETIC_SCARCITY_WEIGHT": getattr(config, "DYNAMIC_SYNTHETIC_SCARCITY_WEIGHT", 0.25),
            "DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR": getattr(config, "DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR", 0.20),
            "DYNAMIC_SYNTHETIC_PROCESS_POWER": getattr(config, "DYNAMIC_SYNTHETIC_PROCESS_POWER", 1.0),
            "DYNAMIC_SYNTHETIC_MECHANISM_POWER": getattr(config, "DYNAMIC_SYNTHETIC_MECHANISM_POWER", 1.0),
        }
        config.USE_EL_AS_INPUT = False
        config.USE_LAPLACE = True
        config.D_MODEL = 16
        config.GRAPH_EMBED_DIM = 16
        config.GRAPH_BACKBONE_LAYERS = 1
        config.DROPOUT = 0.0
        config.NUM_EXPERTS = 4
        config.TOP_K = 2
        config.MOE_AUX_LAMBDA = 0.01
        config.EXPERT_CALIBRATION_LAMBDA = 0.0
        config.EXPERT_CALIBRATION_QUALITY_LAMBDA = 0.0
        config.EXPERT_CALIBRATION_QUALITY_INDEX = 3
        config.AGENT_HIDDEN_DIM = 16
        config.AGENT_DROPOUT = 0.0
        config.AGENT_USE_PROCESS_FEATURES = True
        config.AGENT_USE_EXPERT_PREDS = True
        config.AGENT_USE_UNCERTAINTY = True
        config.AGENT_OUTPUT_SAMPLE_CONFIDENCE = True
        config.AGENT_REASON_DIM = 4
        config.AGENT_RELIABILITY_ROUTING_LAMBDA = 1.0
        config.AGENT_USE_SAMPLE_WEIGHT_FOR_SUPERVISED_LOSS = False
        config.AGENT_CONFIDENCE_REG_LAMBDA = 0.0
        config.USE_AGENT_REWARD = False
        config.AGENT_REWARD_LAMBDA = 0.01
        config.REWARD_ALPHA_ERROR = 1.0
        config.REWARD_ALPHA_UNCERTAINTY = 0.5
        config.REWARD_ALPHA_ENTROPY = 0.1
        config.REWARD_ALPHA_TAIL = 0.2
        config.REWARD_CLAMP_MIN = -3.0
        config.REWARD_CLAMP_MAX = 3.0
        config.TARGET_CONFIDENCE_MEAN = 0.6
        config.CONFIDENCE_MEAN_REG_LAMBDA = 0.01
        config.CONFIDENCE_ENTROPY_REG_LAMBDA = 0.001
        config.USE_CONFIDENCE_WEIGHTED_SUPERVISED_LOSS = False
        config.TAIL_QUANTILE_LOW = 0.1
        config.TAIL_QUANTILE_HIGH = 0.9
        config.SYNTHETIC_PRETRAIN_CONFIDENCE_THRESHOLD = 0.0
        config.SYNTHETIC_USE_PROCESS_CONSISTENCY = True
        config.SYNTHETIC_PROCESS_CONSISTENCY_THRESHOLD = 0.0
        config.SYNTHETIC_PROCESS_RANGE_QUANTILE_LOW = 0.0
        config.SYNTHETIC_PROCESS_RANGE_QUANTILE_HIGH = 1.0
        config.SYNTHETIC_PROCESS_RANGE_MARGIN = 0.0
        config.SYNTHETIC_PROCESS_KNN_K = 2
        config.SYNTHETIC_PROCESS_RANGE_WEIGHT = 0.35
        config.SYNTHETIC_PROCESS_MANIFOLD_WEIGHT = 0.35
        config.SYNTHETIC_PROCESS_LABEL_WEIGHT = 0.30
        config.SYNTHETIC_PROCESS_SCORE_POWER = 1.0
        config.SYNTHETIC_REWARD_MSE_WEIGHT = 0.50
        config.SYNTHETIC_REWARD_PROCESS_WEIGHT = 0.25
        config.SYNTHETIC_REWARD_MECHANISM_WEIGHT = 0.25
        config.USE_LAYERWISE_FINETUNE_LR = True
        config.FINETUNE_BACKBONE_LR = 1.0e-4
        config.FINETUNE_HEAD_LR = 5.0e-4
        config.FINETUNE_AGENT_LR = 5.0e-4
        config.SYNTHETIC_AGENT_HIDDEN_DIM = 16
        config.SYNTHETIC_AGENT_DROPOUT = 0.0
        config.SYNTHETIC_AGENT_EPOCHS = 1
        config.SYNTHETIC_AGENT_LR = 1.0e-3
        config.DYNAMIC_SYNTHETIC_WEIGHT_MIN = 0.05
        config.DYNAMIC_SYNTHETIC_WEIGHT_MAX = 3.0
        config.DYNAMIC_SYNTHETIC_TOP_RATIO = 0.60
        config.DYNAMIC_SYNTHETIC_EMA = 0.0
        config.DYNAMIC_SYNTHETIC_ERROR_WEIGHT = 0.45
        config.DYNAMIC_SYNTHETIC_TRAIN_REGION_WEIGHT = 0.30
        config.DYNAMIC_SYNTHETIC_SCARCITY_WEIGHT = 0.25
        config.DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR = 0.20
        config.DYNAMIC_SYNTHETIC_PROCESS_POWER = 1.0
        config.DYNAMIC_SYNTHETIC_MECHANISM_POWER = 1.0

    def tearDown(self) -> None:
        for name, value in self._original.items():
            setattr(config, name, value)

    def test_mtam_hg_forward_contract_and_backward(self) -> None:
        model = build_experiment_model()
        x = torch.randn(4, len(config.input_node_names(config.USE_EL_AS_INPUT)))
        y = torch.randn(4, 1)

        outputs = model(x)
        loss, logs = total_loss(outputs, y, x=x)
        loss.backward()

        self.assertEqual(outputs["mu"].shape, (4, 1))
        self.assertEqual(outputs["y_pred"].shape, (4, 1))
        self.assertEqual(outputs["expert_preds"].shape, (4, 4, 1))
        self.assertEqual(outputs["expert_weights"].shape, (4, 4))
        self.assertEqual(outputs["gate_probs"].shape, (4, 4))
        self.assertEqual(outputs["topk_indices"].shape, (4, 2))
        self.assertEqual(outputs["A0"].shape, outputs["A_kg"].shape)
        self.assertEqual(outputs["A_kg_experts"].shape, (4, *outputs["A0"].shape))
        self.assertEqual(outputs["A0_experts"].shape, outputs["A_kg_experts"].shape)
        self.assertEqual(outputs["mechanism_masks"].shape, (4, outputs["A0"].shape[0]))
        self.assertEqual(outputs["expert_focus_masks"].shape, outputs["mechanism_masks"].shape)
        self.assertEqual(outputs["process_order_ids"].shape, (outputs["A0"].shape[0],))
        self.assertEqual(outputs["aux_loss"].ndim, 0)
        self.assertEqual(outputs["diversity_loss"].ndim, 0)
        self.assertTrue(torch.allclose(outputs["expert_weights"].sum(dim=1), torch.ones(4), atol=1.0e-5))
        self.assertTrue(torch.equal((outputs["expert_weights"] > 0).sum(dim=1), torch.full((4,), 2)))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("moe_aux_loss", logs)
        self.assertIn("expert_calibration_loss", logs)
        self.assertGreater(logs["graph_loss"], 0.0)
        expected_graph_loss = torch.sum(
            (outputs["A_kg_experts"] - outputs["A0_experts"]) ** 2
        )
        self.assertAlmostEqual(logs["graph_loss"], float(expected_graph_loss.detach().cpu()), places=6)
        self.assertIn("expert_diversity_loss", logs)
        for idx in range(4):
            self.assertIn(f"expert_weight_{idx}_mean", logs)
            self.assertIn(f"expert_usage_{idx}", logs)

    def test_expert_calibration_loss_adds_supervision_for_all_experts(self) -> None:
        config.EXPERT_CALIBRATION_LAMBDA = 0.1
        config.EXPERT_CALIBRATION_QUALITY_LAMBDA = 0.5
        config.EXPERT_CALIBRATION_QUALITY_INDEX = 3
        y = torch.zeros(2, 1)
        expert_preds = torch.zeros(2, 4, 1)
        expert_preds[:, 3, :] = 2.0
        outputs = {
            "mu": torch.zeros(2, 1),
            "expert_preds": expert_preds,
            "gate_weights": [torch.full((2, 4), 0.25)],
        }

        loss, logs = total_loss(outputs, y)

        self.assertGreater(logs["expert_calibration_loss"], 0.0)
        self.assertGreater(float(loss.detach()), 0.0)

    def test_mtam_hg_uses_four_ipohgn_experts_with_mechanism_masks(self) -> None:
        model = build_experiment_model()
        ipohgn_experts = [
            module
            for module in model.modules()
            if module.__class__.__name__ == "IPOHGNExpert"
        ]

        self.assertEqual(len(ipohgn_experts), 4)
        self.assertTrue(hasattr(model, "experts"))
        self.assertEqual(len(model.experts), 4)
        self.assertEqual(model.mechanism_masks.shape, (4, len(config.active_node_names())))
        for idx, name in enumerate(model.expert_names):
            with self.subTest(expert=name):
                self.assertGreater(float(model.mechanism_masks[idx].sum()), 0.0)
                self.assertTrue(torch.equal(model.experts[idx].mechanism_focus_mask.cpu(), model.mechanism_masks[idx].cpu()))

        x = torch.randn(3, len(config.input_node_names(config.USE_EL_AS_INPUT)))
        outputs = model(x)

        self.assertEqual(outputs["expert_preds"].shape, (3, 4, 1))
        self.assertEqual(outputs["mechanism_masks"].shape, (4, len(config.active_node_names())))
        self.assertEqual(outputs["expert_focus_masks"].shape, outputs["mechanism_masks"].shape)

    def test_default_experiment_model_is_mtam_hg(self) -> None:
        model = build_experiment_model()

        self.assertEqual(model.__class__.__name__, "MTAMHG")
        self.assertEqual(model.num_experts, 4)
        self.assertEqual(model.top_k, 2)

    def test_reliability_aware_router_forward_contract_and_backward(self) -> None:
        model = build_experiment_model()
        x = torch.randn(4, len(config.input_node_names(config.USE_EL_AS_INPUT)))
        y = torch.randn(4, 1)

        outputs = model(x)
        loss, logs = total_loss(outputs, y, x=x)
        loss.backward()

        self.assertEqual(outputs["mu"].shape, (4, 1))
        self.assertEqual(outputs["expert_preds"].shape, (4, 4, 1))
        self.assertEqual(outputs["expert_weights"].shape, (4, 4))
        self.assertEqual(outputs["topk_indices"].shape, (4, 2))
        self.assertEqual(outputs["sample_confidence"].shape, (4, 1))
        self.assertEqual(outputs["synthetic_keep_score"].shape, (4, 1))
        self.assertEqual(outputs["training_weight"].shape, (4, 1))
        self.assertEqual(outputs["expert_reliability"].shape, (4, 4))
        self.assertEqual(outputs["uncertainty_reason_vector"].shape, (4, 4))
        self.assertTrue(torch.all(outputs["sample_confidence"] >= 0))
        self.assertTrue(torch.all(outputs["sample_confidence"] <= 1))
        self.assertTrue(torch.all(outputs["synthetic_keep_score"] >= 0))
        self.assertTrue(torch.all(outputs["synthetic_keep_score"] <= 1))
        self.assertTrue(torch.all(outputs["training_weight"] >= 0))
        self.assertTrue(torch.all(outputs["training_weight"] <= 1))
        self.assertTrue(torch.all(outputs["expert_reliability"] >= 0))
        self.assertTrue(torch.all(outputs["expert_reliability"] <= 1))
        self.assertGreater(outputs["agent_state"].shape[-1], config.D_MODEL)
        self.assertTrue(torch.allclose(outputs["expert_weights"].sum(dim=1), torch.ones(4), atol=1.0e-5))
        self.assertTrue(torch.equal((outputs["expert_weights"] > 0).sum(dim=1), torch.full((4,), 2)))
        self.assertEqual(outputs["aux_loss"].ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("sample_confidence_mean", logs)
        self.assertIn("sample_confidence_std", logs)
        self.assertIn("synthetic_keep_score_mean", logs)
        self.assertIn("agent_training_weight_mean", logs)
        self.assertIn("expert_reliability_mean", logs)
        self.assertIn("uncertainty_reason_0_mean", logs)
        self.assertIn("expert_uncertainty_mean", logs)
        self.assertIn("agent_gate_entropy", logs)

    def test_agent_reward_loss_logs_and_backward(self) -> None:
        config.USE_AGENT_REWARD = True
        model = build_experiment_model()
        x = torch.randn(4, len(config.input_node_names(config.USE_EL_AS_INPUT)))
        y = torch.randn(4, 1)

        outputs = model(x)
        loss, logs = total_loss(outputs, y, x=x)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("agent_reward_loss", logs)
        self.assertIn("agent_total_loss", logs)
        self.assertIn("reward_mean", logs)
        self.assertIn("reward_std", logs)
        self.assertIn("reward_min", logs)
        self.assertIn("reward_max", logs)
        self.assertIn("prediction_error_reward_mean", logs)
        self.assertIn("uncertainty_reward_mean", logs)
        self.assertIn("entropy_reward_mean", logs)
        self.assertIn("tail_bonus_mean", logs)
        self.assertIn("tail_sample_ratio", logs)
        self.assertIn("confidence_mean_reg", logs)
        self.assertIn("confidence_entropy", logs)

    def test_synthetic_quality_agent_scores_feature_label_and_main_feedback(self) -> None:
        feedback_dim = len(AGENT_FEEDBACK_FEATURE_NAMES)
        agent = SyntheticQualityAgent(
            input_dim=4,
            hidden_dim=8,
            dropout=0.0,
            attention_dim=8,
            attention_heads=2,
        )
        x = torch.randn(5, 3)
        y_generated = torch.randn(5, 1)
        feedback = torch.rand(5, feedback_dim)

        score = agent(x, y_generated, feedback)

        self.assertIsInstance(agent.cross_attention, torch.nn.MultiheadAttention)
        self.assertEqual(agent.net[0].in_features, 4 + feedback_dim + 16)
        self.assertEqual(score.shape, (5, 1))
        self.assertTrue(torch.all(score >= 0.0))
        self.assertTrue(torch.all(score <= 1.0))

    def test_synthetic_agent_reward_uses_real_mse_inverse_formula(self) -> None:
        mse_real = torch.tensor([0.0, 1.0, 3.0])

        reward = synthetic_agent_reward_from_real_mse(mse_real)

        self.assertTrue(torch.allclose(reward, torch.tensor([1.0, 0.5, 0.25])))

    def test_cluster_balance_reward_matches_working_condition_formula(self) -> None:
        config.CLUSTER_BALANCE_LAMBDA = 0.5
        y_pred = torch.tensor([1.0, 3.0, 2.0, 6.0])
        y_true = torch.tensor([0.0, 0.0, 2.0, 2.0])
        labels = torch.tensor([0, 0, 1, 1])

        rmse_mean, var_cluster, details = compute_cluster_balance_stats(
            y_pred,
            y_true,
            labels,
            num_clusters=2,
        )
        rmse_all = float(torch.sqrt(torch.mean((y_pred - y_true) ** 2)))
        reward = cluster_balance_reward(rmse_all, float(var_cluster), lambda_cluster=0.5)

        expected_rmse_0 = float(torch.sqrt(torch.tensor((1.0 + 9.0) / 2.0)))
        expected_rmse_1 = float(torch.sqrt(torch.tensor((0.0 + 16.0) / 2.0)))
        expected_mean = (expected_rmse_0 + expected_rmse_1) / 2.0
        expected_var = ((expected_rmse_0 - expected_mean) ** 2 + (expected_rmse_1 - expected_mean) ** 2) / 2.0
        expected_reward = 1.0 / (1.0 + rmse_all + 0.5 * expected_var)

        self.assertAlmostEqual(float(rmse_mean), expected_mean, places=6)
        self.assertAlmostEqual(float(var_cluster), expected_var, places=6)
        self.assertAlmostEqual(reward, expected_reward, places=6)
        self.assertEqual(details["cluster_active_count"], 2)

    def test_agent_reward_penalizes_cluster_rmse_imbalance(self) -> None:
        config.USE_CLUSTER_BALANCE_REWARD = True
        config.NUM_WORKING_CONDITION_CLUSTERS = 2
        config.REWARD_ALPHA_CLUSTER = 0.4
        y_pred = torch.tensor([[0.0], [4.0], [0.0], [4.0]])
        y_true = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
        expert_preds = torch.stack([y_pred, y_pred + 0.1], dim=1)
        gate_probs = torch.full((4, 2), 0.5)

        _, balanced = compute_agent_reward(
            y_pred,
            y_true,
            expert_preds,
            gate_probs,
            cluster_labels=torch.tensor([0, 0, 1, 1]),
        )
        _, imbalanced = compute_agent_reward(
            y_pred,
            y_true,
            expert_preds,
            gate_probs,
            cluster_labels=torch.tensor([0, 1, 1, 1]),
        )

        self.assertEqual(float(balanced["var_cluster"]), 0.0)
        self.assertGreater(float(imbalanced["var_cluster"]), 0.0)
        self.assertLess(
            float(imbalanced["cluster_balance_penalty"].mean()),
            float(balanced["cluster_balance_penalty"].mean()),
        )

    def test_cluster_balance_r2_variance_is_stable_for_flat_clusters(self) -> None:
        config.CLUSTER_BALANCE_R2_MIN = -1.0
        config.CLUSTER_BALANCE_R2_MAX = 1.0
        y_pred = torch.tensor([0.0, 1000.0, 0.0, 1.0])
        y_true = torch.tensor([1.0, 1.0, 2.0, 3.0])
        labels = torch.tensor([0, 0, 1, 1])

        _rmse_mean, _var_cluster, details = compute_cluster_balance_stats(
            y_pred,
            y_true,
            labels,
            num_clusters=2,
        )

        self.assertGreaterEqual(details["cluster_0_r2"], -1.0)
        self.assertLessEqual(details["cluster_0_r2"], 1.0)
        self.assertLessEqual(float(details["var_r2_tensor"]), 1.0)

    def test_cluster_balance_penalty_is_clamped_to_reward_scale(self) -> None:
        config.USE_CLUSTER_BALANCE_REWARD = True
        config.NUM_WORKING_CONDITION_CLUSTERS = 2
        config.REWARD_ALPHA_CLUSTER = 0.4
        config.CLUSTER_BALANCE_PENALTY_MIN = -1.0
        config.CLUSTER_BALANCE_PENALTY_MAX = 0.0
        y_pred = torch.tensor([[0.0], [1000.0], [0.0], [1.0]])
        y_true = torch.tensor([[1.0], [1.0], [2.0], [3.0]])
        expert_preds = torch.stack([y_pred, y_pred + 0.1], dim=1)
        gate_probs = torch.full((4, 2), 0.5)

        _reward, components = compute_agent_reward(
            y_pred,
            y_true,
            expert_preds,
            gate_probs,
            cluster_labels=torch.tensor([0, 0, 1, 1]),
        )

        penalty = components["cluster_balance_penalty"]
        self.assertGreaterEqual(float(penalty.min()), -1.0)
        self.assertLessEqual(float(penalty.max()), 0.0)

    def test_synthetic_agent_reward_blends_real_error_distribution_and_mechanism(self) -> None:
        mse_real = torch.tensor([0.0, 3.0])
        process = torch.tensor([0.2, 0.8])
        mechanism = torch.tensor([0.6, 0.4])

        parts = synthetic_agent_reward_components(mse_real, process, mechanism)

        expected = 0.5 * torch.tensor([1.0, 0.25]) + 0.25 * process + 0.25 * mechanism
        self.assertTrue(torch.allclose(parts["reward"], expected))
        self.assertTrue(torch.allclose(parts["reward_mse"], torch.tensor([1.0, 0.25])))
        self.assertTrue(torch.allclose(parts["reward_process"], process))
        self.assertTrue(torch.allclose(parts["reward_mechanism"], mechanism))

    def test_synthetic_quality_filter_is_strictly_greater_than_threshold(self) -> None:
        scores = np.array([0.49, 0.50, 0.51], dtype=np.float32)

        selected = select_synthetic_by_quality_score(scores, threshold=0.5)

        self.assertEqual(selected.tolist(), [False, False, True])

    def test_dynamic_top_ratio_selects_high_scoring_synthetic_rows(self) -> None:
        scores = np.array([0.4, 0.9, 0.2, 0.8, 0.1], dtype=np.float64)

        indices, mask = select_top_synthetic_indices(scores, top_ratio=0.60)

        self.assertEqual(indices.tolist(), [0, 1, 3])
        self.assertEqual(mask.tolist(), [True, True, False, True, False])

    def test_rebuild_synthetic_loader_uses_selected_top_ratio_subset(self) -> None:
        dataset = CAPLDataset(
            np.arange(10, dtype=np.float32).reshape(5, 2),
            np.arange(5, dtype=np.float32).reshape(5, 1),
            np.arange(5),
        )
        bundle = type("SyntheticBundleStub", (), {})()
        bundle.loader = DataLoader(dataset, batch_size=5, shuffle=False)
        state = DynamicSyntheticState(
            weights=np.ones(5, dtype=np.float64),
            selected_indices=np.array([1, 3, 4], dtype=np.int64),
            selected_mask=np.array([False, True, False, True, True], dtype=bool),
            previous_raw_weights=np.ones(5, dtype=np.float64),
            scarcity_bonus=np.zeros(5, dtype=np.float64),
            bin_ids=np.zeros(5, dtype=np.int64),
            bin_edges=np.array([0.0, 1.0], dtype=np.float64),
            train_bin_counts=np.array([5], dtype=np.int64),
            feedback_features=np.zeros((5, len(AGENT_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
            feedback_target=np.zeros(5, dtype=np.float64),
        )
        config.DYNAMIC_SYNTHETIC_USE_SAMPLER = True
        config.SYNTHETIC_BATCH_SIZE = 8

        loader = rebuild_synthetic_loader(bundle, state)
        seen = sorted(int(sample_id) for _x, _y, sample_ids in loader for sample_id in sample_ids.tolist())

        self.assertEqual(seen, [1, 3, 4])

    def test_rebuild_synthetic_loader_reselects_from_original_dataset(self) -> None:
        dataset = CAPLDataset(
            np.arange(10, dtype=np.float32).reshape(5, 2),
            np.arange(5, dtype=np.float32).reshape(5, 1),
            np.arange(5),
        )
        bundle = type("SyntheticBundleStub", (), {})()
        bundle.loader = DataLoader(dataset, batch_size=5, shuffle=False)
        state = DynamicSyntheticState(
            weights=np.ones(5, dtype=np.float64),
            selected_indices=np.array([1, 3, 4], dtype=np.int64),
            selected_mask=np.array([False, True, False, True, True], dtype=bool),
            previous_raw_weights=np.ones(5, dtype=np.float64),
            scarcity_bonus=np.zeros(5, dtype=np.float64),
            bin_ids=np.zeros(5, dtype=np.int64),
            bin_edges=np.array([0.0, 1.0], dtype=np.float64),
            train_bin_counts=np.array([5], dtype=np.int64),
            feedback_features=np.zeros((5, len(AGENT_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
            feedback_target=np.zeros(5, dtype=np.float64),
        )
        config.DYNAMIC_SYNTHETIC_USE_SAMPLER = True
        config.SYNTHETIC_BATCH_SIZE = 8

        rebuild_synthetic_loader(bundle, state)
        state.selected_indices = np.array([0, 2, 4], dtype=np.int64)
        loader = rebuild_synthetic_loader(bundle, state)
        seen = sorted(int(sample_id) for _x, _y, sample_ids in loader for sample_id in sample_ids.tolist())

        self.assertEqual(seen, [0, 2, 4])

    def test_dynamic_synthetic_scarcity_bonus_marks_sparse_train_bins(self) -> None:
        reference_y = np.array([10.0, 11.0, 12.0, 90.0], dtype=np.float32)
        synthetic_y = np.array([10.5, 90.0], dtype=np.float32)

        bonus, bin_ids, _edges, counts = synthetic_scarcity_bonus(synthetic_y, reference_y, n_bins=2)

        self.assertEqual(counts.tolist(), [3, 1])
        self.assertLess(bonus[0], bonus[1])
        self.assertNotEqual(int(bin_ids[0]), int(bin_ids[1]))

    def test_dynamic_synthetic_weight_penalizes_low_process_or_mechanism_consistency(self) -> None:
        config.DYNAMIC_SYNTHETIC_WEIGHT_MIN = 0.05
        config.DYNAMIC_SYNTHETIC_WEIGHT_MAX = 3.0
        config.DYNAMIC_SYNTHETIC_EMA = 0.0
        config.DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR = 0.20
        current = np.ones(3, dtype=np.float64)
        quality = np.array([0.95, 0.95, 0.95], dtype=np.float64)
        synthetic_mse = np.array([1.0, 10.0, 9.0], dtype=np.float64)
        train_region_mse = np.array([1.0, 10.0, 9.0], dtype=np.float64)
        process = np.array([1.0, 0.05, 1.0], dtype=np.float64)
        mechanism = np.array([1.0, 1.0, 0.05], dtype=np.float64)
        scarcity = np.array([0.2, 1.0, 1.0], dtype=np.float64)

        parts = compute_dynamic_synthetic_weights(
            current,
            quality,
            synthetic_mse,
            train_region_mse,
            process,
            mechanism,
            scarcity,
        )

        self.assertGreater(float(parts["weights"][0]), float(parts["weights"][1]))
        self.assertGreater(float(parts["weights"][0]), float(parts["weights"][2]))
        self.assertAlmostEqual(float(parts["selection_score"][1]), 0.95 * 0.05 * 1.0 * 2.0)
        self.assertAlmostEqual(float(parts["selection_score"][2]), 0.95 * 1.0 * 0.05 * 2.0)

    def test_agent_feedback_contains_four_states_for_each_paper_metric(self) -> None:
        overall = np.array([6.0, 4.0, 3.0, 0.1], dtype=np.float64)
        run_std = np.array([0.2, 0.1, 0.15, 0.01], dtype=np.float64)
        per_cluster = np.array(
            [
                [5.0, 3.0, 2.0, 0.08],
                [8.0, 6.0, 4.5, 0.20],
            ],
            dtype=np.float64,
        )

        parts = build_paper_cbtg_feedback(
            overall,
            run_std,
            per_cluster,
            np.array([0, 0, 1, 1], dtype=np.int64),
        )

        self.assertEqual(len(AGENT_FEEDBACK_FEATURE_NAMES), 16)
        self.assertEqual(parts["features"].shape, (4, 16))
        self.assertEqual(parts["raw_state"].shape, (4, 4, 4))
        self.assertTrue(np.all(parts["target"] >= -1.0))
        self.assertTrue(np.all(parts["target"] <= 1.0))

    def test_dynamic_synthetic_weight_is_driven_by_agent_policy_score(self) -> None:
        original = {
            "DYNAMIC_SYNTHETIC_WEIGHT_MIN": config.DYNAMIC_SYNTHETIC_WEIGHT_MIN,
            "DYNAMIC_SYNTHETIC_WEIGHT_MAX": config.DYNAMIC_SYNTHETIC_WEIGHT_MAX,
            "DYNAMIC_SYNTHETIC_EMA": config.DYNAMIC_SYNTHETIC_EMA,
            "DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR": config.DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR,
        }
        config.DYNAMIC_SYNTHETIC_WEIGHT_MIN = 0.05
        config.DYNAMIC_SYNTHETIC_WEIGHT_MAX = 3.0
        config.DYNAMIC_SYNTHETIC_EMA = 0.0
        config.DYNAMIC_SYNTHETIC_RELIABILITY_FLOOR = 0.0
        current = np.ones(4, dtype=np.float64)
        agent_policy = np.array([0.1, 0.2, 0.9, 1.0], dtype=np.float64)
        synthetic_mse = np.zeros(4, dtype=np.float64)
        train_region_mse = np.zeros(4, dtype=np.float64)
        process = np.ones(4, dtype=np.float64)
        mechanism = np.ones(4, dtype=np.float64)
        scarcity = np.zeros(4, dtype=np.float64)

        try:
            parts = compute_dynamic_synthetic_weights(
                current,
                agent_policy,
                synthetic_mse,
                train_region_mse,
                process,
                mechanism,
                scarcity,
            )

            self.assertLess(float(parts["weights"][0]), float(parts["weights"][2]))
            self.assertLess(float(parts["agent_policy_score"][0]), float(parts["agent_policy_score"][2]))
        finally:
            for name, value in original.items():
                setattr(config, name, value)

    def test_agent_policy_quota_multiplier_boosts_high_agent_policy(self) -> None:
        original = {
            "DYNAMIC_SYNTHETIC_QUOTA_STRENGTH": config.DYNAMIC_SYNTHETIC_QUOTA_STRENGTH,
            "DYNAMIC_SYNTHETIC_QUOTA_MIN": config.DYNAMIC_SYNTHETIC_QUOTA_MIN,
            "DYNAMIC_SYNTHETIC_QUOTA_MAX": config.DYNAMIC_SYNTHETIC_QUOTA_MAX,
        }
        config.DYNAMIC_SYNTHETIC_QUOTA_STRENGTH = 1.0
        config.DYNAMIC_SYNTHETIC_QUOTA_MIN = 0.25
        config.DYNAMIC_SYNTHETIC_QUOTA_MAX = 2.0
        agent_policy = np.array([0.1, 0.2, 0.9, 1.0], dtype=np.float64)
        reliability = np.ones(4, dtype=np.float64)

        try:
            quota = agent_policy_quota_multiplier(agent_policy, reliability)

            self.assertGreater(float(quota[2]), float(quota[0]))
            self.assertGreaterEqual(float(np.min(quota)), config.DYNAMIC_SYNTHETIC_QUOTA_MIN)
            self.assertLessEqual(float(np.max(quota)), config.DYNAMIC_SYNTHETIC_QUOTA_MAX)
        finally:
            for name, value in original.items():
                setattr(config, name, value)

    def test_agent_selection_alignment_metrics_include_confidence_error_relation(self) -> None:
        diag = pd.DataFrame(
            {
                "y_true": [100.0, 110.0, 180.0, 190.0],
                "abs_error": [1.0, 2.0, 12.0, 16.0],
                "sample_confidence": [0.95, 0.90, 0.40, 0.30],
                "expert_uncertainty": [0.1, 0.2, 0.8, 0.9],
            }
        )

        summary = agent_selection_alignment_metrics(diag)

        self.assertEqual(summary["samples"], 4)
        self.assertLess(summary["confidence_abs_error_spearman"], 0.0)
        self.assertGreater(summary["uncertainty_abs_error_spearman"], 0.0)
        self.assertIn("tail_abs_error_mean", summary)

    def test_cli_accepts_agent_reward_flags(self) -> None:
        args = build_parser().parse_args(["--use_agent_reward", "--agent_reward_lambda", "0.05"])

        self.assertTrue(args.use_agent_reward)
        self.assertEqual(args.agent_reward_lambda, 0.05)

    def test_cli_accepts_dynamic_synthetic_agent_flags(self) -> None:
        args = build_parser().parse_args(
            [
                "--use_dynamic_synthetic_agent",
                "--synthetic_agent_attention_dim",
                "48",
                "--synthetic_agent_attention_heads",
                "4",
                "--dynamic_synthetic_refresh_epochs",
                "4",
                "--dynamic_synthetic_warmup_epochs",
                "2",
                "--dynamic_synthetic_use_sampler",
                "False",
                "--dynamic_synthetic_use_loss_weight",
                "True",
                "--dynamic_synthetic_real_feedback_weight",
                "0.45",
                "--dynamic_synthetic_quota_strength",
                "0.6",
                "--dynamic_synthetic_quota_min",
                "0.4",
                "--dynamic_synthetic_quota_max",
                "1.6",
                "--dynamic_synthetic_train_reward_metric",
                "rmse_tail",
            ]
        )

        self.assertTrue(args.use_dynamic_synthetic_agent)
        self.assertEqual(args.synthetic_agent_attention_dim, 48)
        self.assertEqual(args.synthetic_agent_attention_heads, 4)
        self.assertEqual(args.dynamic_synthetic_refresh_epochs, 4)
        self.assertEqual(args.dynamic_synthetic_warmup_epochs, 2)
        self.assertFalse(args.dynamic_synthetic_use_sampler)
        self.assertTrue(args.dynamic_synthetic_use_loss_weight)
        self.assertEqual(args.dynamic_synthetic_real_feedback_weight, 0.45)
        self.assertEqual(args.dynamic_synthetic_quota_strength, 0.6)
        self.assertEqual(args.dynamic_synthetic_quota_min, 0.4)
        self.assertEqual(args.dynamic_synthetic_quota_max, 1.6)
        self.assertEqual(args.dynamic_synthetic_train_reward_metric, "rmse_tail")

    def test_cli_dynamic_synthetic_agent_flag_preserves_config_by_default(self) -> None:
        config.USE_DYNAMIC_SYNTHETIC_AGENT = True
        args = build_parser().parse_args([])

        apply_cli_overrides(args)

        self.assertTrue(config.USE_DYNAMIC_SYNTHETIC_AGENT)

        args = build_parser().parse_args(["--no_dynamic_synthetic_agent"])
        apply_cli_overrides(args)
        self.assertFalse(config.USE_DYNAMIC_SYNTHETIC_AGENT)

        args = build_parser().parse_args(["--use_dynamic_synthetic_agent"])
        apply_cli_overrides(args)
        self.assertTrue(config.USE_DYNAMIC_SYNTHETIC_AGENT)

    def test_cli_accepts_expert_calibration_overrides(self) -> None:
        args = build_parser().parse_args(
            [
                "--expert_calibration_lambda",
                "0.04",
                "--expert_calibration_quality_lambda",
                "0.09",
                "--expert_calibration_quality_index",
                "3",
            ]
        )

        self.assertEqual(args.expert_calibration_lambda, 0.04)
        self.assertEqual(args.expert_calibration_quality_lambda, 0.09)
        self.assertEqual(args.expert_calibration_quality_index, 3)

    def test_cli_accepts_early_stopping_controls(self) -> None:
        args = build_parser().parse_args(["--early_stopping_patience", "8", "--min_delta", "0.05"])

        self.assertEqual(args.early_stopping_patience, 8)
        self.assertEqual(args.min_delta, 0.05)

    def test_cli_accepts_checkpoint_selection_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "--checkpoint_selection_metric",
                "rmse_tail",
                "--checkpoint_tail_mae_lambda",
                "0.25",
            ]
        )

        self.assertEqual(args.checkpoint_selection_metric, "rmse_tail")
        self.assertEqual(args.checkpoint_tail_mae_lambda, 0.25)

    def test_checkpoint_selection_score_can_include_tail_mae(self) -> None:
        config.CHECKPOINT_SELECTION_METRIC = "rmse_tail"
        config.CHECKPOINT_TAIL_MAE_LAMBDA = 0.25

        score, logs = checkpoint_selection_score({"RMSE": 8.0, "TAIL_MAE": 4.0})

        self.assertEqual(score, 9.0)
        self.assertEqual(logs["checkpoint_selection_metric"], "rmse_tail")
        self.assertEqual(logs["checkpoint_score_rmse"], 8.0)
        self.assertEqual(logs["checkpoint_score_tail_mae"], 4.0)

    def test_process_consistency_downweights_off_manifold_synthetic_samples(self) -> None:
        x_train_raw = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
        y_train_raw = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        x_scaler = StandardScaler().fit(x_train_raw)
        y_scaler = StandardScaler().fit(y_train_raw)
        x_train = x_scaler.transform(x_train_raw)
        y_train = y_scaler.transform(y_train_raw)
        loader = DataLoader(CAPLDataset(x_train, y_train, np.arange(len(x_train))), batch_size=2)
        bundle = DataBundle(
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            standard_node_names=["a", "b"],
            graph_node_names=["a", "b"],
            feature_columns=["a", "b"],
            column_mapping={},
            label_column="y",
            tail_thresholds=(0.0, 3.0),
            y_train_raw=y_train_raw,
            train_sample_ids=np.arange(len(x_train)),
            val_sample_ids=np.array([], dtype=np.int64),
            test_sample_ids=np.array([], dtype=np.int64),
            split_sizes={"train": len(x_train), "val": 0, "test": 0},
            use_el_as_input=False,
            split_method="unit",
            data_path="unit",
        )
        x_syn_raw = np.array([[1.1, 1.1], [20.0, 20.0]], dtype=np.float32)
        y_syn_raw = np.array([[1.1], [20.0]], dtype=np.float32)
        scores = compute_process_consistency_scores(
            bundle,
            x_scaler.transform(x_syn_raw),
            y_scaler.transform(y_syn_raw),
            x_syn_raw,
            y_syn_raw,
        )

        self.assertGreater(scores["process_consistency"][0], scores["process_consistency"][1])
        self.assertGreater(scores["range_score"][0], scores["range_score"][1])
        self.assertGreater(scores["manifold_score"][0], scores["manifold_score"][1])

    def test_mechanism_consistency_downweights_metallurgical_outliers(self) -> None:
        feature_columns = ["C", "Mn", "S", "P", "ATh", "AWd", "CT", "RF", "BF", "CRR", "Q_T"]
        x_train_raw = np.tile(np.linspace(0.1, 1.1, len(feature_columns), dtype=np.float32), (4, 1))
        x_train_raw += np.arange(4, dtype=np.float32).reshape(-1, 1) * 0.01
        y_train_raw = np.arange(4, dtype=np.float32).reshape(-1, 1)
        x_scaler = StandardScaler().fit(x_train_raw)
        y_scaler = StandardScaler().fit(y_train_raw)
        loader = DataLoader(
            CAPLDataset(x_scaler.transform(x_train_raw), y_scaler.transform(y_train_raw), np.arange(len(x_train_raw))),
            batch_size=2,
        )
        bundle = DataBundle(
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            standard_node_names=feature_columns,
            graph_node_names=feature_columns,
            feature_columns=feature_columns,
            column_mapping={},
            label_column="y",
            tail_thresholds=(0.0, 3.0),
            y_train_raw=y_train_raw,
            train_sample_ids=np.arange(len(x_train_raw)),
            val_sample_ids=np.array([], dtype=np.int64),
            test_sample_ids=np.array([], dtype=np.int64),
            split_sizes={"train": len(x_train_raw), "val": 0, "test": 0},
            use_el_as_input=False,
            split_method="unit",
            data_path="unit",
        )
        in_mechanism = x_train_raw[:1].copy()
        off_mechanism = in_mechanism.copy()
        off_mechanism[:, [feature_columns.index("C"), feature_columns.index("Mn"), feature_columns.index("CT")]] = 99.0

        scores = compute_mechanism_consistency_scores(bundle, np.concatenate([in_mechanism, off_mechanism], axis=0))

        self.assertGreater(scores[0], scores[1])
        self.assertLess(scores[1], scores[0] * 0.75)

    def test_stratified_split_keeps_small_smoke_sets_trainable(self) -> None:
        y = np.arange(10, dtype=np.float32).reshape(-1, 1)

        train_idx, val_idx, test_idx = _stratified_random_split_indices(y, seed=42)

        self.assertEqual(len(train_idx), 7)
        self.assertEqual(len(val_idx), 2)
        self.assertEqual(len(test_idx), 1)
        self.assertEqual(len(set(train_idx) | set(val_idx) | set(test_idx)), 10)
        self.assertFalse(set(train_idx) & set(val_idx))
        self.assertFalse(set(train_idx) & set(test_idx))
        self.assertFalse(set(val_idx) & set(test_idx))

    def test_chronological_split_uses_non_empty_small_validation_split(self) -> None:
        train_idx, val_idx, test_idx = _chronological_split_indices(5)

        self.assertEqual(len(train_idx), 3)
        self.assertEqual(len(val_idx), 1)
        self.assertEqual(len(test_idx), 1)
        self.assertEqual(list(train_idx), [0, 1, 2])
        self.assertEqual(list(val_idx), [3])
        self.assertEqual(list(test_idx), [4])

    def test_layerwise_finetune_optimizer_uses_distinct_learning_rates(self) -> None:
        from train import build_finetune_optimizer

        config.USE_LAYERWISE_FINETUNE_LR = True
        config.FINETUNE_BACKBONE_LR = 1.0e-4
        config.FINETUNE_HEAD_LR = 5.0e-4
        config.FINETUNE_AGENT_LR = 7.0e-4
        model = build_experiment_model()

        optimizer = build_finetune_optimizer(model)
        lrs = {round(float(group["lr"]), 7) for group in optimizer.param_groups}

        self.assertIn(1.0e-4, lrs)
        self.assertIn(5.0e-4, lrs)
        self.assertIn(7.0e-4, lrs)

    def test_finetune_freeze_policy_trains_only_heads_gates_and_agent(self) -> None:
        from train import configure_finetune_trainability

        config.FREEZE_FINETUNE_BACKBONE = True
        config.FINETUNE_TRAINABLE_KEYWORDS = [
            "readout",
            "mu_head",
            "log_b_head",
            "gate_state_proj",
            "router",
        ]
        model = build_experiment_model()

        policy = configure_finetune_trainability(model)
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        frozen_names = [name for name, param in model.named_parameters() if not param.requires_grad]

        self.assertTrue(policy["freeze_backbone"])
        self.assertGreater(len(trainable_names), 0)
        self.assertGreater(len(frozen_names), 0)
        self.assertTrue(
            all(any(keyword in name for keyword in config.FINETUNE_TRAINABLE_KEYWORDS) for name in trainable_names)
        )
        self.assertTrue(any("experts.0.blocks.0" in name for name in frozen_names))
        self.assertTrue(any("experts.0.value_proj" in name for name in frozen_names))

    def test_paper_mr_lora_injection_separates_graph_attention_and_routing(self) -> None:
        from models.mr_lora import (
            ATTENTION_LORA_TARGETS,
            GRAPH_LORA_TARGETS,
            ROUTING_LORA_TARGETS,
            LoRALinear,
        )
        from train import configure_finetune_trainability, maybe_enable_mr_lora

        config.USE_MR_LORA = True
        config.MR_LORA_SCOPE = "graph_attention_routing"
        config.MR_LORA_RANK_GRAPH = 8
        config.MR_LORA_RANK_ROUTING = 4
        config.MR_LORA_ALPHA_GRAPH = 16.0
        config.MR_LORA_ALPHA_ROUTING = 8.0
        config.MR_LORA_DROPOUT = 0.05
        config.MR_LORA_TRAIN_OUTPUT_HEAD = False
        model = build_experiment_model()

        injection = maybe_enable_mr_lora(model)
        policy = configure_finetune_trainability(model)
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

        self.assertTrue(injection["enabled"])
        self.assertGreater(injection["graph_modules"], 0)
        self.assertGreater(injection["attention_modules"], 0)
        self.assertGreater(injection["routing_modules"], 0)
        lora_modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, LoRALinear)
        }
        self.assertEqual(len(lora_modules), injection["total_modules"])
        for name, module in lora_modules.items():
            with self.subTest(module=name):
                if any(name.endswith(target) for target in GRAPH_LORA_TARGETS):
                    self.assertEqual(module.rank, 8)
                    self.assertEqual(module.alpha, 16.0)
                elif any(name.endswith(target) for target in ATTENTION_LORA_TARGETS):
                    self.assertEqual(module.rank, 8)
                    self.assertEqual(module.alpha, 16.0)
                elif any(name.endswith(target) for target in ROUTING_LORA_TARGETS):
                    self.assertEqual(module.rank, 4)
                    self.assertEqual(module.alpha, 8.0)
                else:
                    self.fail(f"Unclassified LoRA target: {name}")
                self.assertIsInstance(module.dropout, torch.nn.Dropout)
                self.assertAlmostEqual(module.dropout.p, 0.05)
        self.assertTrue(policy["mr_lora"])
        self.assertGreater(len(trainable_names), 0)
        self.assertTrue(all(".lora_" in name for name in trainable_names))

    def test_mr_lora_initial_injection_preserves_pretrained_forward_output(self) -> None:
        from train import maybe_enable_mr_lora

        config.USE_MR_LORA = True
        config.MR_LORA_SCOPE = "graph_attention_routing"
        config.MR_LORA_RANK_GRAPH = 8
        config.MR_LORA_RANK_ROUTING = 4
        torch.manual_seed(123)
        model = build_experiment_model()
        model.eval()
        x = torch.randn(2, len(config.input_node_names(config.USE_EL_AS_INPUT)))
        with torch.no_grad():
            before = model(x)["mu"].detach().clone()

        maybe_enable_mr_lora(model)
        model.eval()
        with torch.no_grad():
            after = model(x)["mu"].detach().clone()

        self.assertTrue(torch.allclose(before, after, atol=1.0e-6, rtol=1.0e-6))

    def test_mr_lora_checkpoint_load_injects_adapters_before_state_dict(self) -> None:
        import tempfile

        from models.mr_lora import LoRALinear
        from train import load_checkpoint, maybe_enable_mr_lora

        config.USE_MR_LORA = True
        config.MR_LORA_SCOPE = "graph_attention_routing"
        source = build_experiment_model()
        maybe_enable_mr_lora(source)
        payload = {"model_state_dict": source.state_dict()}

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "mr_lora_model.pth"
            torch.save(payload, ckpt_path)
            target = build_experiment_model()

            load_checkpoint(target, ckpt_path, torch.device("cpu"))

        self.assertTrue(any(isinstance(module, LoRALinear) for module in target.modules()))
        self.assertTrue(any(".lora_" in name for name, _param in target.named_parameters()))

    def test_topk_sparse_weights_follow_reference_logits_softmax(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])

        sparse, topk_indices, topk_values = _topk_sparse_weights(logits, top_k=2)

        self.assertTrue(torch.equal(topk_indices, torch.tensor([[0, 1]])))
        self.assertTrue(torch.allclose(topk_values, torch.tensor([[2.0, 1.0]])))
        self.assertTrue(torch.allclose(sparse[0, :2], torch.softmax(torch.tensor([2.0, 1.0]), dim=0)))
        self.assertTrue(torch.allclose(sparse[0, 2:], torch.zeros(2)))

    def test_moe_aux_loss_matches_reference_importance_and_load(self) -> None:
        weights = torch.tensor(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 0.7, 0.3, 0.0],
            ]
        )
        uniform = torch.full((4,), 0.25)
        importance = weights.sum(dim=0)
        importance = importance / (importance.sum() + 1.0e-8)
        expected = ((importance - uniform) ** 2).mean() + ((weights.mean(dim=0) - uniform) ** 2).mean()

        self.assertTrue(torch.allclose(router_load_balance_loss(weights), expected))
        self.assertTrue(torch.allclose(moe_load_balance_loss([weights]), expected))


if __name__ == "__main__":
    unittest.main()
