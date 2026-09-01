from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import config
import train as train_module
from dataset import CAPLDataset
from losses import total_loss
from training.cbtg import (
    AGENT_FEEDBACK_FEATURE_NAMES,
    PAPER_CBTG_LAMBDA_H,
    PAPER_CBTG_LAMBDA_M,
    PAPER_CBTG_TARGET_CONFIDENCE,
    DynamicSyntheticState,
    SyntheticBundle,
    TrainTensorBundle,
    _config_sha256,
    _paper_oriented_metrics,
    _protocol_metrics,
    _synthetic_step,
    build_dynamic_synthetic_state,
    build_paper_cbtg_feedback,
    build_synthetic_pretrain_optimizer,
    calibrate_quality_agent_on_real,
    compute_dynamic_synthetic_weights,
    dynamic_synthetic_refresh_due,
    evaluate_validation_pretrain_feedback,
    load_cross_run_validation_stats,
    paper_cbtg_agent_loss,
    quality_agent_updates_enabled,
    select_top_synthetic_indices,
)
from training.clusters import WorkingConditionCluster


def _state(n: int, clusters: np.ndarray | None = None) -> DynamicSyntheticState:
    return DynamicSyntheticState(
        weights=np.ones(n, dtype=np.float64),
        selected_indices=np.arange(n, dtype=np.int64),
        selected_mask=np.ones(n, dtype=bool),
        previous_raw_weights=np.ones(n, dtype=np.float64),
        scarcity_bonus=np.zeros(n, dtype=np.float64),
        bin_ids=np.zeros(n, dtype=np.int64),
        bin_edges=np.array([0.0, 1.0], dtype=np.float64),
        train_bin_counts=np.array([n], dtype=np.int64),
        feedback_features=np.zeros((n, len(AGENT_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        feedback_target=np.zeros(n, dtype=np.float64),
        cluster_ids=clusters,
    )


def test_protocol_metrics_bind_split_source_and_synthetic_fingerprints() -> None:
    data = SimpleNamespace(
        split_seed=42,
        split_method="stratified_random",
        combined_split_hash="a" * 64,
        source_sha256="b" * 64,
    )
    synthetic = SimpleNamespace(
        synthetic_sha256="c" * 64,
        provenance_sha256="e" * 64,
    )
    original_seed = getattr(config, "TABDIFF_GENERATION_SEED", 0)
    original_config_hash = getattr(config, "CONFIG_SHA256", "")
    config.TABDIFF_GENERATION_SEED = 0
    config.CONFIG_SHA256 = "d" * 64
    try:
        metrics = _protocol_metrics(data, synthetic, require_config=True)
    finally:
        config.TABDIFF_GENERATION_SEED = original_seed
        config.CONFIG_SHA256 = original_config_hash

    assert metrics == {
        "Split_Seed": 42,
        "Split_Method": "stratified_random",
        "Combined_Split_SHA256": "a" * 64,
        "Source_Data_SHA256": "b" * 64,
        "Synthetic_SHA256": "c" * 64,
        "Synthetic_Provenance_SHA256": "e" * 64,
        "Generation_Seed": 0,
        "Config_SHA256": "d" * 64,
    }


def test_formal_protocol_rejects_missing_config_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CONFIG_SHA256", "", raising=False)

    with pytest.raises(RuntimeError, match="stable hashed YAML"):
        _config_sha256(required=True)


def test_five_metrics_use_lower_is_better_orientation() -> None:
    y = np.array([1.0, 2.0, 4.0, 8.0])
    oriented = _paper_oriented_metrics(y, y, (2.0, 4.0))

    assert oriented.shape == (5,)
    assert np.allclose(oriented, 0.0, atol=1.0e-10)


def test_feedback_state_has_four_standardized_values_per_metric_and_clipped_reward() -> None:
    overall = np.array([6.0, 4.4, 2.8, 0.09, 3.4])
    run_std = np.array([0.24, 0.21, 0.18, 0.008, 0.49])
    per_cluster = np.array(
        [
            [5.0, 3.8, 2.5, 0.07, 2.7],
            [8.0, 5.9, 4.0, 0.17, 6.2],
        ]
    )

    result = build_paper_cbtg_feedback(
        overall,
        run_std,
        per_cluster,
        np.array([0, 1, 1], dtype=np.int64),
    )

    assert len(AGENT_FEEDBACK_FEATURE_NAMES) == 5 * 4
    assert result["raw_state"].shape == (3, 5, 4)
    assert result["features"].shape == (3, 20)
    assert np.allclose(result["raw_state"][0, :, 0], overall)
    assert np.allclose(result["raw_state"][0, :, 1], run_std)
    assert np.allclose(result["raw_state"][0, :, 2], per_cluster[0])
    assert np.allclose(result["raw_state"][1, :, 2], per_cluster[1])
    assert np.isfinite(result["features"]).all()
    assert np.all(result["target"] >= -1.0)
    assert np.all(result["target"] <= 1.0)


def test_contextual_bandit_loss_matches_paper_equation() -> None:
    confidence = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    reward = torch.tensor([-0.5, 1.0], dtype=torch.float64)

    loss, parts = paper_cbtg_agent_loss(confidence, reward)
    entropy = -(
        confidence * torch.log(confidence)
        + (1.0 - confidence) * torch.log(1.0 - confidence)
    ).mean()
    expected = (
        -(confidence * reward).mean()
        + PAPER_CBTG_LAMBDA_M * (confidence.mean() - PAPER_CBTG_TARGET_CONFIDENCE) ** 2
        - PAPER_CBTG_LAMBDA_H * entropy
    )

    assert torch.allclose(loss, expected)
    assert torch.allclose(parts["confidence_entropy"], entropy)
    loss.backward()
    assert confidence.grad is not None


def test_retention_score_is_confidence_times_process_times_mechanism_times_bonus() -> None:
    confidence = np.array([0.2, 0.5, 0.9, 0.7, 0.6], dtype=np.float64)
    process = np.array([1.0, 0.8, 0.5, 0.9, 1.0], dtype=np.float64)
    mechanism = np.array([0.5, 1.0, 0.8, 0.7, 1.0], dtype=np.float64)
    scarcity = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)

    result = compute_dynamic_synthetic_weights(
        np.ones(5),
        confidence,
        np.zeros(5),
        np.zeros(5),
        process,
        mechanism,
        scarcity,
    )
    expected = confidence * process * mechanism * (1.0 + scarcity)
    selected, _ = select_top_synthetic_indices(result["selection_score"], top_ratio=0.60)

    assert np.allclose(result["selection_score"], expected)
    assert len(selected) == 3


def test_initial_warmup_keeps_all_candidates_instead_of_first_sixty_percent() -> None:
    n = 10
    dataset = CAPLDataset(
        np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        np.arange(n, dtype=np.float32).reshape(n, 1),
        np.arange(n),
    )
    synthetic = SimpleNamespace(
        frame=pd.DataFrame(index=np.arange(n)),
        y_raw=np.arange(n, dtype=np.float64),
        loader=DataLoader(dataset, batch_size=n, shuffle=False),
    )
    data = SimpleNamespace(y_train_raw=np.arange(n, dtype=np.float64))

    state = build_dynamic_synthetic_state(synthetic, data, np.zeros(5, dtype=np.float64))

    assert state.selected_indices.tolist() == list(range(n))
    assert state.selected_mask.all()


class _IdentityScaler:
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values)


class _ScaleModel(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mu": self.scale * x[:, :1]}


class _TwoClusters:
    is_fitted = True
    effective_n_clusters = 2

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x)[:, 0] >= 5.0).astype(np.int64)

    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.predict(x.detach().cpu().numpy())).long()


def test_dynamic_feedback_uses_injected_cross_run_validation_std() -> None:
    x = np.array([[1.0], [2.0], [10.0], [20.0]], dtype=np.float32)
    dataset = CAPLDataset(x, x.copy(), np.arange(4))
    data = SimpleNamespace(
        val_loader=DataLoader(dataset, batch_size=2, shuffle=False),
        y_scaler=_IdentityScaler(),
        tail_thresholds=(2.0, 10.0),
    )
    model = _ScaleModel(1.0)
    run_std = np.array([0.2, 0.1, 0.3, 0.01, 0.4], dtype=np.float64)
    original_standardize = config.STANDARDIZE_Y
    config.STANDARDIZE_Y = True
    try:
        first = evaluate_validation_pretrain_feedback(
            model, data, torch.device("cpu"), run_std, cluster_model=_TwoClusters()
        )
        model.scale.data.fill_(0.5)
        second = evaluate_validation_pretrain_feedback(
            model, data, torch.device("cpu"), run_std, cluster_model=_TwoClusters()
        )
    finally:
        config.STANDARDIZE_Y = original_standardize

    assert np.allclose(first["overall_metrics"], 0.0, atol=1.0e-10)
    assert np.allclose(first["run_std"], run_std)
    assert np.allclose(second["run_std"], run_std)
    assert np.any(np.asarray(second["overall_metrics"]) > 0.0)


def test_cross_run_validation_stats_require_independent_matching_split(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "validation_std.json"
    run_metrics = [
        [float(seed), float(seed + 1), float(seed + 2), seed / 100.0, float(seed + 3)]
        for seed in range(10)
    ]
    path.write_text(
        json.dumps(
            {
                "format": "mtam_hg_cbtg_cross_run_v1",
                "source": "independent_validation_runs",
                "partition": "validation",
                "ddof": 1,
                "metric_names": ["RMSE", "MAE", "MAPE", "ONE_MINUS_R2", "TAIL_MAE"],
                "validation_metric_std": np.std(run_metrics, axis=0, ddof=1).tolist(),
                "num_runs": 10,
                "combined_split_sha256": "a" * 64,
                "source_data_sha256": "c" * 64,
                "config_sha256": "b" * 64,
                "producer_protocol_sha256": "d" * 64,
                "runs": [
                    {
                        "metrics_artifact_sha256": f"{index:064x}",
                        "validation_metrics": metrics,
                    }
                    for index, metrics in enumerate(run_metrics, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_SHA256", "b" * 64)
    stats = load_cross_run_validation_stats(
        SimpleNamespace(combined_split_hash="a" * 64, source_sha256="c" * 64),
        path,
    )

    assert stats.num_runs == 10
    assert np.allclose(stats.metric_std, np.std(run_metrics, axis=0, ddof=1))

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"] = "refresh_epochs"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="independent validation runs"):
        load_cross_run_validation_stats(
            SimpleNamespace(combined_split_hash="a" * 64, source_sha256="c" * 64),
            path,
        )


def test_synthetic_pretrain_optimizer_uses_one_paper_learning_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "LR", 1.0e-3)
    monkeypatch.setattr(config, "SYNTHETIC_AGENT_LR", 1.0e-3)
    optimizer = build_synthetic_pretrain_optimizer(_ScaleModel(), _QualityAgent())

    assert {group["lr"] for group in optimizer.param_groups} == {1.0e-3}

    monkeypatch.setattr(config, "SYNTHETIC_AGENT_LR", 5.0e-4)
    with pytest.raises(ValueError, match="same pretraining learning rate"):
        build_synthetic_pretrain_optimizer(_ScaleModel(), _QualityAgent())


def test_agent_epoch_configuration_controls_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SYNTHETIC_AGENT_EPOCHS", 100)

    assert quality_agent_updates_enabled(100)
    assert not quality_agent_updates_enabled(101)


def test_dynamic_refresh_honors_five_complete_warmup_epochs() -> None:
    assert not any(dynamic_synthetic_refresh_due(epoch, 5, 5) for epoch in range(1, 6))
    assert dynamic_synthetic_refresh_due(6, 5, 5)
    assert not dynamic_synthetic_refresh_due(10, 5, 5)
    assert dynamic_synthetic_refresh_due(11, 5, 5)


def test_cbtg_real_calibration_disables_batch_error_router_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    y = torch.tensor([[1.0], [2.0]])
    mu = torch.tensor([[0.5], [2.5]], requires_grad=True)
    outputs = {
        "mu": mu,
        "expert_preds": torch.stack((mu - 0.1, mu + 0.1), dim=1),
        "gate_probs": torch.full((2, 2), 0.5),
        "sample_confidence": torch.full((2, 1), 0.6, requires_grad=True),
        "aux_loss": mu.new_tensor(0.0),
    }
    monkeypatch.setattr(config, "USE_AGENT_REWARD", True)
    monkeypatch.setattr(config, "USE_LAPLACE", False)
    monkeypatch.setattr(config, "EXPERT_CALIBRATION_LAMBDA", 0.0)
    monkeypatch.setattr(config, "EXPERT_DIVERSITY_LAMBDA", 0.0)
    monkeypatch.setattr(config, "AGENT_USE_SAMPLE_WEIGHT_FOR_SUPERVISED_LOSS", True)
    external_weights = torch.tensor([[1.0], [0.0]])

    loss, logs = total_loss(
        outputs,
        y,
        batch_weights=external_weights,
        use_internal_agent_weight=False,
        use_agent_reward=False,
    )
    loss.backward()

    assert logs["agent_reward_loss"] == 0.0
    assert logs["pred_loss"] == pytest.approx(0.125)
    assert logs["batch_weight_mean"] == pytest.approx(0.5)
    assert outputs["sample_confidence"].grad is None


def test_real_domain_calibration_updates_quality_agent_without_test_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.array([[1.0], [2.0], [10.0], [20.0]], dtype=np.float32)
    dataset = CAPLDataset(x, x.copy(), np.arange(4))
    data = SimpleNamespace(
        train_loader=DataLoader(dataset, batch_size=2, shuffle=False),
        val_loader=DataLoader(dataset, batch_size=2, shuffle=False),
        y_scaler=_IdentityScaler(),
        tail_thresholds=(2.0, 10.0),
    )
    model = _ScaleModel(0.5)
    agent = _QualityAgent()
    optimizer = AdamW(agent.parameters(), lr=1.0e-3)
    before = agent.logit.detach().clone()
    monkeypatch.setattr(config, "STANDARDIZE_Y", True)

    logs, batch_weight_fn = calibrate_quality_agent_on_real(
        model,
        agent,
        data,
        torch.device("cpu"),
        1,
        optimizer=optimizer,
        cross_run_validation_std=np.array([0.2, 0.1, 0.3, 0.01, 0.4]),
        cluster_model=_TwoClusters(),
    )

    assert np.isfinite(logs["real_cbtg_agent_loss"])
    assert logs["real_cbtg_validation_tail_mae"] > 0.0
    assert not torch.equal(before, agent.logit.detach())
    weights = batch_weight_fn(dataset.x[:2], dataset.y[:2])
    assert weights.shape == (2, 1)
    assert not weights.requires_grad


def test_real_finetune_calibrates_agent_before_weighted_model_epoch(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    model = _ScaleModel(0.5)
    agent = _QualityAgent()
    dataset = CAPLDataset(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.arange(2),
    )
    data = SimpleNamespace(
        train_loader=DataLoader(dataset, batch_size=2, shuffle=False),
        val_loader=DataLoader(dataset, batch_size=2, shuffle=False),
    )

    def calibrate(*_args):
        events.append("calibrate")

        def weights(x: torch.Tensor, _y: torch.Tensor) -> torch.Tensor:
            events.append("weight")
            return torch.ones((len(x), 1), device=x.device)

        return {"real_cbtg_agent_loss": 0.1}, weights

    def train_epoch(*_args, **kwargs):
        assert kwargs["use_internal_agent_weight"] is False
        assert kwargs["use_agent_reward"] is False
        kwargs["batch_weight_fn"](dataset.x, dataset.y)
        events.append("train")
        return {"total_loss": 0.2, "pred_loss": 0.2}

    monkeypatch.setattr(train_module, "save_split_artifacts", lambda *_args: None)
    monkeypatch.setattr(train_module, "maybe_enable_mr_lora", lambda *_args, **_kwargs: {"enabled": False})
    monkeypatch.setattr(
        train_module,
        "configure_finetune_trainability",
        lambda *_args, **_kwargs: {
            "freeze_backbone": False,
            "trainable_params": 1,
            "frozen_params": 0,
        },
    )
    monkeypatch.setattr(train_module, "save_json", lambda *_args: None)
    monkeypatch.setattr(
        train_module,
        "build_finetune_optimizer",
        lambda current_model: AdamW(current_model.parameters(), lr=1.0e-3),
    )
    monkeypatch.setattr(train_module, "train_one_epoch", train_epoch)
    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        lambda *_args: ({"RMSE": 1.0, "MAE": 1.0, "TAIL_MAE": 1.0}, {}),
    )
    monkeypatch.setattr(train_module, "evaluate_prediction_loss", lambda *_args: 1.0)
    monkeypatch.setattr(train_module, "append_csv", lambda *_args: None)
    monkeypatch.setattr(train_module, "checkpoint_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(config, "USE_CLUSTER_BALANCE_REWARD", False)
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULT_DIR", tmp_path)

    train_module.supervised_finetune(
        model,
        data,
        torch.device("cpu"),
        epochs=1,
        quality_agent=agent,
        quality_agent_calibration_fn=calibrate,
    )

    assert events == ["calibrate", "weight", "train"]


def test_working_condition_labels_are_ordered_by_training_mean_yield_strength() -> None:
    x_train = np.array(
        [[-10.0], [-9.5], [-9.0], [0.0], [0.5], [1.0], [10.0], [10.5], [11.0]],
        dtype=np.float32,
    )
    y_train = np.array(
        [300.0, 302.0, 304.0, 100.0, 102.0, 104.0, 200.0, 202.0, 204.0],
        dtype=np.float64,
    )
    clusterer = WorkingConditionCluster(n_clusters=3, random_state=7).fit(x_train, y_train)
    labels = clusterer.predict(x_train)
    ordered_means = np.array([y_train[labels == g].mean() for g in range(3)])

    assert np.all(np.diff(ordered_means) > 0.0)
    assert np.allclose(clusterer.cluster_mean_y_, ordered_means)


class _PaperLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu = self.weight * x[:, :1]
        experts = torch.stack((mu - 0.1, mu + 0.1), dim=1)
        gate = torch.full((len(x), 2), 0.5, dtype=x.dtype, device=x.device)
        a0 = torch.zeros((2, 2), dtype=x.dtype, device=x.device)
        return {
            "mu": mu,
            "expert_preds": experts,
            "gate_probs": gate,
            "aux_loss": 0.02 * self.weight.square(),
            "diversity_loss": 0.01 * self.weight.square(),
            "A0": a0,
            "A_kg": self.weight * torch.ones_like(a0),
        }


class _QualityAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.sigmoid(self.logit).expand(len(x), 1)


def test_synthetic_step_uses_all_six_paper_loss_terms() -> None:
    n = 4
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[1.2], [2.1], [2.9], [4.2]])
    model = _PaperLossModel()
    agent = _QualityAgent()
    optimizer = AdamW([*model.parameters(), *agent.parameters()], lr=1.0e-3)
    synthetic = SyntheticBundle(
        loader=DataLoader(CAPLDataset(x.numpy(), y.numpy(), np.arange(n)), batch_size=n),
        frame=pd.DataFrame(index=np.arange(n)),
        y_raw=y.numpy().reshape(-1),
        is_tail=np.zeros(n, dtype=np.float64),
        synthetic_source=np.full(n, "test"),
        generation_condition=np.full(n, "test"),
        process_consistency=np.ones(n),
        range_score=np.ones(n),
        manifold_score=np.ones(n),
        label_consistency_score=np.ones(n),
        mechanism_consistency=np.ones(n),
        nearest_train_distance=np.zeros(n),
        nearest_train_index=np.arange(n),
        nearest_train_y_raw=y.numpy().reshape(-1),
    )
    train = TrainTensorBundle(x=x, y=y, y_raw=y)
    state = _state(n)
    state.feedback_target = np.array([-1.0, -0.5, 0.5, 1.0])
    names = (
        "USE_LAPLACE",
        "MOE_AUX_LAMBDA",
        "EXPERT_CALIBRATION_LAMBDA",
        "EXPERT_DIVERSITY_LAMBDA",
        "LAMBDA_GRAPH",
        "AGENT_REWARD_LAMBDA",
        "SYNTHETIC_USE_REWARD_LOSS",
        "DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT",
    )
    old = {name: getattr(config, name) for name in names}
    updates = {
        "USE_LAPLACE": False,
        "MOE_AUX_LAMBDA": 0.2,
        "EXPERT_CALIBRATION_LAMBDA": 0.03,
        "EXPERT_DIVERSITY_LAMBDA": 0.001,
        "LAMBDA_GRAPH": 0.1,
        "AGENT_REWARD_LAMBDA": 0.01,
        "SYNTHETIC_USE_REWARD_LOSS": True,
        "DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT": True,
    }
    try:
        for name, value in updates.items():
            setattr(config, name, value)
        logs = _synthetic_step(
            model,
            agent,
            (x, y, torch.arange(n)),
            optimizer,
            synthetic,
            train,
            torch.device("cpu"),
            dynamic_state=state,
        )
    finally:
        for name, value in old.items():
            setattr(config, name, value)

    assert logs["synthetic_pred_loss"] > 0.0
    assert logs["synthetic_moe_aux_loss"] > 0.0
    assert logs["synthetic_expert_calibration_loss"] > 0.0
    assert logs["synthetic_expert_diversity_loss"] > 0.0
    assert logs["synthetic_graph_loss"] > 0.0
    assert np.isfinite(logs["synthetic_agent_total_loss"])
