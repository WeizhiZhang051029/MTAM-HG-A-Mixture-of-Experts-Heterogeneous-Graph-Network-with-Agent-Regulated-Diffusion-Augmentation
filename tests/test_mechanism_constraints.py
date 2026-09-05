from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import QuantileTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABDIFF_ROOT = PROJECT_ROOT / "third_party" / "TabDiff"
for path in (PROJECT_ROOT, TABDIFF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tabdiff.trainer as trainer_module  # noqa: E402
from tabdiff.main import validate_trainable_scope  # noqa: E402
from tabdiff.models.capl_mechanism import (  # noqa: E402
    CAPLMechanismConstraint,
    CAPLMechanismWeights,
)

PAPER_COLUMNS = [
    "屈服强度",
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
]


def _paper_constraint(*, yield_tolerance: float = 5.0) -> CAPLMechanismConstraint:
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(64, len(PAPER_COLUMNS))).astype(np.float32)
    raw[:, 0] = rng.normal(150.0, 20.0, len(raw))
    raw[:, PAPER_COLUMNS.index("HF_T")] = rng.normal(810.0, 8.0, len(raw))
    raw[:, PAPER_COLUMNS.index("SF_T")] = rng.normal(815.0, 8.0, len(raw))
    raw[:, PAPER_COLUMNS.index("SC_T")] = rng.normal(650.0, 8.0, len(raw))
    transformer = QuantileTransformer(
        n_quantiles=len(raw),
        output_distribution="normal",
        random_state=0,
    )
    transformed = transformer.fit_transform(raw).astype(np.float32)
    return CAPLMechanismConstraint(
        PAPER_COLUMNS,
        transformed,
        raw,
        quantile_values=transformer.quantiles_,
        quantile_references=transformer.references_,
        low_quantile=0.0,
        high_quantile=1.0,
        margin=0.0,
        yield_tolerance=yield_tolerance,
    )


def test_mechanism_exposes_exactly_the_three_paper_terms() -> None:
    constraint = _paper_constraint()
    physical = constraint.raw_mean.unsqueeze(0)

    parts = constraint.energy_parts_from_physical(physical)

    assert list(parts) == ["temperature_path", "production_window", "yield_residual"]
    assert [field.name for field in fields(CAPLMechanismWeights)] == [
        "temperature_path",
        "production_window",
        "yield_residual",
    ]


def test_temperature_path_uses_four_consecutive_furnace_measurements() -> None:
    constraint = _paper_constraint()
    name_to_idx = {name: idx for idx, name in enumerate(PAPER_COLUMNS)}

    assert constraint.temperature_hold_tolerance == pytest.approx(10.0)
    assert constraint.temperature_paths == [
        (
            name_to_idx["JPF_PT"],
            name_to_idx["HF_T"],
            name_to_idx["SF_T"],
            name_to_idx["SC_T"],
        )
    ]
    assert CAPLMechanismConstraint.TEMPERATURE_PATH_COLUMNS == (
        "JPF_PT",
        "HF_T",
        "SF_T",
        "SC_T",
    )
    assert len(set(constraint.temperature_paths[0])) == 4

    physical = constraint.raw_mean.unsqueeze(0).clone()
    physical[:, name_to_idx["JPF_PT"]] = 820.0
    physical[:, name_to_idx["HF_T"]] = 800.0
    physical[:, name_to_idx["SF_T"]] = 815.0
    physical[:, name_to_idx["SC_T"]] = 825.0

    assert constraint.temperature_path_energy(physical).item() == pytest.approx(525.0)


def test_production_window_excludes_generated_yield_strength() -> None:
    constraint = _paper_constraint()
    physical = constraint.raw_mean.unsqueeze(0).clone()
    physical[:, 1:] = (constraint.window_low + constraint.window_high) / 2.0

    base = constraint.production_window_energy(physical)
    physical[:, 0] = 1.0e9
    changed_y = constraint.production_window_energy(physical)

    assert constraint.window_low.numel() == 21
    assert torch.allclose(base, torch.zeros_like(base))
    assert torch.allclose(changed_y, base)


def test_production_window_matches_feature_only_relu_formula() -> None:
    constraint = _paper_constraint()
    physical = constraint.raw_mean.unsqueeze(0).clone()
    physical[:, 1:] = (constraint.window_low + constraint.window_high) / 2.0
    physical[:, 1] = constraint.window_low[0] - 2.0 * constraint.window_spread[0]

    energy = constraint.production_window_energy(physical)

    features = physical[:, 1:]
    below = torch.relu(constraint.window_low - features) / constraint.window_spread
    above = torch.relu(features - constraint.window_high) / constraint.window_spread
    expected = (below + above).pow(2).mean(dim=1)
    assert torch.allclose(energy, expected)


def test_yield_energy_matches_tolerant_empirical_linear_reference() -> None:
    constraint = _paper_constraint(yield_tolerance=5.0)
    physical = constraint.raw_mean.unsqueeze(0).clone()
    with torch.no_grad():
        constraint.yield_coef.zero_()
        constraint.yield_intercept.fill_(150.0)
        constraint.yield_std.fill_(20.0)
    physical[:, 0] = 160.0

    energy = constraint.yield_residual_energy(physical)

    assert energy.item() == pytest.approx(0.0625)


def test_forward_is_weighted_sum_of_only_the_three_terms() -> None:
    constraint = _paper_constraint()
    transformed = torch.zeros(4, len(PAPER_COLUMNS))

    parts = constraint.energy_parts_from_physical(constraint.to_physical(transformed))
    expected = sum(parts.values())

    assert torch.allclose(constraint(transformed, reduce=False), expected)
    assert torch.isfinite(expected).all()


def test_quantile_inverse_matches_the_fitted_sklearn_transformer_and_is_differentiable() -> None:
    rng = np.random.default_rng(23)
    raw = rng.lognormal(mean=2.0, sigma=0.7, size=(96, len(PAPER_COLUMNS))).astype(np.float32)
    transformer = QuantileTransformer(
        n_quantiles=32,
        output_distribution="normal",
        random_state=0,
    )
    transformed_train = transformer.fit_transform(raw).astype(np.float32)
    constraint = CAPLMechanismConstraint(
        PAPER_COLUMNS,
        transformed_train,
        raw,
        quantile_values=transformer.quantiles_,
        quantile_references=transformer.references_,
    )
    query = rng.normal(0.0, 0.8, size=(7, len(PAPER_COLUMNS))).astype(np.float32)
    query[0] = -8.0
    query[-1] = 8.0
    transformed = torch.tensor(query, dtype=torch.float32, requires_grad=True)

    physical = constraint.to_physical(transformed)
    expected = transformer.inverse_transform(transformed.detach().numpy())

    np.testing.assert_allclose(physical.detach().numpy(), expected, rtol=2.0e-5, atol=2.0e-5)
    physical.sum().backward()
    assert transformed.grad is not None
    assert torch.isfinite(transformed.grad).all()
    assert torch.count_nonzero(transformed.grad).item() > 0


def test_missing_paper_temperature_column_fails_fast() -> None:
    columns = [name for name in PAPER_COLUMNS if name != "SC_T"]
    raw = np.ones((8, len(columns)), dtype=np.float32)

    with pytest.raises(ValueError, match="slow-cooling-furnace"):
        references = np.linspace(0.0, 1.0, len(raw), dtype=np.float32)
        CAPLMechanismConstraint(
            columns,
            raw,
            raw,
            quantile_values=np.sort(raw, axis=0),
            quantile_references=references,
        )


def test_partial_tabdiff_scope_requires_a_pretrained_checkpoint() -> None:
    with pytest.raises(ValueError, match="pretrained checkpoint"):
        validate_trainable_scope("mlp_detokenizer", None)

    validate_trainable_scope("all", None)
    validate_trainable_scope("mlp_detokenizer", "base.pt")


def test_finetune_checkpoint_initializes_model_and_ema_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Diffusion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._denoise_fn = torch.nn.Linear(2, 2)
            self.num_schedule = torch.nn.Linear(1, 1)
            self.cat_schedule = torch.nn.Linear(1, 1)

    source = Diffusion()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(3.0)
    checkpoint = tmp_path / "base.pt"
    torch.save(
        {
            "denoise_fn": source._denoise_fn.state_dict(),
            "num_schedule": source.num_schedule.state_dict(),
            "cat_schedule": source.cat_schedule.state_dict(),
        },
        checkpoint,
    )
    monkeypatch.setattr(trainer_module, "ReduceLROnPlateau", lambda *_args, **_kwargs: object())

    trainer = trainer_module.Trainer(
        Diffusion(),
        train_iter=[],
        dataset=None,
        test_dataset=None,
        metrics=None,
        logger=None,
        lr=1.0e-4,
        weight_decay=0.0,
        steps=500,
        batch_size=8,
        check_val_every=500,
        sample_batch_size=8,
        model_save_path=str(tmp_path / "ckpt"),
        result_save_path=str(tmp_path / "result"),
        ckpt_path=checkpoint,
        device=torch.device("cpu"),
        reset_train_epoch=True,
    )

    for loaded, ema in (
        (trainer.diffusion._denoise_fn, trainer.ema_model),
        (trainer.diffusion.num_schedule, trainer.ema_num_schedule),
        (trainer.diffusion.cat_schedule, trainer.ema_cat_schedule),
    ):
        for loaded_parameter, ema_parameter in zip(loaded.parameters(), ema.parameters()):
            assert torch.all(loaded_parameter == 3.0)
            assert torch.equal(loaded_parameter, ema_parameter)

    from types import SimpleNamespace

    (tmp_path / "ckpt").mkdir()
    trainer.steps = 3
    trainer.train_iter = [torch.ones(2, 2) for _ in range(4)]
    trainer.dataset = SimpleNamespace(d_numerical=0, categories=[])
    trainer.logger = SimpleNamespace(define_metric=lambda *a, **kw: None, log=lambda *a, **kw: None)
    trainer.lr_scheduler = "fixed"
    calls = []
    def step(*args):
        calls.append(1)
        return torch.tensor(0.0), torch.tensor(1.0)
    monkeypatch.setattr(trainer, "_run_step", step)
    monkeypatch.setattr(trainer, "compute_loss", lambda: (0.0, 1.0))
    trainer.run_loop()
    assert len(calls) == trainer.optimizer_steps == 3


def test_sampling_guidance_is_independent_of_batch_size() -> None:
    from types import SimpleNamespace

    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion

    def constraint(x, reduce=False):
        return x.square().sum(dim=1)
    diffusion = SimpleNamespace(
        num_numerical_features=3,
        mechanism_constraint=constraint,
        guidance_scale=0.05,
        num_timesteps=50,
    )
    sample = torch.tensor([[1.0, 2.0, 3.0]])
    one = UnifiedCtimeDiffusion.apply_mechanism_guidance(diffusion, sample, 49)
    batch = UnifiedCtimeDiffusion.apply_mechanism_guidance(diffusion, sample.repeat(8, 1), 49)
    torch.testing.assert_close(one, sample * 0.9)
    torch.testing.assert_close(batch, one.repeat(8, 1))


def test_sampling_does_not_generate_unused_rows() -> None:
    from types import SimpleNamespace

    from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion

    calls = []
    def sample(n):
        calls.append(n)
        return torch.ones(n, 3)
    diffusion = SimpleNamespace(sample=sample)
    result = UnifiedCtimeDiffusion.sample_all(diffusion, 10, 6)
    assert calls == [6, 4]
    assert result.shape == (10, 3)
