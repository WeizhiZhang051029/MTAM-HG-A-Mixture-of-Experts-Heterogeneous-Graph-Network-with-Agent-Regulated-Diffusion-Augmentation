from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABDIFF_ROOT = PROJECT_ROOT / "third_party" / "TabDiff"
for path in (PROJECT_ROOT, TABDIFF_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    transformed = (raw - raw.mean(axis=0, keepdims=True)) / np.clip(
        raw.std(axis=0, keepdims=True), 1.0e-6, None
    )
    return CAPLMechanismConstraint(
        PAPER_COLUMNS,
        transformed,
        raw,
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


def test_temperature_path_maps_the_single_soaking_feature_to_both_endpoints() -> None:
    constraint = _paper_constraint()
    name_to_idx = {name: idx for idx, name in enumerate(PAPER_COLUMNS)}

    assert constraint.temperature_hold_tolerance == pytest.approx(10.0)
    assert constraint.temperature_paths == [
        (
            name_to_idx["HF_T"],
            name_to_idx["SF_T"],
            name_to_idx["SF_T"],
            name_to_idx["SC_T"],
        )
    ]

    physical = constraint.raw_mean.unsqueeze(0).clone()
    physical[:, name_to_idx["HF_T"]] = 820.0
    physical[:, name_to_idx["SF_T"]] = 800.0
    physical[:, name_to_idx["SC_T"]] = 815.0

    assert constraint.temperature_path_energy(physical).item() == pytest.approx(625.0)


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

    parts = constraint.energy_parts_from_physical(constraint.proxy_to_physical(transformed))
    expected = sum(parts.values())

    assert torch.allclose(constraint(transformed, reduce=False), expected)
    assert torch.isfinite(expected).all()


def test_missing_paper_temperature_column_fails_fast() -> None:
    columns = [name for name in PAPER_COLUMNS if name != "SC_T"]
    raw = np.ones((8, len(columns)), dtype=np.float32)

    with pytest.raises(ValueError, match="slow-cooling-furnace"):
        CAPLMechanismConstraint(columns, raw, raw)
