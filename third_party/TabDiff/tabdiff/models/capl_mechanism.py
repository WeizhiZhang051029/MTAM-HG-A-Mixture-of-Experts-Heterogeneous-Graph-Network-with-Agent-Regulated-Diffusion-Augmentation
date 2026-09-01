from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CAPLMechanismWeights:
    """Weights of the three MP-TabDiff mechanism terms in Eq. (5)."""

    temperature_path: float = 1.0
    production_window: float = 1.0
    yield_residual: float = 1.0


class CAPLMechanismConstraint(nn.Module):
    """Mechanism energy evaluated in CAPL physical units."""

    QUANTILE_BOUND_THRESHOLD = 1.0e-7
    TEMPERATURE_PATH_COLUMNS = ("JPF_PT", "HF_T", "SF_T", "SC_T")

    def __init__(
        self,
        column_names: list[str],
        transformed_train: np.ndarray,
        raw_train: np.ndarray,
        *,
        quantile_values: np.ndarray,
        quantile_references: np.ndarray,
        low_quantile: float = 0.01,
        high_quantile: float = 0.99,
        margin: float = 0.10,
        ridge_alpha: float = 1.0e-3,
        temperature_hold_tolerance: float = 10.0,
        yield_tolerance: float = 0.0,
        weights: CAPLMechanismWeights | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        transformed = np.asarray(transformed_train, dtype=np.float32)
        raw = np.asarray(raw_train, dtype=np.float32)
        if transformed.ndim != 2 or raw.ndim != 2:
            raise ValueError("transformed_train and raw_train must be two-dimensional arrays.")
        if transformed.shape != raw.shape:
            raise ValueError(
                "transformed_train and raw_train must have the same shape: "
                f"{transformed.shape} != {raw.shape}"
            )
        if transformed.shape[1] != len(column_names):
            raise ValueError("column_names length must match numerical feature dimension.")
        if transformed.shape[1] < 2:
            raise ValueError("The joint MP-TabDiff vector must contain yield plus CAPL features.")
        quantiles = np.asarray(quantile_values, dtype=np.float32)
        references = np.asarray(quantile_references, dtype=np.float32)
        if quantiles.ndim != 2 or quantiles.shape[1] != transformed.shape[1]:
            raise ValueError("quantile_values must match the joint numerical feature dimension.")
        if references.ndim != 1 or references.shape[0] != quantiles.shape[0]:
            raise ValueError("quantile_references must match the quantile grid.")
        if references.shape[0] < 2 or np.any(np.diff(references) <= 0.0):
            raise ValueError("quantile_references must be strictly increasing.")
        if not np.isclose(references[0], 0.0) or not np.isclose(references[-1], 1.0):
            raise ValueError("quantile_references must span [0, 1].")
        if np.any(np.diff(quantiles, axis=0) < 0.0):
            raise ValueError("quantile_values must be non-decreasing by feature.")
        if not 0.0 <= float(low_quantile) < float(high_quantile) <= 1.0:
            raise ValueError("Expected 0 <= low_quantile < high_quantile <= 1.")
        if float(margin) < 0.0:
            raise ValueError("margin must be non-negative.")
        if float(temperature_hold_tolerance) < 0.0:
            raise ValueError("temperature_hold_tolerance must be non-negative.")
        if float(yield_tolerance) < 0.0:
            raise ValueError("yield_tolerance must be non-negative.")

        self.column_names = list(column_names)
        self.enabled = bool(enabled)
        self.weights = weights or CAPLMechanismWeights()
        self.ridge_alpha = float(ridge_alpha)
        self.temperature_hold_tolerance = float(temperature_hold_tolerance)
        self.yield_tolerance = float(yield_tolerance)

        transformed_mean = np.nanmean(transformed, axis=0)
        transformed_std = np.nanstd(transformed, axis=0)
        raw_mean = np.nanmean(raw, axis=0)
        transformed_std = np.where(transformed_std < 1.0e-6, 1.0, transformed_std)

        # Eq. (7) is defined only over x, never over the generated target y.
        raw_features = raw[:, 1:]
        low = np.nanquantile(raw_features, low_quantile, axis=0).astype(np.float32)
        high = np.nanquantile(raw_features, high_quantile, axis=0).astype(np.float32)
        spread = np.maximum(high - low, 1.0e-6).astype(np.float32)
        low = low - float(margin) * spread
        high = high + float(margin) * spread

        self.register_buffer("transformed_mean", torch.tensor(transformed_mean, dtype=torch.float32))
        self.register_buffer("transformed_std", torch.tensor(transformed_std, dtype=torch.float32))
        self.register_buffer("raw_mean", torch.tensor(raw_mean, dtype=torch.float32))
        self.register_buffer("quantile_values", torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer("quantile_references", torch.tensor(references, dtype=torch.float32))
        self.register_buffer("window_low", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("window_high", torch.tensor(high, dtype=torch.float32))
        self.register_buffer("window_spread", torch.tensor(spread, dtype=torch.float32))

        coef, intercept = self._fit_ridge_yield_proxy(raw, ridge_alpha)
        self.register_buffer("yield_coef", torch.tensor(coef, dtype=torch.float32))
        self.register_buffer("yield_intercept", torch.tensor([intercept], dtype=torch.float32))
        yield_scale = max(float(np.nanstd(raw[:, 0])), 1.0e-6)
        self.register_buffer("yield_std", torch.tensor([yield_scale], dtype=torch.float32))
        self.temperature_paths = self._build_temperature_paths()

    @classmethod
    def from_tabdiff_dataset(
        cls,
        data_dir: str | Path,
        info: dict[str, Any],
        transformed_train: np.ndarray,
        quantile_transformer: Any,
        **kwargs: Any,
    ) -> "CAPLMechanismConstraint":
        data_path = Path(data_dir)
        x_num = np.load(data_path / "X_num_train.npy").astype(np.float32)
        y = np.load(data_path / "y_train.npy").astype(np.float32).reshape(-1, 1)
        raw_train = np.concatenate([y, x_num], axis=1)
        column_names = [info["column_names"][info["target_col_idx"][0]]]
        column_names.extend(info["column_names"][idx] for idx in info["num_col_idx"])
        if getattr(quantile_transformer, "output_distribution", None) != "normal":
            raise ValueError("MP-TabDiff requires a normal-output QuantileTransformer.")
        return cls(
            column_names,
            transformed_train,
            raw_train,
            quantile_values=quantile_transformer.quantiles_,
            quantile_references=quantile_transformer.references_,
            **kwargs,
        )

    def to_physical(self, x_transformed: torch.Tensor) -> torch.Tensor:
        """Invert the fitted QuantileTransformer with linear interpolation."""

        if x_transformed.shape[-1] != self.quantile_values.shape[1]:
            raise ValueError("x_transformed has an unexpected numerical feature dimension.")
        original_shape = x_transformed.shape
        flat = x_transformed.reshape(-1, original_shape[-1])
        probabilities = 0.5 * (1.0 + torch.erf(flat / np.sqrt(2.0)))
        references = self.quantile_references.to(dtype=flat.dtype)
        lower_bound = probabilities - self.QUANTILE_BOUND_THRESHOLD < references[0]
        upper_bound = probabilities + self.QUANTILE_BOUND_THRESHOLD > references[-1]
        nan_mask = torch.isnan(probabilities)
        probabilities = torch.nan_to_num(
            probabilities,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(references[0], references[-1])

        upper = torch.searchsorted(references, probabilities, right=True)
        upper = upper.clamp(1, references.numel() - 1)
        lower = upper - 1
        feature = torch.arange(flat.shape[1], device=flat.device).unsqueeze(0).expand_as(lower)
        values = self.quantile_values.to(dtype=flat.dtype)
        value_low = values[lower, feature]
        value_high = values[upper, feature]
        reference_low = references[lower]
        reference_high = references[upper]
        fraction = (probabilities - reference_low) / (reference_high - reference_low)
        physical = value_low + fraction * (value_high - value_low)
        physical = torch.where(lower_bound, values[0].unsqueeze(0), physical)
        physical = torch.where(upper_bound, values[-1].unsqueeze(0), physical)
        physical = torch.where(nan_mask, flat, physical)
        return physical.reshape(original_shape)

    def forward(self, x_transformed: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        if not self.enabled:
            energy = x_transformed.new_zeros(x_transformed.shape[0])
            return energy.mean() if reduce else energy
        physical = self.to_physical(x_transformed)
        parts = self.energy_parts_from_physical(physical)
        energy = (
            self.weights.temperature_path * parts["temperature_path"]
            + self.weights.production_window * parts["production_window"]
            + self.weights.yield_residual * parts["yield_residual"]
        )
        return energy.mean() if reduce else energy

    def energy_parts_from_physical(self, physical: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return exactly the three terms in the paper's mechanism energy."""

        return {
            "temperature_path": self.temperature_path_energy(physical),
            "production_window": self.production_window_energy(physical),
            "yield_residual": self.yield_residual_energy(physical),
        }

    def temperature_path_energy(self, physical: torch.Tensor) -> torch.Tensor:
        penalties = []
        delta_t = physical.new_tensor(self.temperature_hold_tolerance)
        for heat, soak_start, soak_end, cool in self.temperature_paths:
            heat_penalty = F.relu(physical[:, heat] - physical[:, soak_start]).pow(2)
            hold_penalty = F.relu(
                torch.abs(physical[:, soak_start] - physical[:, soak_end]) - delta_t
            ).pow(2)
            cool_penalty = F.relu(physical[:, cool] - physical[:, soak_end]).pow(2)
            penalties.append(heat_penalty + hold_penalty + cool_penalty)
        return torch.stack(penalties, dim=1).mean(dim=1)

    def production_window_energy(self, physical: torch.Tensor) -> torch.Tensor:
        features = physical[:, 1:]
        below = F.relu(self.window_low - features) / self.window_spread.clamp_min(1.0e-6)
        above = F.relu(features - self.window_high) / self.window_spread.clamp_min(1.0e-6)
        return (below + above).pow(2).mean(dim=1)

    def yield_residual_energy(self, physical: torch.Tensor) -> torch.Tensor:
        y = physical[:, 0]
        x = physical[:, 1:]
        pred = self.yield_intercept[0] + x.matmul(self.yield_coef)
        excess = F.relu(torch.abs(y - pred) - self.yield_tolerance)
        return (excess / self.yield_std[0].clamp_min(1.0e-6)).pow(2)

    @torch.no_grad()
    def violation_rate(self, x_transformed: torch.Tensor, threshold: float = 0.05) -> float:
        energy = self.forward(x_transformed, reduce=False)
        return float((energy > float(threshold)).float().mean().cpu())

    @torch.no_grad()
    def distribution_shift(self, x_transformed: torch.Tensor) -> float:
        z = (x_transformed - self.transformed_mean) / self.transformed_std
        return float(z.pow(2).mean().sqrt().cpu())

    def _build_temperature_paths(self) -> list[tuple[int, int, int, int]]:
        name_to_idx = {str(name).strip().casefold(): idx for idx, name in enumerate(self.column_names)}

        def resolve(role: str, aliases: tuple[str, ...]) -> int:
            for alias in aliases:
                idx = name_to_idx.get(alias.strip().casefold())
                if idx is not None:
                    return idx
            raise ValueError(
                f"Missing the paper-required {role} temperature column; tried {list(aliases)}."
            )

        heat = resolve("preheating-furnace", ("JPF_PT", "JPF-PT", "JPF PT", "JPF预热炉温度"))
        soak_start = resolve("heating-furnace", ("HF_T", "HF-T", "HF T", "加热炉温度"))
        soak_end = resolve("soaking-furnace", ("SF_T", "SF-T", "SF T", "均热炉温度"))
        cool = resolve("slow-cooling-furnace", ("SC_T", "SC-T", "SC T", "缓冷炉温度"))
        if len({heat, soak_start, soak_end, cool}) != 4:
            raise ValueError("The CAPL temperature path requires four distinct measurements.")
        return [(heat, soak_start, soak_end, cool)]

    @staticmethod
    def _fit_ridge_yield_proxy(raw: np.ndarray, ridge_alpha: float) -> tuple[np.ndarray, float]:
        x = raw[:, 1:].astype(np.float64)
        y = raw[:, 0].astype(np.float64)
        x_mean = np.nanmean(x, axis=0)
        x_std = np.nanstd(x, axis=0)
        x_std = np.where(x_std < 1.0e-6, 1.0, x_std)
        xz = (x - x_mean) / x_std
        design = np.concatenate([np.ones((xz.shape[0], 1)), xz], axis=1)
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_alpha)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        intercept_z = float(beta[0])
        coef_z = beta[1:]
        coef_raw = coef_z / x_std
        intercept_raw = intercept_z - float(np.sum(coef_z * x_mean / x_std))
        return coef_raw.astype(np.float32), intercept_raw
