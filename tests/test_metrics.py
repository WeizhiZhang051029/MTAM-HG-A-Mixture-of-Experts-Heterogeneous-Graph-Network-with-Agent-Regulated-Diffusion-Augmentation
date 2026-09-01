from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import evaluate_model, save_evaluation_outputs
from metrics import compute_metrics, tail_mae


class MetricTest(unittest.TestCase):
    def test_compute_metrics_includes_required_tail_mae_name(self) -> None:
        y_true = np.arange(10, dtype=np.float32).reshape(-1, 1)
        y_pred = y_true.copy()
        y_pred[0, 0] += 2.0
        y_pred[-1, 0] -= 3.0

        metrics = compute_metrics(y_true, y_pred, tail_thresholds=(0.9, 8.1))

        self.assertIn("RMSE", metrics)
        self.assertIn("MAPE", metrics)
        self.assertIn("R2", metrics)
        self.assertIn("TAIL_MAE", metrics)
        self.assertNotIn("Tail_MAE", metrics)
        self.assertAlmostEqual(metrics["TAIL_MAE"], 2.5)

    def test_tail_mae_uses_bottom_and_top_ten_percent_by_true_label(self) -> None:
        y_true = np.arange(20, dtype=np.float32).reshape(-1, 1)
        y_pred = y_true.copy()
        y_pred[[0, 1, 18, 19], 0] += np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        self.assertAlmostEqual(tail_mae(y_true, y_pred, quantile=0.10), 2.5)

    def test_evaluate_model_uses_training_tail_thresholds(self) -> None:
        class IdentityModel(torch.nn.Module):
            def forward(self, x):
                return {"mu": x[:, :1]}

        x = torch.arange(10, dtype=torch.float32).reshape(-1, 1)
        y = x.clone()
        loader = DataLoader(TensorDataset(x, y), batch_size=5)
        bundle = SimpleNamespace(tail_thresholds=(-999.0, 999.0), y_scaler=None)

        with patch("evaluate.config.STANDARDIZE_Y", False), patch("evaluate.compute_metrics") as mocked:
            mocked.return_value = {}
            evaluate_model(IdentityModel(), loader, torch.device("cpu"), bundle)

        _, kwargs = mocked.call_args
        self.assertEqual(kwargs.get("tail_thresholds"), (-999.0, 999.0))

    def test_save_evaluation_outputs_accepts_top1_diagnostics(self) -> None:
        collected = {
            "y": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "mu": np.array([[1.1], [1.9], [3.2]], dtype=np.float32),
            "sample_confidence": np.array([0.9, 0.8, 0.7], dtype=np.float32),
            "expert_uncertainty": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "agent_gate_entropy": np.array([0.2, 0.3, 0.4], dtype=np.float32),
            "topk_indices": np.array([[0], [1], [0]], dtype=np.int64),
            "expert_weights": np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]], dtype=np.float32),
            "expert_preds": np.array(
                [
                    [[1.0], [1.2]],
                    [[1.8], [2.0]],
                    [[3.1], [3.3]],
                ],
                dtype=np.float32,
            ),
        }

        with tempfile.TemporaryDirectory() as tmp:
            save_evaluation_outputs({}, collected, Path(tmp))
            diag = pd.read_csv(Path(tmp) / "agent_diagnostics.csv")

        self.assertIn("top1_index", diag.columns)
        self.assertNotIn("top2_index", diag.columns)



if __name__ == "__main__":
    unittest.main()
