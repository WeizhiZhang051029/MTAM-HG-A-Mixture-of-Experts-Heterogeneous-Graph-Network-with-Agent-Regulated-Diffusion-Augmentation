"""Opt-in integration test with a private CAPL table."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not os.environ.get("MTAM_TEST_DATA"), reason="Set MTAM_TEST_DATA to a private CAPL table.")
def test_private_data_generation_training_and_reload(tmp_path: Path):
    vendor = tmp_path / "TabDiff"
    shutil.copytree(
        ROOT / "third_party" / "TabDiff", vendor,
        ignore=shutil.ignore_patterns("__pycache__", "data", "ckpt", "result", "synthetic"),
    )
    model_config = vendor / "tabdiff/configs/tabdiff_configs.toml"
    text = model_config.read_text(encoding="utf-8")
    model_config.write_text(text.replace("steps = 8000", "steps = 2"), encoding="utf-8")
    config = yaml.safe_load((ROOT / "configs/mtam_hg.yaml").read_text(encoding="utf-8"))
    config.update({
        "tabdiff_repo_path": str(vendor),
        "tabdiff_data_dir": str(tmp_path / "prepared"),
        "tabdiff_output_dir": str(tmp_path / "generated"),
        "synthetic_data_path": str(tmp_path / "synthetic.csv"),
        "output_base": str(tmp_path / "runs"),
        "model_seeds": [7, 19],
        "epochs": 2,
        "synthetic_pretrain_epochs": 2,
        "synthetic_agent_epochs": 1,
        "tabdiff_finetune_steps": 2,
        "tabdiff_num_samples": 64,
    })
    cfg = tmp_path / "integration.yaml"
    cfg.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    command = [
        sys.executable, str(ROOT / "run_experiment.py"),
        "--config", str(cfg), "--data_path", os.environ["MTAM_TEST_DATA"],
    ]
    with (tmp_path / "pipeline.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (tmp_path / "pipeline.log").read_text(encoding="utf-8")[-12000:]
    summary_path, = (tmp_path / "runs").rglob("experiment_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = summary["runs"]
    assert len(runs) == 2
    assert runs[0]["metrics"]["Combined_Split_SHA256"] != runs[1]["metrics"]["Combined_Split_SHA256"]
    assert runs[0]["metrics"]["Synthetic_SHA256"] != runs[1]["metrics"]["Synthetic_SHA256"]
    for run in runs:
        metrics = run["metrics"]
        assert metrics["Split_Seed"] == run["seed"]
        assert metrics["CBTG_Feedback_Source"] == "real_training_set"
        output = Path(run["metrics_path"]).parent
        adjacency = np.load(output / "learned_A_kg_experts.npy")
        edges = np.abs(adjacency).mean(axis=0)
        np.fill_diagonal(edges, 0)
        scores = edges.sum(0) + edges.sum(1)
        np.testing.assert_allclose(np.load(output / "node_importance.npy"), scores / max(scores.max(), 1e-12))
        for name in ("RMSE", "MAE", "MAPE", "R2"):
            assert np.isfinite(metrics[name])
        checkpoint, = list(output.parent.rglob("best_model.pth"))
        reload_dir = tmp_path / f"reload_{run['seed']}"
        reload_command = [
            sys.executable, str(ROOT / "pipeline.py"),
            "--config", str(cfg), "--mode", "evaluate",
            "--data_path", os.environ["MTAM_TEST_DATA"],
            "--seed", str(run["seed"]), "--checkpoint", str(checkpoint),
            "--output_dir", str(reload_dir),
        ]
        reloaded = subprocess.run(reload_command, cwd=ROOT, env=env, capture_output=True, text=True)
        assert reloaded.returncode == 0, reloaded.stdout + reloaded.stderr
        reloaded_path, = list(reload_dir.rglob("metrics.json"))
        reloaded_metrics = json.loads(reloaded_path.read_text(encoding="utf-8"))
        for name in ("RMSE", "MAE", "MAPE", "R2"):
            assert reloaded_metrics[name] == pytest.approx(metrics[name], rel=1e-6, abs=1e-6)
