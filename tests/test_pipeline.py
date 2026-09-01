from __future__ import annotations

from pathlib import Path

import pytest

import config
import config_loader
from pipeline import load_config_overrides
from protocol_integrity import FileMutationError, file_sha256


def test_relative_yaml_path_uses_project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "test.yaml").write_text(
        "experiment_name: relative_yaml_loaded\nsplit_method: chronological\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(config, "EXPERIMENT_NAME", "before")
    monkeypatch.setattr(config, "SPLIT_METHOD", "stratified_random")
    monkeypatch.chdir(outside)

    load_config_overrides("configs/test.yaml")

    assert config.EXPERIMENT_NAME == "relative_yaml_loaded"
    assert config.SPLIT_METHOD == "chronological"
    assert config.CONFIG_SHA256 == file_sha256(config_dir / "test.yaml")


def test_yaml_change_during_parse_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: before\n", encoding="utf-8")
    original_parse = config_loader.load_yaml_config_text

    def mutate_after_snapshot(text: str) -> dict[str, object]:
        loaded = original_parse(text)
        config_path.write_text("experiment_name: changed\n", encoding="utf-8")
        return loaded

    monkeypatch.setattr(config_loader, "load_yaml_config_text", mutate_after_snapshot)

    with pytest.raises(FileMutationError, match="YAML configuration changed"):
        load_config_overrides(str(config_path))

    assert config.CONFIG_SHA256 == ""
