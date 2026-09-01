from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generation.postprocess as postprocess  # noqa: E402
import generation.prepare as prepare  # noqa: E402
import generation.sample as sample  # noqa: E402
import generation.train as generation_train  # noqa: E402
from generation.tabdiff import (  # noqa: E402
    base_train_command,
    checkpoint_output_snapshot,
    copy_latest_sample,
    find_fresh_checkpoint,
    sample_command,
    sample_output_snapshot,
    train_command,
)
from protocol_integrity import FileMutationError, file_sha256  # noqa: E402


def _source_table(rows: int = 20) -> pd.DataFrame:
    values: dict[str, object] = {
        key: np.arange(rows, dtype=np.float32) + column_idx
        for column_idx, key in enumerate(reversed(prepare.PAPER_FEATURE_KEYS), start=1)
    }
    values["YS"] = np.linspace(109.0, 213.0, rows, dtype=np.float32)
    values["EL"] = np.linspace(1.0, 2.0, rows, dtype=np.float32)
    values["ID"] = [f"coil-{idx:04d}" for idx in range(rows)]
    values["operator_note"] = ["private annotation"] * rows
    ordered = ["ID", "EL", "operator_note", *reversed(prepare.PAPER_FEATURE_KEYS), "YS"]
    return pd.DataFrame(values).loc[:, ordered]


def _patch_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: pd.DataFrame,
) -> tuple[Path, dict[str, object]]:
    source_path = tmp_path / "private_source.csv"
    repo_path = tmp_path / "tabdiff"
    calls: dict[str, object] = {}
    source.to_csv(source_path, index=False)

    monkeypatch.setattr(prepare, "_resolve_data_path", lambda _: source_path)
    monkeypatch.setattr(prepare, "_read_table", lambda _: source.copy())
    monkeypatch.setattr(prepare, "require_tabdiff_repo", lambda _: repo_path)

    def fixed_split(labels: np.ndarray, seed: int, method: str):
        calls["seed"] = seed
        calls["method"] = method
        return np.arange(12), np.arange(12, 16), np.arange(16, 20)

    monkeypatch.setattr(prepare, "_split_indices", fixed_split)
    monkeypatch.setattr(prepare.config, "SPLIT_SEED", 42, raising=False)
    monkeypatch.setattr(prepare.config, "SEED", 999, raising=False)
    monkeypatch.setattr(prepare.config, "SPLIT_METHOD", "stratified_random", raising=False)
    return repo_path, calls


def test_prepare_exports_only_the_21_paper_features_then_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_table()
    repo_path, calls = _patch_io(monkeypatch, tmp_path, source)
    output_dir = tmp_path / "prepared"

    result = prepare.prepare_tabdiff_data(
        data_path=tmp_path / "ignored.csv",
        label_col="YS",
        output_dir=output_dir,
        repo_path=repo_path,
        dataset_name="capl_test",
    )

    expected_columns = [*prepare.PAPER_FEATURE_KEYS, "YS"]
    exported = pd.read_csv(result["project_train_csv"])
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    info = json.loads(Path(result["tabdiff_info_json"]).read_text(encoding="utf-8"))

    assert exported.columns.tolist() == expected_columns
    assert len(exported) == 12
    assert not {"EL", "ID", "operator_note"}.intersection(exported.columns)
    assert result["numerical_columns"] == list(prepare.PAPER_FEATURE_KEYS)
    assert result["categorical_columns"] == []
    assert metadata["feature_columns"] == list(prepare.PAPER_FEATURE_KEYS)
    assert metadata["paper_feature_keys"] == list(prepare.PAPER_FEATURE_KEYS)
    assert metadata["all_columns"] == expected_columns
    assert metadata["source_data_sha256"] == file_sha256(tmp_path / "private_source.csv")
    assert metadata["prepared_train_csv_sha256"] == file_sha256(result["project_train_csv"])
    assert metadata["schema_sha256"]
    assert set(metadata["split_id_sha256"]) == {"train", "val", "test"}
    assert metadata["generator_input_id_sha256"] == metadata["split_id_sha256"]["train"]
    assert metadata["combined_split_sha256"]
    assert info["column_names"] == expected_columns
    assert info["num_col_idx"] == list(range(21))
    assert info["cat_col_idx"] == []
    assert info["target_col_idx"] == [21]


def test_prepare_uses_fixed_split_seed_independent_of_run_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path, calls = _patch_io(monkeypatch, tmp_path, _source_table())

    result = prepare.prepare_tabdiff_data(
        data_path=tmp_path / "ignored.csv",
        label_col="YS",
        output_dir=tmp_path / "prepared",
        repo_path=repo_path,
        dataset_name="capl_test",
    )
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))

    assert calls["seed"] == 42
    assert metadata["split_seed"] == 42
    assert metadata["project_split_sizes"] == {"train": 12, "val": 4, "test": 4}


def test_prepare_rejects_a_source_missing_any_paper_feature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_table().drop(columns=["CT"])
    repo_path, _ = _patch_io(monkeypatch, tmp_path, source)

    with pytest.raises(KeyError, match="Missing required CAPL variable columns"):
        prepare.prepare_tabdiff_data(
            data_path=tmp_path / "ignored.csv",
            label_col="YS",
            output_dir=tmp_path / "prepared",
            repo_path=repo_path,
            dataset_name="capl_test",
        )


def test_sampling_is_locked_to_the_unconditional_paper_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sample, "require_tabdiff_repo", lambda: tmp_path)
    monkeypatch.setattr(sample, "dataname", lambda: "capl_test")
    monkeypatch.setattr(sample, "sample_command", lambda *_args, **_kwargs: ["python", "scripts/sample.py"])
    monkeypatch.setattr(sample, "command_summary", lambda _commands: ["python scripts/sample.py"])
    monkeypatch.setattr(sample, "check_tabdiff_dependencies", lambda _repo: (True, "ready"))
    monkeypatch.setattr(sample, "tabdiff_remote", lambda _repo: "vendored")
    result = sample.run_tabdiff_sample(dry_run=True, num_samples=64)

    assert result["sampling_mode"] == "unconditional"
    assert "condition_mode" not in result


def test_tabdiff_commands_enable_official_deterministic_seed_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare.config, "TABDIFF_GENERATION_SEED", 0, raising=False)

    assert "--deterministic" in train_command(tmp_path).command
    assert "--deterministic" in base_train_command(tmp_path).command
    assert "--deterministic" in sample_command(tmp_path).command

    monkeypatch.setattr(prepare.config, "TABDIFF_GENERATION_SEED", 7, raising=False)
    with pytest.raises(ValueError, match="must be 0"):
        train_command(tmp_path)


def test_tabdiff_training_reprocesses_even_when_outputs_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[object] = []
    monkeypatch.setattr(generation_train.config, "TABDIFF_CKPT_PATH", "", raising=False)
    monkeypatch.setattr(generation_train, "require_tabdiff_repo", lambda: tmp_path)
    monkeypatch.setattr(generation_train, "dataname", lambda: "capl_test")
    monkeypatch.setattr(generation_train, "process_command", lambda *_: "process")
    monkeypatch.setattr(generation_train, "base_train_command", lambda *_: "base")
    monkeypatch.setattr(generation_train, "base_exp_name", lambda: "base_exp")
    monkeypatch.setattr(generation_train, "train_command", lambda *_: "finetune")
    monkeypatch.setattr(generation_train, "processed_dataset_ready", lambda *_: True)
    monkeypatch.setattr(generation_train, "check_tabdiff_dependencies", lambda _: (True, "ok"))
    monkeypatch.setattr(generation_train, "tabdiff_remote", lambda _: "vendored")
    monkeypatch.setattr(generation_train, "command_summary", lambda values: list(values))
    monkeypatch.setattr(generation_train, "run_command", commands.append)
    monkeypatch.setattr(generation_train, "generation_seed", lambda: 0)
    monkeypatch.setattr(generation_train, "validate_prepared_tabdiff_input", lambda _: {})
    checkpoint = tmp_path / "fresh.pt"
    checkpoint.write_bytes(b"current checkpoint")
    monkeypatch.setattr(generation_train, "checkpoint_output_snapshot", lambda *_: {})
    monkeypatch.setattr(generation_train, "find_fresh_checkpoint", lambda *_: checkpoint)

    result = generation_train.run_tabdiff_train(dry_run=False)

    assert result["processed_dataset_ready_before_process"] is True
    assert commands == ["process", "base", "finetune"]
    assert result["training_stages"] == ["base", "mechanism_finetune"]
    assert result["base_checkpoint_path"] == str(checkpoint)
    assert result["checkpoint_path"] == str(checkpoint)


def test_tabdiff_base_training_is_full_scope_without_finetune_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare.config, "TABDIFF_GENERATION_SEED", 0, raising=False)

    command = base_train_command(tmp_path, "capl_test").command

    for flag in (
        "--ckpt_path",
        "--mechanism_constraint",
        "--trainable_scope",
        "--finetune_lr",
        "--finetune_steps",
        "--reset_train_epoch",
    ):
        assert flag not in command


def test_tabdiff_external_checkpoint_skips_base_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(generation_train.config, "TABDIFF_CKPT_PATH", str(tmp_path / "base.pt"), raising=False)
    monkeypatch.setattr(generation_train, "require_tabdiff_repo", lambda: tmp_path)
    monkeypatch.setattr(generation_train, "dataname", lambda: "capl_test")
    monkeypatch.setattr(generation_train, "process_command", lambda *_: "process")
    monkeypatch.setattr(
        generation_train,
        "base_train_command",
        lambda *_: pytest.fail("an external checkpoint must skip base training"),
    )
    monkeypatch.setattr(generation_train, "train_command", lambda *_: "finetune")
    monkeypatch.setattr(generation_train, "processed_dataset_ready", lambda *_: True)
    monkeypatch.setattr(generation_train, "check_tabdiff_dependencies", lambda _: (True, "ok"))
    monkeypatch.setattr(generation_train, "tabdiff_remote", lambda _: "vendored")
    monkeypatch.setattr(generation_train, "command_summary", lambda values: list(values))
    monkeypatch.setattr(generation_train, "generation_seed", lambda: 0)

    result = generation_train.run_tabdiff_train(dry_run=True)

    assert result["commands"] == ["process", "finetune"]
    assert result["training_stages"] == ["mechanism_finetune"]


def test_tabdiff_mechanism_finetune_requires_the_selected_base_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare.config, "TABDIFF_GENERATION_SEED", 0, raising=False)
    checkpoint = (tmp_path / "base.pt").resolve()

    command = train_command(tmp_path, "capl_test", checkpoint).command

    assert Path(command[command.index("--ckpt_path") + 1]) == checkpoint
    assert "--mechanism_constraint" in command
    assert "--reset_train_epoch" in command


def test_postprocess_writes_split_specific_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path, _ = _patch_io(monkeypatch, tmp_path, _source_table())
    prepared = prepare.prepare_tabdiff_data(
        data_path=tmp_path / "ignored.csv",
        label_col="YS",
        output_dir=tmp_path / "prepared",
        repo_path=repo_path,
        dataset_name="capl_test",
    )
    raw_path = tmp_path / "samples.csv"
    pd.read_csv(prepared["project_train_csv"]).to_csv(raw_path, index=False)
    output_path = tmp_path / "synthetic.csv"
    checkpoint_path = tmp_path / "best_ema_model_current.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(postprocess, "scientific_code_sha256", lambda: "a" * 64)

    result = postprocess.postprocess_tabdiff_samples(
        raw_path=raw_path,
        metadata_path=prepared["metadata_path"],
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        expected_scientific_code_sha256="a" * 64,
        generation_protocol_sha256="b" * 64,
    )
    provenance = json.loads(Path(result["provenance_path"]).read_text(encoding="utf-8"))

    assert provenance["source"]["sha256"] == file_sha256(tmp_path / "private_source.csv")
    assert provenance["split"]["id_sha256"]["train"] == provenance["split"]["generator_input_id_sha256"]
    assert provenance["generation"]["seed"] == 0
    assert provenance["generation"]["deterministic"] is True
    assert provenance["generation"]["checkpoint_path"] == str(checkpoint_path.resolve())
    assert provenance["generation"]["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert provenance["generation"]["prepared_csv_path"] == str(Path(prepared["project_train_csv"]).resolve())
    assert provenance["generation"]["raw_samples_path"] == str(raw_path.resolve())
    assert provenance["generation"]["scientific_code_sha256"] == "a" * 64
    assert provenance["generation"]["protocol_sha256"] == "b" * 64
    assert provenance["synthetic"]["sha256"] == file_sha256(output_path)


def test_postprocess_rejects_scientific_code_change_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path, _ = _patch_io(monkeypatch, tmp_path, _source_table())
    prepared = prepare.prepare_tabdiff_data(
        data_path=tmp_path / "ignored.csv",
        label_col="YS",
        output_dir=tmp_path / "prepared",
        repo_path=repo_path,
        dataset_name="capl_test",
    )
    raw_path = tmp_path / "samples.csv"
    pd.read_csv(prepared["project_train_csv"]).to_csv(raw_path, index=False)
    checkpoint_path = tmp_path / "best_ema_model_current.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    output_path = tmp_path / "synthetic.csv"
    monkeypatch.setattr(postprocess, "scientific_code_sha256", lambda: "a" * 64)

    with pytest.raises(RuntimeError, match="Scientific source code changed"):
        postprocess.postprocess_tabdiff_samples(
            raw_path=raw_path,
            metadata_path=prepared["metadata_path"],
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            expected_scientific_code_sha256="b" * 64,
            generation_protocol_sha256="c" * 64,
        )

    assert not output_path.exists()


def test_sampling_rejects_checkpoint_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best_ema_model_current.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    expected_hash = file_sha256(checkpoint)
    monkeypatch.setattr(sample, "require_tabdiff_repo", lambda: tmp_path)
    monkeypatch.setattr(sample, "dataname", lambda: "capl_test")
    monkeypatch.setattr(sample, "sample_command", lambda *_args, **_kwargs: "sample")
    monkeypatch.setattr(sample, "check_tabdiff_dependencies", lambda _: (True, "ok"))
    monkeypatch.setattr(sample, "tabdiff_remote", lambda _: "vendored")
    monkeypatch.setattr(sample, "command_summary", lambda values: list(values))
    monkeypatch.setattr(sample, "generation_seed", lambda: 0)
    monkeypatch.setattr(sample, "sample_output_snapshot", lambda *_: {})
    monkeypatch.setattr(sample, "_raw_output_path", lambda: tmp_path / "raw.csv")

    def mutate_checkpoint(_command: object) -> None:
        checkpoint.write_bytes(b"checkpoint-b")

    monkeypatch.setattr(sample, "run_command", mutate_checkpoint)
    monkeypatch.setattr(
        sample,
        "copy_latest_sample",
        lambda *_args, **_kwargs: pytest.fail("mutated checkpoint must stop collection"),
    )

    with pytest.raises(FileMutationError, match="sampling checkpoint changed"):
        sample.run_tabdiff_sample(
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256=expected_hash,
        )


def test_postprocess_does_not_publish_provenance_after_checkpoint_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path, _ = _patch_io(monkeypatch, tmp_path, _source_table())
    prepared = prepare.prepare_tabdiff_data(
        data_path=tmp_path / "ignored.csv",
        label_col="YS",
        output_dir=tmp_path / "prepared",
        repo_path=repo_path,
        dataset_name="capl_test",
    )
    raw_path = tmp_path / "samples.csv"
    pd.read_csv(prepared["project_train_csv"]).to_csv(raw_path, index=False)
    output_path = tmp_path / "synthetic.csv"
    checkpoint = tmp_path / "best_ema_model_current.pt"
    checkpoint.write_bytes(b"checkpoint-a")
    expected_hash = file_sha256(checkpoint)
    save_samples = postprocess._save_samples

    def save_then_mutate(frame: pd.DataFrame, path: Path) -> None:
        save_samples(frame, path)
        checkpoint.write_bytes(b"checkpoint-b")

    monkeypatch.setattr(postprocess, "_save_samples", save_then_mutate)

    with pytest.raises(FileMutationError, match="checkpoint changed"):
        postprocess.postprocess_tabdiff_samples(
            raw_path=raw_path,
            metadata_path=prepared["metadata_path"],
            output_path=output_path,
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256=expected_hash,
        )

    provenance_path = output_path.with_suffix(".provenance.json")
    assert not provenance_path.exists()
    assert not provenance_path.with_suffix(provenance_path.suffix + ".tmp").exists()


def test_sampling_refuses_to_copy_an_unchanged_old_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare.config, "TABDIFF_EXP_NAME", "freshness_test", raising=False)
    sample_dir = tmp_path / "tabdiff" / "result" / "capl_test" / "freshness_test"
    sample_dir.mkdir(parents=True)
    old_sample = sample_dir / "samples.csv"
    old_sample.write_text("x\n1\n", encoding="utf-8")
    snapshot = sample_output_snapshot(tmp_path, "capl_test")

    with pytest.raises(RuntimeError, match="refusing to reuse a stale output"):
        copy_latest_sample(
            tmp_path,
            tmp_path / "collected.csv",
            "capl_test",
            previous_snapshot=snapshot,
        )

    old_sample.write_text("x\n2\n", encoding="utf-8")
    copied = copy_latest_sample(
        tmp_path,
        tmp_path / "collected.csv",
        "capl_test",
        previous_snapshot=snapshot,
    )
    assert copied.read_text(encoding="utf-8") == "x\n2\n"


def test_sampling_rejects_multiple_fresh_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare.config, "TABDIFF_EXP_NAME", "freshness_test", raising=False)
    sample_dir = tmp_path / "tabdiff" / "result" / "capl_test" / "freshness_test"
    sample_dir.mkdir(parents=True)
    snapshot = sample_output_snapshot(tmp_path, "capl_test")
    (sample_dir / "samples.csv").write_text("x\n1\n", encoding="utf-8")
    (sample_dir / "samples_1.csv").write_text("x\n2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="found 2"):
        copy_latest_sample(
            tmp_path,
            tmp_path / "collected.csv",
            "capl_test",
            previous_snapshot=snapshot,
        )

    assert not (tmp_path / "collected.csv").exists()


def test_postprocess_rejects_a_modified_prepared_training_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path, _ = _patch_io(monkeypatch, tmp_path, _source_table())
    prepared = prepare.prepare_tabdiff_data(
        data_path=tmp_path / "ignored.csv",
        label_col="YS",
        output_dir=tmp_path / "prepared",
        repo_path=repo_path,
        dataset_name="capl_test",
    )
    raw_path = tmp_path / "samples.csv"
    frame = pd.read_csv(prepared["project_train_csv"])
    frame.to_csv(raw_path, index=False)
    frame.iloc[:-1].to_csv(prepared["project_train_csv"], index=False)
    checkpoint_path = tmp_path / "best_ema_model_current.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    with pytest.raises(RuntimeError, match="prepared_train_csv_sha256"):
        postprocess.postprocess_tabdiff_samples(
            raw_path=raw_path,
            metadata_path=prepared["metadata_path"],
            output_path=tmp_path / "synthetic.csv",
            checkpoint_path=checkpoint_path,
        )


def test_postprocess_rejects_invalid_checkpoint_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_path, _ = _patch_io(monkeypatch, tmp_path, _source_table())
    prepared = prepare.prepare_tabdiff_data(
        data_path=tmp_path / "ignored.csv",
        label_col="YS",
        output_dir=tmp_path / "prepared",
        repo_path=repo_path,
        dataset_name="capl_test",
    )
    raw_path = tmp_path / "samples.csv"
    pd.read_csv(prepared["project_train_csv"]).to_csv(raw_path, index=False)
    output_path = tmp_path / "synthetic.csv"

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        postprocess.postprocess_tabdiff_samples(
            raw_path=raw_path,
            metadata_path=prepared["metadata_path"],
            output_path=output_path,
            checkpoint_path=tmp_path / "missing.pt",
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".postprocess.json").exists()
    assert not output_path.with_suffix(".provenance.json").exists()


def test_checkpoint_selection_rejects_stale_and_ambiguous_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare.config, "TABDIFF_EXP_NAME", "checkpoint_test", raising=False)
    checkpoint_dir = tmp_path / "tabdiff" / "ckpt" / "capl_test" / "checkpoint_test"
    checkpoint_dir.mkdir(parents=True)
    old_checkpoint = checkpoint_dir / "best_ema_model_old.pt"
    old_checkpoint.write_bytes(b"old")
    snapshot = checkpoint_output_snapshot(tmp_path, "capl_test")

    with pytest.raises(RuntimeError, match="found 0"):
        find_fresh_checkpoint(tmp_path, "capl_test", snapshot)

    (checkpoint_dir / "best_ema_model_new_a.pt").write_bytes(b"new-a")
    (checkpoint_dir / "best_ema_model_new_b.pt").write_bytes(b"new-b")
    with pytest.raises(RuntimeError, match="found 2"):
        find_fresh_checkpoint(tmp_path, "capl_test", snapshot)


def test_sampling_command_uses_the_exact_fresh_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare.config, "TABDIFF_GENERATION_SEED", 0, raising=False)
    checkpoint = (tmp_path / "best_ema_model_current.pt").resolve()
    checkpoint.write_bytes(b"current")

    command = sample_command(
        tmp_path,
        "capl_test",
        num_samples=32,
        checkpoint_path=checkpoint,
    ).command

    assert "--ckpt_path" in command
    assert Path(command[command.index("--ckpt_path") + 1]) == checkpoint
