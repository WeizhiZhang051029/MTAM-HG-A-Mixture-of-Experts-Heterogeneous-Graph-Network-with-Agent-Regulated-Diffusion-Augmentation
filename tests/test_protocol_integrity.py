from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
import training.cbtg as cbtg  # noqa: E402
from dataset import _split_indices, create_dataloaders  # noqa: E402
from generation.prepare import PAPER_FEATURE_KEYS  # noqa: E402
from protocol_integrity import (  # noqa: E402
    PROVENANCE_FORMAT,
    FileMutationError,
    SyntheticProvenanceError,
    assert_file_snapshot_current,
    canonical_sha256,
    file_sha256,
    generation_config_snapshot,
    read_table_snapshot,
    schema_sha256,
    split_fingerprints,
    synthetic_provenance_path,
    validate_synthetic_provenance_for_data_bundle,
    validate_synthetic_provenance_for_runner,
)
from train import split_metadata  # noqa: E402
from training.cbtg import _load_synthetic_bundle  # noqa: E402


def _source_table(rows: int = 40) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            name: np.linspace(index, index + rows - 1, rows, dtype=np.float32)
            for index, name in enumerate(PAPER_FEATURE_KEYS)
        }
    )
    frame["YS"] = np.linspace(110.0, 240.0, rows, dtype=np.float32)
    return frame


def _write_valid_artifacts(tmp_path: Path) -> tuple[Path, Path, SimpleNamespace]:
    source_path = tmp_path / "source.csv"
    source = _source_table()
    source.to_csv(source_path, index=False)
    labels = source[["YS"]].to_numpy(dtype=np.float32)
    train_ids, val_ids, test_ids = _split_indices(labels, 42, "stratified_random")
    split_hashes, combined_hash = split_fingerprints(
        train_ids,
        val_ids,
        test_ids,
        total_rows=len(source),
    )

    synthetic_path = tmp_path / "synthetic.csv"
    source.iloc[train_ids[:8]].to_csv(synthetic_path, index=False)
    checkpoint_path = tmp_path / "best_ema_model_current.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    config_snapshot = generation_config_snapshot()
    provenance = {
        "format": PROVENANCE_FORMAT,
        "source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "schema_sha256": schema_sha256([*PAPER_FEATURE_KEYS, "YS"]),
        },
        "split": {
            "method": "stratified_random",
            "seed": 42,
            "sizes": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "id_sha256": split_hashes,
            "combined_sha256": combined_hash,
            "generator_input_id_sha256": split_hashes["train"],
            "generator_input_scope": "main_train_split_only",
        },
        "generation": {
            "backend": "TabDiff",
            "seed": 0,
            "backend_seed": 0,
            "deterministic": True,
            "config": config_snapshot,
            "config_sha256": canonical_sha256(config_snapshot),
            "prepared_csv_sha256": file_sha256(source_path),
            "raw_samples_sha256": file_sha256(synthetic_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
        },
        "synthetic": {
            "path": str(synthetic_path),
            "sha256": file_sha256(synthetic_path),
            "rows": 8,
            "columns": [*PAPER_FEATURE_KEYS, "YS"],
        },
    }
    synthetic_provenance_path(synthetic_path).write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        source_sha256=file_sha256(source_path),
        schema_hash=schema_sha256([*PAPER_FEATURE_KEYS, "YS"]),
        split_seed=42,
        split_method="stratified_random",
        split_hashes=split_hashes,
        combined_split_hash=combined_hash,
    )
    return source_path, synthetic_path, bundle


def test_table_snapshot_binds_parsing_and_hash_to_the_same_bytes(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("value\n1\n", encoding="utf-8")
    expected_hash = file_sha256(path)

    snapshot = read_table_snapshot(path)
    path.write_text("value\n2\n", encoding="utf-8")

    assert snapshot.file.sha256 == expected_hash
    assert snapshot.frame["value"].tolist() == [1]
    with pytest.raises(FileMutationError, match="changed during use"):
        assert_file_snapshot_current(snapshot.file, "source data")


def test_split_fingerprints_reject_cross_split_overlap() -> None:
    with pytest.raises(ValueError, match="Split leakage"):
        split_fingerprints([0, 1], [1, 2], [3], total_rows=4)


def test_runner_validator_accepts_matching_source_split_and_generation_seed(tmp_path: Path) -> None:
    source_path, synthetic_path, _ = _write_valid_artifacts(tmp_path)

    result = validate_synthetic_provenance_for_runner(
        synthetic_path,
        source_path,
        split_seed=42,
        split_method="stratified_random",
        generation_seed=0,
    )

    assert result["synthetic_sha256"] == file_sha256(synthetic_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("split_seed", 43, "split.seed"),
        ("split_method", "chronological", "split.method"),
        ("generation_seed", 1, "generation.seed"),
    ],
)
def test_runner_validator_rejects_protocol_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source_path, synthetic_path, _ = _write_valid_artifacts(tmp_path)
    arguments: dict[str, object] = {
        "synthetic_path": synthetic_path,
        "data_path": source_path,
        "split_seed": 42,
        "split_method": "stratified_random",
        "generation_seed": 0,
    }
    arguments[field] = value

    with pytest.raises(SyntheticProvenanceError, match=message):
        validate_synthetic_provenance_for_runner(**arguments)


def test_validator_rejects_modified_source_or_synthetic_artifact(tmp_path: Path) -> None:
    source_path, synthetic_path, bundle = _write_valid_artifacts(tmp_path)
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SyntheticProvenanceError, match="source.sha256"):
        validate_synthetic_provenance_for_runner(
            synthetic_path,
            source_path,
            split_seed=42,
            split_method="stratified_random",
            generation_seed=0,
        )

    source_path, synthetic_path, bundle = _write_valid_artifacts(tmp_path)
    synthetic_path.write_text(synthetic_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SyntheticProvenanceError, match="synthetic.sha256"):
        validate_synthetic_provenance_for_data_bundle(synthetic_path, bundle, generation_seed=0)


def test_cbtg_loader_fails_closed_without_provenance(tmp_path: Path) -> None:
    _, synthetic_path, bundle = _write_valid_artifacts(tmp_path)
    synthetic_provenance_path(synthetic_path).unlink()

    with pytest.raises(SyntheticProvenanceError, match="sidecar is required"):
        _load_synthetic_bundle(bundle, synthetic_path)


def test_cbtg_loader_does_not_bypass_provenance_for_smoke_filename(tmp_path: Path) -> None:
    _, synthetic_path, bundle = _write_valid_artifacts(tmp_path)
    smoke_path = tmp_path / "synthetic_smoke.csv"
    synthetic_path.replace(smoke_path)

    with pytest.raises(SyntheticProvenanceError, match="sidecar is required"):
        _load_synthetic_bundle(bundle, smoke_path)


def test_cbtg_loader_rejects_synthetic_mutation_during_processing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, synthetic_path, _ = _write_valid_artifacts(tmp_path)
    bundle = create_dataloaders(
        data_path=source_path,
        label_column="YS",
        batch_size=8,
        seed=42,
        split_seed=42,
        split_method="stratified_random",
        use_el_as_input=False,
        save_scaler_path=tmp_path / "scaler.pkl",
    )
    compute_scores = cbtg.compute_process_consistency_scores

    def mutate_synthetic(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        scores = compute_scores(*args, **kwargs)
        synthetic_path.write_text(
            synthetic_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return scores

    monkeypatch.setattr(cbtg, "compute_process_consistency_scores", mutate_synthetic)

    with pytest.raises(FileMutationError, match="synthetic data changed"):
        _load_synthetic_bundle(bundle, synthetic_path)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("generation", "backend", "Other", "generation.backend"),
        ("generation", "prepared_csv_sha256", "bad", "prepared_csv_sha256"),
        ("generation", "raw_samples_sha256", "bad", "raw_samples_sha256"),
        ("generation", "checkpoint_sha256", "bad", "checkpoint_sha256"),
        ("synthetic", "rows", 9, "synthetic.rows"),
        ("synthetic", "columns", ["YS", *PAPER_FEATURE_KEYS], "synthetic.columns"),
    ],
)
def test_validator_rejects_invalid_backend_hashes_or_layout(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    _, synthetic_path, bundle = _write_valid_artifacts(tmp_path)
    sidecar_path = synthetic_provenance_path(synthetic_path)
    provenance = json.loads(sidecar_path.read_text(encoding="utf-8"))
    provenance[section][field] = value
    sidecar_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(SyntheticProvenanceError, match=message):
        validate_synthetic_provenance_for_data_bundle(synthetic_path, bundle, generation_seed=0)


def test_runner_validator_accepts_explicit_schema_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, synthetic_path, _ = _write_valid_artifacts(tmp_path)
    monkeypatch.setattr(config, "LABEL_COL", "missing", raising=False)
    monkeypatch.setattr(config, "USE_EL_AS_INPUT", True, raising=False)

    validate_synthetic_provenance_for_runner(
        synthetic_path,
        source_path,
        split_seed=42,
        split_method="stratified_random",
        generation_seed=0,
        label_col="YS",
        use_el_as_input=False,
    )


def test_data_bundle_validator_rejects_split_membership_mismatch(tmp_path: Path) -> None:
    _, synthetic_path, bundle = _write_valid_artifacts(tmp_path)
    bundle.split_hashes = dict(bundle.split_hashes)
    bundle.split_hashes["train"] = "0" * 64

    with pytest.raises(SyntheticProvenanceError, match="split.id_sha256.train"):
        validate_synthetic_provenance_for_data_bundle(synthetic_path, bundle, generation_seed=0)


def test_runner_does_not_compare_unloaded_yaml_config_but_pipeline_validator_does(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path, synthetic_path, bundle = _write_valid_artifacts(tmp_path)
    original_exp_name = config.TABDIFF_EXP_NAME
    monkeypatch.setattr(config, "TABDIFF_EXP_NAME", f"{original_exp_name}_yaml_override")

    validate_synthetic_provenance_for_runner(
        synthetic_path,
        source_path,
        split_seed=42,
        split_method="stratified_random",
        generation_seed=0,
    )

    with pytest.raises(SyntheticProvenanceError, match="generation.config_sha256"):
        validate_synthetic_provenance_for_data_bundle(synthetic_path, bundle, generation_seed=0)

    with pytest.raises(SyntheticProvenanceError, match="generation.config_sha256"):
        validate_synthetic_provenance_for_runner(
            synthetic_path,
            source_path,
            split_seed=42,
            split_method="stratified_random",
            generation_seed=0,
            validate_current_generation_config=True,
        )


def test_validator_checks_recorded_checkpoint_hash_when_checkpoint_exists(tmp_path: Path) -> None:
    _, synthetic_path, bundle = _write_valid_artifacts(tmp_path)
    checkpoint = tmp_path / "best_ema_model_current.pt"
    checkpoint.write_bytes(b"checkpoint")
    sidecar_path = synthetic_provenance_path(synthetic_path)
    provenance = json.loads(sidecar_path.read_text(encoding="utf-8"))
    provenance["generation"]["checkpoint_path"] = str(checkpoint)
    provenance["generation"]["checkpoint_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(SyntheticProvenanceError, match="generation.checkpoint_sha256"):
        validate_synthetic_provenance_for_data_bundle(synthetic_path, bundle, generation_seed=0)


def test_data_bundle_and_split_metadata_separate_model_and_split_seeds(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    _source_table().to_csv(source_path, index=False)

    bundle = create_dataloaders(
        data_path=source_path,
        label_column="YS",
        batch_size=8,
        seed=99,
        split_seed=42,
        split_method="stratified_random",
        use_el_as_input=False,
        save_scaler_path=tmp_path / "scaler.pkl",
    )
    metadata = split_metadata(bundle)

    assert bundle.model_seed == 99
    assert bundle.run_seed == 99
    assert bundle.split_seed == 42
    assert set(bundle.split_hashes) == {"train", "val", "test"}
    assert metadata["model_seed"] == 99
    assert metadata["run_seed"] == 99
    assert metadata["split_seed"] == 42
    assert metadata["split_id_sha256"] == bundle.split_hashes
    assert metadata["combined_split_sha256"] == bundle.combined_split_hash


def test_create_dataloaders_preserves_allow_synthetic_positional_slot() -> None:
    parameters = list(inspect.signature(create_dataloaders).parameters)

    assert parameters.index("allow_synthetic") < parameters.index("split_seed")
