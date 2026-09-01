"""Validate split-specific synthetic-data provenance."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROVENANCE_FORMAT = "mtam_hg_synthetic_provenance_v1"
TABDIFF_DETERMINISTIC_SEED = 0


class FileMutationError(RuntimeError):
    """Raised when a file changes while its snapshot is in use."""


class SyntheticProvenanceError(RuntimeError):
    """Raised when synthetic data cannot be tied to the active run protocol."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    sha256: str
    signature: tuple[int, int, int, int]
    content: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True)
class TableSnapshot:
    file: FileSnapshot
    frame: pd.DataFrame = field(repr=False)


class ValidatedPreparedInput(dict[str, object]):
    """Prepared metadata with the exact files it validated."""

    def __init__(
        self,
        metadata: Mapping[str, object],
        metadata_snapshot: FileSnapshot,
        prepared_snapshot: FileSnapshot,
        repo_snapshot: FileSnapshot,
    ) -> None:
        super().__init__(metadata)
        self.metadata_snapshot = metadata_snapshot
        self.prepared_snapshot = prepared_snapshot
        self.repo_snapshot = repo_snapshot

    def assert_current(self) -> None:
        assert_file_snapshot_current(self.metadata_snapshot, "prepared metadata")
        assert_file_snapshot_current(self.prepared_snapshot, "prepared training CSV")
        assert_file_snapshot_current(self.repo_snapshot, "TabDiff input CSV")


def _stat_signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def capture_file_snapshot(
    path: str | Path,
    *,
    include_content: bool = False,
) -> FileSnapshot:
    """Hash one stable file snapshot, retaining bytes only when requested."""
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if include_content else None
    total_size = 0
    try:
        with resolved.open("rb") as handle:
            before = _stat_signature(os.fstat(handle.fileno()))
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total_size += len(chunk)
                if chunks is not None:
                    chunks.append(chunk)
            after = _stat_signature(os.fstat(handle.fileno()))
        current = _stat_signature(resolved.stat())
    except OSError as exc:
        raise FileMutationError(f"Could not capture a stable file snapshot: {resolved}") from exc
    if before != after or after != current or total_size != after[2]:
        raise FileMutationError(f"File changed while being read: {resolved}")
    return FileSnapshot(
        path=resolved,
        sha256=digest.hexdigest(),
        signature=current,
        content=b"".join(chunks) if chunks is not None else None,
    )


def assert_file_snapshot_current(snapshot: FileSnapshot, label: str = "file") -> None:
    """Fail if a path no longer contains the captured bytes."""
    current = capture_file_snapshot(snapshot.path)
    if current.sha256 != snapshot.sha256:
        raise FileMutationError(f"{label} changed during use: {snapshot.path}")


def read_table_snapshot(path: str | Path) -> TableSnapshot:
    """Parse a table from the bytes used for its SHA-256 digest."""
    snapshot = capture_file_snapshot(path, include_content=True)
    assert snapshot.content is not None
    suffix = snapshot.path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(BytesIO(snapshot.content))
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(BytesIO(snapshot.content))
        else:
            raise ValueError(f"Unsupported table file type: {snapshot.path.suffix}")
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Unsupported table file type:"):
            raise
        raise RuntimeError(f"Could not parse table snapshot: {snapshot.path}") from exc
    return TableSnapshot(file=snapshot, frame=frame)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable value."""
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one stable file snapshot."""
    return capture_file_snapshot(path).sha256


def schema_sha256(columns: Sequence[object]) -> str:
    """Fingerprint an ordered model-table schema."""
    return canonical_sha256([str(column) for column in columns])


def sample_ids_sha256(sample_ids: Sequence[int] | np.ndarray) -> str:
    """Fingerprint ordered sample membership independently of NumPy dtype."""
    values = [int(value) for value in np.asarray(sample_ids).reshape(-1)]
    return canonical_sha256(values)


def split_fingerprints(
    train_ids: Sequence[int] | np.ndarray,
    val_ids: Sequence[int] | np.ndarray,
    test_ids: Sequence[int] | np.ndarray,
    *,
    total_rows: int,
) -> tuple[dict[str, str], str]:
    """Validate a complete partition and return per-split and combined hashes."""
    split_arrays = {
        "train": np.asarray(train_ids, dtype=np.int64).reshape(-1),
        "val": np.asarray(val_ids, dtype=np.int64).reshape(-1),
        "test": np.asarray(test_ids, dtype=np.int64).reshape(-1),
    }
    for name, values in split_arrays.items():
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Duplicate sample ids found inside the {name} split.")
        if np.any(values < 0) or np.any(values >= int(total_rows)):
            raise ValueError(f"Out-of-range sample ids found inside the {name} split.")

    names = tuple(split_arrays)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = np.intersect1d(split_arrays[left_name], split_arrays[right_name])
            if len(overlap):
                raise ValueError(
                    f"Split leakage detected between {left_name} and {right_name}: "
                    f"{len(overlap)} overlapping sample ids."
                )

    combined = np.concatenate(list(split_arrays.values()))
    expected = np.arange(int(total_rows), dtype=np.int64)
    if len(combined) != int(total_rows) or not np.array_equal(np.sort(combined), expected):
        raise ValueError("Train/validation/test ids do not form a complete source-data partition.")

    hashes = {
        name: sample_ids_sha256(values)
        for name, values in split_arrays.items()
    }
    combined_hash = canonical_sha256(
        {
            name: [int(value) for value in values]
            for name, values in split_arrays.items()
        }
    )
    return hashes, combined_hash


def generation_config_snapshot() -> dict[str, object]:
    """Capture the TabDiff settings that determine the generated table."""
    import config

    names = (
        "TABDIFF_DATANAME",
        "TABDIFF_EXP_NAME",
        "TABDIFF_NUM_SAMPLES",
        "TABDIFF_TRAIN_EPOCHS",
        "TABDIFF_CKPT_PATH",
        "TABDIFF_MECHANISM_CONSTRAINT",
        "TABDIFF_MECHANISM_LAMBDA",
        "TABDIFF_GUIDANCE_SCALE",
        "TABDIFF_MECHANISM_TEMPERATURE_HOLD_TOLERANCE",
        "TABDIFF_MECHANISM_YIELD_TOLERANCE",
        "TABDIFF_TRAINABLE_SCOPE",
        "TABDIFF_MIN_SAVE_EPOCH",
        "TABDIFF_FINETUNE_LR",
        "TABDIFF_FINETUNE_STEPS",
        "TABDIFF_NUM_TIMESTEPS_OVERRIDE",
        "TABDIFF_STOCHASTIC_SAMPLER",
        "TAIL_THRESHOLD_MODE",
        "TAIL_QUANTILE_LOW",
        "TAIL_QUANTILE_HIGH",
    )
    snapshot = {name: getattr(config, name, None) for name in names}
    snapshot["TABDIFF_GENERATION_SEED"] = int(
        getattr(config, "TABDIFF_GENERATION_SEED", TABDIFF_DETERMINISTIC_SEED)
    )
    snapshot["TABDIFF_DETERMINISTIC"] = True
    return snapshot


def synthetic_provenance_path(synthetic_path: str | Path) -> Path:
    """Return the JSON sidecar path for a synthetic table."""
    path = Path(synthetic_path)
    return path.with_suffix(".provenance.json")


def _project_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw.resolve()
    import config

    return (Path(config.PROJECT_ROOT) / raw).resolve()


def validate_prepared_tabdiff_input(metadata_path: str | Path) -> dict[str, object]:
    """Validate the exact CSV exposed to TabDiff before preprocessing it."""
    metadata_file = Path(metadata_path).resolve()
    try:
        metadata_snapshot = capture_file_snapshot(metadata_file, include_content=True)
        assert metadata_snapshot.content is not None
        metadata = json.loads(metadata_snapshot.content.decode("utf-8"))
    except (FileMutationError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticProvenanceError(f"Could not read prepared TabDiff metadata: {metadata_file}") from exc
    if not isinstance(metadata, Mapping):
        raise SyntheticProvenanceError("Prepared TabDiff metadata root must be a JSON object.")
    prepared_path = Path(str(metadata.get("tabdiff_project_train_csv", "")))
    try:
        prepared_snapshot = capture_file_snapshot(prepared_path)
    except FileMutationError as exc:
        raise SyntheticProvenanceError(f"Prepared TabDiff training CSV was not found: {prepared_path}") from exc
    _expect_equal(
        "prepared_train_csv_sha256",
        prepared_snapshot.sha256,
        metadata.get("prepared_train_csv_sha256"),
    )
    repo_csv = Path(str(metadata.get("tabdiff_repo_csv", "")))
    try:
        repo_snapshot = capture_file_snapshot(repo_csv)
    except FileMutationError as exc:
        raise SyntheticProvenanceError(f"TabDiff input CSV was not found: {repo_csv}") from exc
    _expect_equal(
        "tabdiff_repo_csv_sha256",
        repo_snapshot.sha256,
        metadata.get("tabdiff_repo_csv_sha256"),
    )
    _expect_equal(
        "TabDiff input copy",
        repo_snapshot.sha256,
        prepared_snapshot.sha256,
    )
    split_hashes = _require_mapping(metadata.get("split_id_sha256"), "split_id_sha256")
    _expect_equal(
        "generator_input_id_sha256",
        metadata.get("generator_input_id_sha256"),
        split_hashes.get("train"),
    )
    validated = ValidatedPreparedInput(
        metadata,
        metadata_snapshot,
        prepared_snapshot,
        repo_snapshot,
    )
    validated.assert_current()
    return validated


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SyntheticProvenanceError(f"Synthetic provenance field {name!r} is missing or invalid.")
    return value


def _load_sidecar(synthetic_path: Path) -> tuple[Path, Mapping[str, Any], FileSnapshot]:
    if not synthetic_path.is_file():
        raise SyntheticProvenanceError(f"Synthetic data file was not found: {synthetic_path}")
    sidecar_path = synthetic_provenance_path(synthetic_path)
    try:
        sidecar_snapshot = capture_file_snapshot(sidecar_path, include_content=True)
        assert sidecar_snapshot.content is not None
        payload = json.loads(sidecar_snapshot.content.decode("utf-8"))
    except FileMutationError as exc:
        raise SyntheticProvenanceError(
            "Synthetic provenance sidecar is required but was not found or changed: "
            f"{sidecar_path}. Regenerate the synthetic data for the active split."
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticProvenanceError(f"Could not read synthetic provenance: {sidecar_path}") from exc
    if not isinstance(payload, Mapping):
        raise SyntheticProvenanceError("Synthetic provenance root must be a JSON object.")
    if payload.get("format") != PROVENANCE_FORMAT:
        raise SyntheticProvenanceError(
            f"Unsupported synthetic provenance format: {payload.get('format')!r}."
        )
    return sidecar_path, payload, sidecar_snapshot


def _expect_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SyntheticProvenanceError(
            f"Synthetic provenance mismatch for {name}: expected {expected!r}, got {actual!r}. "
            "Regenerate the synthetic data for the active run protocol."
        )


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise SyntheticProvenanceError(
            f"Synthetic provenance field {name!r} must be a SHA-256 string."
        )
    return value


def _synthetic_layout(path: Path) -> tuple[int, list[str]]:
    try:
        frame = read_table_snapshot(path).frame
    except SyntheticProvenanceError:
        raise
    except Exception as exc:
        raise SyntheticProvenanceError(f"Could not inspect synthetic data: {path}") from exc
    return len(frame), [str(column) for column in frame.columns]


def _validate_common(
    synthetic_path: Path,
    *,
    source_sha256: str,
    split_seed: int,
    split_method: str,
    generation_seed: int,
    split_hashes: Mapping[str, str] | None = None,
    combined_split_sha256: str | None = None,
    schema_hash: str | None = None,
    validate_current_generation_config: bool = True,
    synthetic_snapshot: TableSnapshot | None = None,
) -> dict[str, object]:
    resolved_synthetic = synthetic_path.resolve()
    try:
        table_snapshot = synthetic_snapshot or read_table_snapshot(resolved_synthetic)
    except (FileMutationError, RuntimeError, ValueError) as exc:
        raise SyntheticProvenanceError(
            f"Could not capture synthetic data for validation: {resolved_synthetic}"
        ) from exc
    if table_snapshot.file.path != resolved_synthetic:
        raise SyntheticProvenanceError(
            "Synthetic table snapshot does not match the requested path: "
            f"{table_snapshot.file.path} != {resolved_synthetic}"
        )
    sidecar_path, payload, sidecar_snapshot = _load_sidecar(resolved_synthetic)
    source = _require_mapping(payload.get("source"), "source")
    split = _require_mapping(payload.get("split"), "split")
    generation = _require_mapping(payload.get("generation"), "generation")
    synthetic = _require_mapping(payload.get("synthetic"), "synthetic")

    synthetic_hash = _require_sha256(synthetic.get("sha256"), "synthetic.sha256")
    recorded_source_hash = _require_sha256(source.get("sha256"), "source.sha256")
    _expect_equal("synthetic.sha256", synthetic_hash, table_snapshot.file.sha256)
    _expect_equal("source.sha256", recorded_source_hash, source_sha256)
    actual_rows = len(table_snapshot.frame)
    actual_columns = [str(column) for column in table_snapshot.frame.columns]
    _expect_equal("synthetic.rows", actual_rows, synthetic.get("rows"))
    _expect_equal("synthetic.columns", actual_columns, synthetic.get("columns"))
    _expect_equal("split.seed", split.get("seed"), int(split_seed))
    _expect_equal("split.method", split.get("method"), str(split_method))
    _expect_equal("generation.backend", generation.get("backend"), "TabDiff")
    _expect_equal("generation.seed", generation.get("seed"), int(generation_seed))
    _expect_equal("generation.deterministic", generation.get("deterministic"), True)
    _expect_equal(
        "generation.backend_seed",
        generation.get("backend_seed"),
        TABDIFF_DETERMINISTIC_SEED,
    )
    checkpoint_path = str(generation.get("checkpoint_path", "") or "")
    if not checkpoint_path:
        raise SyntheticProvenanceError(
            "Synthetic provenance must record the exact checkpoint_path and checkpoint_sha256."
        )
    checkpoint_hash = _require_sha256(
        generation.get("checkpoint_sha256"),
        "generation.checkpoint_sha256",
    )
    _require_sha256(
        generation.get("prepared_csv_sha256"),
        "generation.prepared_csv_sha256",
    )
    _require_sha256(
        generation.get("raw_samples_sha256"),
        "generation.raw_samples_sha256",
    )
    checkpoint_snapshot: FileSnapshot | None = None
    if Path(checkpoint_path).is_file():
        checkpoint_snapshot = capture_file_snapshot(checkpoint_path)
        _expect_equal(
            "generation.checkpoint_sha256",
            checkpoint_hash,
            checkpoint_snapshot.sha256,
        )

    current_config = generation_config_snapshot()
    recorded_config = _require_mapping(generation.get("config"), "generation.config")
    recorded_config_hash = _require_sha256(
        generation.get("config_sha256"),
        "generation.config_sha256",
    )
    _expect_equal(
        "generation.config payload",
        canonical_sha256(dict(recorded_config)),
        recorded_config_hash,
    )
    if validate_current_generation_config:
        _expect_equal(
            "generation.config_sha256",
            generation.get("config_sha256"),
            canonical_sha256(current_config),
        )
    if schema_hash is not None:
        _expect_equal("source.schema_sha256", source.get("schema_sha256"), schema_hash)
    if split_hashes is not None:
        recorded_hashes = _require_mapping(split.get("id_sha256"), "split.id_sha256")
        for name in ("train", "val", "test"):
            _expect_equal(f"split.id_sha256.{name}", recorded_hashes.get(name), split_hashes[name])
        _expect_equal(
            "split.generator_input_id_sha256",
            split.get("generator_input_id_sha256"),
            split_hashes["train"],
        )
    if combined_split_sha256 is not None:
        _expect_equal(
            "split.combined_sha256",
            split.get("combined_sha256"),
            combined_split_sha256,
        )

    assert_file_snapshot_current(table_snapshot.file, "synthetic data")
    assert_file_snapshot_current(sidecar_snapshot, "synthetic provenance")
    if checkpoint_snapshot is not None:
        assert_file_snapshot_current(checkpoint_snapshot, "TabDiff checkpoint")

    return {
        "synthetic_path": str(resolved_synthetic),
        "provenance_path": str(sidecar_path),
        "synthetic_sha256": str(synthetic.get("sha256")),
        "source_sha256": str(source.get("sha256")),
        "combined_split_sha256": str(split.get("combined_sha256")),
        "generation_config_sha256": str(generation.get("config_sha256")),
    }


def validate_synthetic_provenance_for_runner(
    synthetic_path: str | Path,
    data_path: str | Path,
    split_seed: int,
    split_method: str,
    generation_seed: int,
    *,
    label_col: str | None = None,
    use_el_as_input: bool | None = None,
    validate_current_generation_config: bool = False,
) -> dict[str, object]:
    """Validate runner source, split, and synthetic provenance."""
    import config
    from dataset import (
        _resolve_data_path,
        _resolve_feature_columns,
        _resolve_label_column,
        _split_indices,
    )

    resolved_source = _resolve_data_path(data_path)
    source_snapshot = read_table_snapshot(resolved_source)
    frame = source_snapshot.frame.copy()
    frame.columns = [str(column) for column in frame.columns]
    label = _resolve_label_column(
        frame,
        getattr(config, "LABEL_COL", None) if label_col is None else label_col,
    )
    use_el = getattr(config, "USE_EL_AS_INPUT", False) if use_el_as_input is None else use_el_as_input
    feature_columns, _ = _resolve_feature_columns(
        frame,
        config.input_node_names(bool(use_el)),
    )
    labels = frame[[label]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    valid = ~np.isnan(labels.reshape(-1))
    labels = labels[valid]
    train_ids, val_ids, test_ids = _split_indices(labels, int(split_seed), str(split_method))
    hashes, combined_hash = split_fingerprints(
        train_ids,
        val_ids,
        test_ids,
        total_rows=len(labels),
    )
    result = _validate_common(
        _project_path(synthetic_path),
        source_sha256=source_snapshot.file.sha256,
        split_seed=int(split_seed),
        split_method=str(split_method),
        generation_seed=int(generation_seed),
        split_hashes=hashes,
        combined_split_sha256=combined_hash,
        schema_hash=schema_sha256([*feature_columns, label]),
        validate_current_generation_config=validate_current_generation_config,
    )
    assert_file_snapshot_current(source_snapshot.file, "source data")
    return result


def validate_synthetic_provenance_for_data_bundle(
    synthetic_path: str | Path,
    data_bundle: object,
    generation_seed: int,
    *,
    synthetic_snapshot: TableSnapshot | None = None,
) -> dict[str, object]:
    """Validate immediately before loading synthetic samples for training."""
    return _validate_common(
        _project_path(synthetic_path),
        source_sha256=str(getattr(data_bundle, "source_sha256")),
        split_seed=int(getattr(data_bundle, "split_seed")),
        split_method=str(getattr(data_bundle, "split_method")),
        generation_seed=int(generation_seed),
        split_hashes=getattr(data_bundle, "split_hashes"),
        combined_split_sha256=str(getattr(data_bundle, "combined_split_hash")),
        schema_hash=str(getattr(data_bundle, "schema_hash")),
        validate_current_generation_config=True,
        synthetic_snapshot=synthetic_snapshot,
    )
