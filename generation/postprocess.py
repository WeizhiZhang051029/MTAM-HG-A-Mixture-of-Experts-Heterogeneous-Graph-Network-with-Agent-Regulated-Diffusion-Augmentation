"""Convert TabDiff samples to the CAPL training schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from generation.tabdiff import project_path
from protocol_integrity import (
    PROVENANCE_FORMAT,
    TABDIFF_DETERMINISTIC_SEED,
    ValidatedPreparedInput,
    assert_file_snapshot_current,
    canonical_sha256,
    capture_file_snapshot,
    generation_config_snapshot,
    read_table_snapshot,
    synthetic_provenance_path,
    validate_prepared_tabdiff_input,
)


def _default_metadata_path() -> Path:
    return project_path(getattr(config, "TABDIFF_DATA_DIR", "data/tabdiff")) / "capl_metadata.json"


def _default_raw_path() -> Path:
    return project_path(getattr(config, "TABDIFF_OUTPUT_DIR", "outputs/tabdiff")) / "synthetic_CAPL_raw.csv"


def _default_output_path() -> Path:
    return project_path(getattr(config, "SYNTHETIC_DATA_PATH", "data/synthetic_CAPL_ma_tabdiff.xlsx"))


def _read_samples(path: Path) -> pd.DataFrame:
    return read_table_snapshot(path).frame


def _save_samples(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported output file type: {path.suffix}")


def postprocess_tabdiff_samples(
    raw_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    output_path: str | Path | None = None,
    generation_condition: str | None = None,
    require_label: bool = True,
    checkpoint_path: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    if checkpoint_path is None:
        raise RuntimeError("Postprocessing requires the checkpoint from the current TabDiff run.")
    checkpoint_file = project_path(checkpoint_path).resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"TabDiff checkpoint was not found: {checkpoint_file}")
    checkpoint_snapshot = capture_file_snapshot(checkpoint_file)
    checkpoint_hash = checkpoint_snapshot.sha256
    if expected_checkpoint_sha256 is not None and checkpoint_hash != str(expected_checkpoint_sha256).lower():
        raise RuntimeError("The postprocessing checkpoint does not match the current training phase.")

    metadata_file = project_path(metadata_path or _default_metadata_path())
    metadata = validate_prepared_tabdiff_input(metadata_file)
    split_hashes = metadata.get("split_id_sha256")
    if not isinstance(split_hashes, dict) or set(split_hashes) != {"train", "val", "test"}:
        raise RuntimeError("Prepared TabDiff metadata does not contain complete split fingerprints.")
    raw_file = project_path(raw_path or _default_raw_path())
    out_file = project_path(output_path or _default_output_path())

    source_file = Path(str(metadata["source_data_path"]))
    if not source_file.is_file():
        raise FileNotFoundError(f"Synthetic provenance source file was not found: {source_file}")
    source_snapshot = capture_file_snapshot(source_file)
    source_hash = source_snapshot.sha256
    if source_hash != metadata.get("source_data_sha256"):
        raise RuntimeError(
            "The source table changed after TabDiff preparation. "
            "Prepare and generate the split-specific synthetic data again."
        )
    generation_seed = int(metadata.get("generation_seed", -1))
    if generation_seed != TABDIFF_DETERMINISTIC_SEED:
        raise RuntimeError(
            "TabDiff provenance requires official deterministic seed 0; "
            f"metadata records {generation_seed}."
        )
    generation_config = generation_config_snapshot()
    if canonical_sha256(generation_config) != metadata.get("generation_config_sha256"):
        raise RuntimeError(
            "TabDiff generation settings changed after data preparation. "
            "Prepare and generate the synthetic data again."
        )

    raw_snapshot = read_table_snapshot(raw_file)
    df = raw_snapshot.frame.copy()
    df = df.loc[:, [not str(col).startswith("Unnamed") for col in df.columns]]
    df.columns = [str(col) for col in df.columns]

    expected_columns = [str(col) for col in metadata["all_columns"]]
    if set(df.columns) == set(str(i) for i in range(len(expected_columns))):
        df = df.rename(columns={str(i): expected_columns[i] for i in range(len(expected_columns))})

    missing = [col for col in expected_columns if col not in df.columns]
    extra = [col for col in df.columns if col not in expected_columns]
    label_col = str(metadata["label_col"])
    if require_label and label_col in missing:
        raise KeyError(f"TabDiff synthetic samples do not contain required label column: {label_col}")
    if missing:
        raise KeyError(f"TabDiff synthetic samples are missing required columns: {missing}")
    df = df[expected_columns].copy()

    notes: dict[str, object] = {
        "raw_path": str(raw_file),
        "output_path": str(out_file),
        "dropped_extra_columns": extra,
        "categorical_invalid_counts": {},
        "continuous_clipped_counts": {},
    }
    categorical_values = metadata.get("categorical_values", {})
    if isinstance(categorical_values, dict):
        for col, legal in categorical_values.items():
            if col not in df.columns:
                continue
            legal_values = set(str(value) for value in legal)
            values = df[col].astype("string").fillna("nan").astype(str)
            invalid = ~values.isin(legal_values)
            notes["categorical_invalid_counts"][col] = int(invalid.sum())
            if invalid.any():
                replacement = next(iter(legal_values)) if legal_values else "nan"
                values.loc[invalid] = replacement
            df[col] = values

    bounds = metadata.get("numerical_bounds", {})
    if isinstance(bounds, dict):
        for col, limits in bounds.items():
            if col not in df.columns or not isinstance(limits, dict):
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            low = float(limits["min"])
            high = float(limits["max"])
            clipped = values.clip(lower=low, upper=high)
            notes["continuous_clipped_counts"][col] = int(((values < low) | (values > high)).sum())
            fill = float(np.nanmedian(clipped.to_numpy())) if clipped.notna().any() else low
            df[col] = clipped.fillna(fill)

    y = pd.to_numeric(df[label_col], errors="coerce") if label_col in df.columns else pd.Series([], dtype=float)
    low_threshold = float(metadata["low_tail_threshold"])
    high_threshold = float(metadata["high_tail_threshold"])
    is_tail = ((y <= low_threshold) | (y >= high_threshold)).astype(bool) if len(y) else False
    df["synthetic_source"] = "TabDiff"
    df["generation_condition"] = generation_condition or "unconditional"
    df["is_tail_synthetic"] = is_tail
    df["tabdiff_sample_id"] = np.arange(len(df), dtype=np.int64)

    _save_samples(df, out_file)
    output_snapshot = capture_file_snapshot(out_file)
    notes_path = out_file.with_suffix(".postprocess.json")
    with notes_path.open("w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)
    provenance_path = synthetic_provenance_path(out_file)
    provenance = {
        "format": PROVENANCE_FORMAT,
        "source": {
            "path": str(source_file.resolve()),
            "sha256": source_hash,
            "schema_sha256": str(metadata["schema_sha256"]),
        },
        "split": {
            "method": str(metadata["split_method"]),
            "seed": int(metadata["split_seed"]),
            "sizes": metadata["project_split_sizes"],
            "id_sha256": split_hashes,
            "combined_sha256": str(metadata["combined_split_sha256"]),
            "generator_input_id_sha256": str(metadata["generator_input_id_sha256"]),
            "generator_input_scope": "main_train_split_only",
        },
        "generation": {
            "backend": "TabDiff",
            "seed": generation_seed,
            "backend_seed": TABDIFF_DETERMINISTIC_SEED,
            "deterministic": True,
            "config": generation_config,
            "config_sha256": canonical_sha256(generation_config),
            "prepared_csv_sha256": str(metadata["prepared_train_csv_sha256"]),
            "raw_samples_sha256": raw_snapshot.file.sha256,
            "checkpoint_path": str(checkpoint_snapshot.path),
            "checkpoint_sha256": checkpoint_hash,
        },
        "synthetic": {
            "path": str(out_file.resolve()),
            "sha256": output_snapshot.sha256,
            "rows": int(len(df)),
            "columns": [str(column) for column in df.columns],
        },
    }
    temporary_path = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        assert_file_snapshot_current(checkpoint_snapshot, "TabDiff checkpoint")
        assert_file_snapshot_current(source_snapshot, "source data")
        assert_file_snapshot_current(raw_snapshot.file, "raw TabDiff samples")
        assert_file_snapshot_current(output_snapshot, "postprocessed synthetic data")
        if isinstance(metadata, ValidatedPreparedInput):
            metadata.assert_current()
        temporary_path.replace(provenance_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return {
        "raw_path": str(raw_file),
        "metadata_path": str(metadata_file),
        "output_path": str(out_file),
        "notes_path": str(notes_path),
        "provenance_path": str(provenance_path),
        "synthetic_sha256": provenance["synthetic"]["sha256"],
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Postprocess official TabDiff CAPL samples.")
    parser.add_argument("--raw_path", default="")
    parser.add_argument("--metadata_path", default="")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--generation_condition", default="")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--checkpoint_sha256", default="")
    parser.add_argument("--allow_missing_label", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = postprocess_tabdiff_samples(
        raw_path=args.raw_path or None,
        metadata_path=args.metadata_path or None,
        output_path=args.output_path or None,
        generation_condition=args.generation_condition or None,
        require_label=not args.allow_missing_label,
        checkpoint_path=args.checkpoint_path or None,
        expected_checkpoint_sha256=args.checkpoint_sha256 or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
