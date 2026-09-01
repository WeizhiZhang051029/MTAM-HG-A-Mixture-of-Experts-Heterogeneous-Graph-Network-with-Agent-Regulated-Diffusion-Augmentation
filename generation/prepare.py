"""Prepare the fixed CAPL training split for TabDiff."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config
from dataset import (
    _read_table_snapshot,
    _resolve_data_path,
    _resolve_feature_columns,
    _resolve_label_column,
    _split_indices,
)
from generation.tabdiff import (
    dataname,
    project_path,
    require_tabdiff_repo,
    tabdiff_repo_path,
)
from protocol_integrity import (
    TABDIFF_DETERMINISTIC_SEED,
    assert_file_snapshot_current,
    canonical_sha256,
    capture_file_snapshot,
    generation_config_snapshot,
    schema_sha256,
    split_fingerprints,
)

PAPER_FEATURE_KEYS: tuple[str, ...] = (
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
)


def _read_table(path: str | Path) -> pd.DataFrame:
    return _read_table_snapshot(path).frame


def _clean_train_table(
    train_df: pd.DataFrame,
    numerical_columns: list[str],
    label_col: str,
) -> pd.DataFrame:
    out = train_df.copy()
    for col in [*numerical_columns, label_col]:
        values = pd.to_numeric(out[col], errors="coerce")
        fill = float(values.median()) if values.notna().any() else 0.0
        out[col] = values.fillna(fill).astype(np.float32)
    return out


def prepare_tabdiff_data(
    data_path: str | Path | None = None,
    label_col: str | None = None,
    output_dir: str | Path | None = None,
    repo_path: str | Path | None = None,
    dataset_name: str | None = None,
    split_seed: int | None = None,
    split_method: str | None = None,
    generation_seed: int | None = None,
) -> dict[str, object]:
    """Prepare the training split and TabDiff metadata."""
    resolved_data_path = _resolve_data_path(data_path or config.DATA_PATH)
    source_snapshot = _read_table_snapshot(resolved_data_path)
    df = source_snapshot.frame.copy()
    df.columns = [str(col) for col in df.columns]
    resolved_label = _resolve_label_column(df, label_col or getattr(config, "LABEL_COL", None))

    paper_feature_columns, feature_mapping = _resolve_feature_columns(
        df, list(PAPER_FEATURE_KEYS)
    )
    if len(set(paper_feature_columns)) != len(PAPER_FEATURE_KEYS):
        raise ValueError("Each paper feature must resolve to a distinct source column.")
    if resolved_label in paper_feature_columns:
        raise ValueError("The yield-strength label cannot also be used as a paper feature.")
    allowed_columns = [*paper_feature_columns, resolved_label]
    df = df.loc[:, allowed_columns].copy()

    label_values = pd.to_numeric(df[resolved_label], errors="coerce").to_numpy(dtype=np.float32).reshape(-1, 1)
    valid_label = ~np.isnan(label_values.reshape(-1))
    if not np.all(valid_label):
        df = df.loc[valid_label].reset_index(drop=True)
        label_values = label_values[valid_label]

    split_seed = int(getattr(config, "SPLIT_SEED", 42) if split_seed is None else split_seed)
    split_method = str(getattr(config, "SPLIT_METHOD", "stratified_random") if split_method is None else split_method)
    generation_seed = int(
        getattr(config, "TABDIFF_GENERATION_SEED", TABDIFF_DETERMINISTIC_SEED)
        if generation_seed is None
        else generation_seed
    )
    if generation_seed != TABDIFF_DETERMINISTIC_SEED:
        raise ValueError(
            "Official TabDiff --deterministic mode fixes its backend seed to 0; "
            f"generation_seed must therefore be 0, got {generation_seed}."
        )
    train_idx, val_idx, test_idx = _split_indices(label_values, split_seed, split_method)
    split_hashes, combined_split_hash = split_fingerprints(
        train_idx,
        val_idx,
        test_idx,
        total_rows=len(label_values),
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    numerical_columns = list(paper_feature_columns)
    categorical_columns: list[str] = []
    train_df = _clean_train_table(train_df, numerical_columns, resolved_label)

    columns = [*numerical_columns, resolved_label]
    train_df = train_df.loc[:, columns]
    label_idx = columns.index(resolved_label)
    num_idx = [columns.index(col) for col in numerical_columns]
    cat_idx = [columns.index(col) for col in categorical_columns]
    if label_idx in num_idx or label_idx in cat_idx:
        raise ValueError("TabDiff target_col_idx must be mutually exclusive with num_col_idx/cat_col_idx.")

    out_dir = project_path(output_dir or getattr(config, "TABDIFF_DATA_DIR", "data/tabdiff"))
    out_dir.mkdir(parents=True, exist_ok=True)
    project_train_csv = out_dir / "capl_train.csv"
    metadata_path = out_dir / "capl_metadata.json"
    columns_path = out_dir / "capl_columns.json"
    train_df.to_csv(project_train_csv, index=False, encoding="utf-8")
    prepared_snapshot = capture_file_snapshot(project_train_csv)
    prepared_train_hash = prepared_snapshot.sha256

    y_train = pd.to_numeric(train_df[resolved_label], errors="coerce").to_numpy(dtype=np.float64)
    low_q = float(getattr(config, "TAIL_QUANTILE_LOW", 0.10))
    high_q = float(getattr(config, "TAIL_QUANTILE_HIGH", 0.90))
    low_threshold = float(np.quantile(y_train, low_q))
    high_threshold = float(np.quantile(y_train, high_q))

    numerical_bounds = {
        col: {
            "min": float(pd.to_numeric(train_df[col], errors="coerce").min()),
            "max": float(pd.to_numeric(train_df[col], errors="coerce").max()),
        }
        for col in [*numerical_columns, resolved_label]
    }
    categorical_values = {
        col: sorted(str(value) for value in train_df[col].dropna().astype(str).unique().tolist())
        for col in categorical_columns
    }
    generation_config = generation_config_snapshot()
    generation_config["TABDIFF_GENERATION_SEED"] = generation_seed
    source_hash = source_snapshot.file.sha256
    schema_hash = schema_sha256(columns)
    metadata = {
        "dataset_name": dataset_name or dataname(),
        "source_data_path": str(resolved_data_path),
        "tabdiff_project_train_csv": str(project_train_csv),
        "prepared_train_csv_sha256": prepared_train_hash,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "label_col": resolved_label,
        "feature_columns": [col for col in columns if col != resolved_label],
        "paper_feature_keys": list(PAPER_FEATURE_KEYS),
        "paper_feature_mapping": feature_mapping,
        "all_columns": columns,
        "tail_threshold_mode": getattr(config, "TAIL_THRESHOLD_MODE", "train_quantile"),
        "tail_quantile_low": low_q,
        "tail_quantile_high": high_q,
        "low_tail_threshold": low_threshold,
        "high_tail_threshold": high_threshold,
        "source_data_sha256": source_hash,
        "schema_sha256": schema_hash,
        "split_method": split_method,
        "split_seed": split_seed,
        "split_id_sha256": split_hashes,
        "train_id_sha256": split_hashes["train"],
        "val_id_sha256": split_hashes["val"],
        "test_id_sha256": split_hashes["test"],
        "combined_split_sha256": combined_split_hash,
        "generator_input_id_sha256": split_hashes["train"],
        "generation_seed": generation_seed,
        "tabdiff_deterministic": True,
        "tabdiff_backend_seed": TABDIFF_DETERMINISTIC_SEED,
        "generation_config": generation_config,
        "generation_config_sha256": canonical_sha256(generation_config),
        "project_split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "numerical_bounds": numerical_bounds,
        "categorical_values": categorical_values,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    columns_info = {
        "columns": columns,
        "num_col_idx": num_idx,
        "cat_col_idx": cat_idx,
        "target_col_idx": [label_idx],
    }
    with columns_path.open("w", encoding="utf-8") as f:
        json.dump(columns_info, f, indent=2, ensure_ascii=False)

    repo = require_tabdiff_repo(repo_path or tabdiff_repo_path())
    ds_name = dataset_name or dataname()
    repo_data_dir = repo / "data" / ds_name
    repo_info_dir = repo / "data" / "Info"
    repo_data_dir.mkdir(parents=True, exist_ok=True)
    repo_info_dir.mkdir(parents=True, exist_ok=True)
    repo_csv = repo_data_dir / f"{ds_name}.csv"
    shutil.copy2(project_train_csv, repo_csv)
    repo_snapshot = capture_file_snapshot(repo_csv)
    if repo_snapshot.sha256 != prepared_snapshot.sha256:
        raise RuntimeError("TabDiff input copy changed while it was being prepared.")
    metadata["tabdiff_repo_csv"] = str(repo_csv)
    metadata["tabdiff_repo_csv_sha256"] = repo_snapshot.sha256
    assert_file_snapshot_current(source_snapshot.file, "source data")
    assert_file_snapshot_current(prepared_snapshot, "prepared training CSV")
    assert_file_snapshot_current(repo_snapshot, "TabDiff input CSV")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    info = {
        "name": ds_name,
        "task_type": "regression",
        "header": "infer",
        "column_names": columns,
        "num_col_idx": num_idx,
        "cat_col_idx": cat_idx,
        "target_col_idx": [label_idx],
        "file_type": "csv",
        "data_path": f"data/{ds_name}/{ds_name}.csv",
        "test_path": None,
        "val_path": None,
    }
    repo_info = repo_info_dir / f"{ds_name}.json"
    with repo_info.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    return {
        "project_train_csv": str(project_train_csv),
        "metadata_path": str(metadata_path),
        "columns_path": str(columns_path),
        "tabdiff_repo_csv": str(repo_csv),
        "tabdiff_info_json": str(repo_info),
        "dataset_name": ds_name,
        "train_rows": int(len(train_df)),
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "label_col": resolved_label,
        "low_tail_threshold": low_threshold,
        "high_tail_threshold": high_threshold,
        "source_data_sha256": source_hash,
        "prepared_train_csv_sha256": prepared_train_hash,
        "schema_sha256": schema_hash,
        "split_id_sha256": split_hashes,
        "combined_split_sha256": combined_split_hash,
        "generation_seed": generation_seed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CAPL train split for official TabDiff.")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--label_col", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--tabdiff_repo_path", default="")
    parser.add_argument("--dataname", default="")
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--split_method", choices=["stratified_random", "chronological"], default="")
    parser.add_argument("--generation_seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_tabdiff_data(
        data_path=args.data_path or None,
        label_col=args.label_col or None,
        output_dir=args.output_dir or None,
        repo_path=args.tabdiff_repo_path or None,
        dataset_name=args.dataname or None,
        split_seed=args.split_seed,
        split_method=args.split_method or None,
        generation_seed=args.generation_seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
