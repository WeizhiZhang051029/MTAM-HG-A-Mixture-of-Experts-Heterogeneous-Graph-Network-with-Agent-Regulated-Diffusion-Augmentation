"""Load and split CAPL tabular data."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config
from protocol_integrity import (
    FileSnapshot,
    TableSnapshot,
    assert_file_snapshot_current,
    canonical_sha256,
    read_table_snapshot,
    schema_sha256,
    split_fingerprints,
)


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = np.nanmean(x, axis=0, keepdims=True)
        self.std_ = np.nanstd(x, axis=0, keepdims=True)
        self.std_[self.std_ < 1.0e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return x * self.std_ + self.mean_


class CAPLDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, sample_ids: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sample_ids = torch.tensor(sample_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx], self.sample_ids[idx]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    standard_node_names: list[str]
    graph_node_names: list[str]
    feature_columns: list[str]
    column_mapping: dict[str, str]
    label_column: str
    tail_thresholds: tuple[float, float]
    y_train_raw: np.ndarray
    train_sample_ids: np.ndarray
    val_sample_ids: np.ndarray
    test_sample_ids: np.ndarray
    split_sizes: dict[str, int]
    use_el_as_input: bool
    split_method: str
    data_path: str
    model_seed: int = 0
    run_seed: int = 0
    split_seed: int = 0
    split_hashes: dict[str, str] = field(default_factory=dict)
    combined_split_hash: str = ""
    source_sha256: str = ""
    schema_hash: str = ""


def _normalize_column_name(name: object) -> str:
    return (
        str(name)
        .strip()
        .casefold()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def _resolve_data_path(path: str | Path) -> Path:
    raw = Path(path)
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        config.PROJECT_ROOT / raw,
        config.PROJECT_ROOT.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"CAPL data file was not found. Tried:\n{tried}")


def _read_table(path: str | Path) -> pd.DataFrame:
    return _read_table_snapshot(path).frame


def _read_table_snapshot(path: str | Path) -> TableSnapshot:
    return read_table_snapshot(_resolve_data_path(path))


def _match_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_to_col = {_normalize_column_name(col): str(col) for col in df.columns}
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized_to_col:
            return normalized_to_col[key]
    return None


def _resolve_feature_columns(df: pd.DataFrame, node_names: list[str]) -> tuple[list[str], dict[str, str]]:
    feature_columns: list[str] = []
    mapping: dict[str, str] = {}
    missing: dict[str, list[str]] = {}
    for node_name in node_names:
        candidates = [node_name, *config.COLUMN_ALIASES.get(node_name, [])]
        resolved = _match_column(df, candidates)
        if resolved is None:
            missing[node_name] = candidates
        else:
            feature_columns.append(resolved)
            mapping[node_name] = resolved

    if missing:
        lines = ["Missing required CAPL variable columns:"]
        for node_name, candidates in missing.items():
            lines.append(f"  - {node_name}: tried {candidates}")
        lines.append(f"Available columns: {[str(col) for col in df.columns]}")
        raise KeyError("\n".join(lines))

    return feature_columns, mapping


def _resolve_label_column(df: pd.DataFrame, label_column: str | None = None) -> str:
    candidates: list[str] = []
    if label_column:
        candidates.append(label_column)
    else:
        candidates.append(getattr(config, "LABEL_COL", ""))
    candidates.extend(getattr(config, "LABEL_ALIASES", []))
    resolved = _match_column(df, [candidate for candidate in candidates if candidate])
    if resolved is None:
        raise KeyError(
            "Could not resolve yield strength label column. "
            f"Tried {candidates}. Available columns: {[str(col) for col in df.columns]}"
        )
    return resolved


def _fill_missing(train: np.ndarray, *splits: np.ndarray) -> tuple[np.ndarray, ...]:
    if config.MISSING_VALUE_STRATEGY != "median":
        raise ValueError("Only median missing-value filling is implemented.")
    fill = np.nanmedian(train, axis=0, keepdims=True)
    fill = np.where(np.isnan(fill), 0.0, fill)
    out = []
    for split in (train, *splits):
        arr = np.array(split, copy=True)
        inds = np.where(np.isnan(arr))
        if inds[0].size:
            arr[inds] = np.take(fill.reshape(-1), inds[1])
        out.append(arr)
    return tuple(out)


def _chronological_split_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train, n_val, _ = _split_target_counts(n)
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def _split_target_counts(n: int) -> tuple[int, int, int]:
    if n <= 0:
        raise ValueError("Cannot split an empty dataset.")
    ratios = np.array([config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO], dtype=np.float64)
    if np.any(ratios < 0) or not np.isfinite(ratios).all() or ratios.sum() <= 0:
        raise ValueError("TRAIN_RATIO, VAL_RATIO and TEST_RATIO must be finite non-negative values with positive sum.")
    ratios = ratios / ratios.sum()
    raw = ratios * n
    counts = np.floor(raw).astype(np.int64)
    remainder = int(n - counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1

    nonzero = np.where(ratios > 0)[0]
    if n >= len(nonzero):
        for idx in nonzero:
            if counts[idx] == 0:
                donors = [j for j in nonzero if counts[j] > 1]
                if donors:
                    donor = max(donors, key=lambda j: counts[j])
                    counts[donor] -= 1
                    counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _allocate_by_capacity(capacities: np.ndarray, target: int) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.int64).reshape(-1)
    target = int(target)
    if target <= 0:
        return np.zeros_like(capacities)
    total_capacity = int(capacities.sum())
    if target >= total_capacity:
        return capacities.copy()
    raw = capacities.astype(np.float64) * (target / max(total_capacity, 1))
    counts = np.minimum(np.floor(raw).astype(np.int64), capacities)
    remainder = target - int(counts.sum())
    while remainder > 0:
        room = capacities - counts
        candidates = np.where(room > 0)[0]
        if len(candidates) == 0:
            break
        fractions = raw - np.floor(raw)
        best = max(candidates, key=lambda idx: (fractions[idx], room[idx], -idx))
        counts[best] += 1
        remainder -= 1
    return counts


def _stratified_random_split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified random split by binned yield strength."""
    y_flat = y.reshape(-1)
    n = len(y_flat)
    n_train_target, n_val_target, _ = _split_target_counts(n)
    q = min(config.STRATIFY_BINS, n)
    ranks = pd.Series(y_flat).rank(method="first")
    bins = pd.qcut(ranks, q=q, labels=False, duplicates="drop").to_numpy()
    rng = np.random.default_rng(seed)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    unique_bins = np.unique(bins)
    bin_indices: list[np.ndarray] = []
    for bin_id in unique_bins:
        bin_idx = np.where(bins == bin_id)[0].copy()
        rng.shuffle(bin_idx)
        bin_indices.append(bin_idx)

    capacities = np.array([len(idx) for idx in bin_indices], dtype=np.int64)
    train_counts = _allocate_by_capacity(capacities, n_train_target)
    remaining_after_train = capacities - train_counts
    val_counts = _allocate_by_capacity(remaining_after_train, n_val_target)

    for bin_idx, n_train, n_val in zip(bin_indices, train_counts, val_counts):
        train_parts.append(bin_idx[:n_train])
        val_parts.append(bin_idx[n_train:n_train + n_val])
        test_parts.append(bin_idx[n_train + n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _split_indices(y: np.ndarray, seed: int, split_method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_method == "chronological":
        return _chronological_split_indices(len(y))
    if split_method == "stratified_random":
        return _stratified_random_split_indices(y, seed)
    raise ValueError("SPLIT_METHOD must be 'stratified_random' or 'chronological'.")


def _load_real_arrays_with_snapshot(
    data_path: str | Path,
    node_names: list[str],
    label_column: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str], str, dict[str, str], str, FileSnapshot]:
    resolved_path = _resolve_data_path(data_path)
    table_snapshot = _read_table_snapshot(resolved_path)
    df = table_snapshot.frame
    feature_columns, mapping = _resolve_feature_columns(df, node_names)
    resolved_label = _resolve_label_column(df, label_column)
    x = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y = df[[resolved_label]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    valid_label = ~np.isnan(y.reshape(-1))
    if not np.all(valid_label):
        dropped = int((~valid_label).sum())
        print(f"[Data] Dropping {dropped} rows with missing label values.")
        x = x[valid_label]
        y = y[valid_label]

    return (
        x,
        y,
        feature_columns,
        resolved_label,
        mapping,
        str(table_snapshot.file.path),
        table_snapshot.file,
    )


def _load_real_arrays(
    data_path: str | Path,
    node_names: list[str],
    label_column: str | None,
) -> tuple[np.ndarray, np.ndarray, list[str], str, dict[str, str], str]:
    return _load_real_arrays_with_snapshot(data_path, node_names, label_column)[:-1]


def _make_synthetic_arrays(
    node_names: list[str],
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], str, dict[str, str], str]:
    rng = np.random.default_rng(seed)
    n = len(node_names)
    x = rng.normal(0.0, 1.0, size=(n_samples, n)).astype(np.float32)
    index = {name: i for i, name in enumerate(node_names)}

    def col(name: str) -> np.ndarray | float:
        return x[:, index[name]] if name in index else 0.0

    y = (
        360.0
        + 18.0 * col("C")
        + 8.0 * col("Mn")
        - 5.0 * col("S")
        + 7.5 * col("CRR")
        + 6.0 * col("ATh")
        - 3.5 * col("AWd")
        + 5.0 * col("FS")
        + 3.0 * col("Q_T")
        + rng.laplace(0.0, 5.0, size=n_samples)
    ).astype(np.float32)
    mapping = {name: name for name in node_names}
    return x, y.reshape(-1, 1), list(node_names), "synthetic_yield_strength", mapping, "synthetic"


def _print_data_summary(bundle: DataBundle) -> None:
    print(f"[Data] Source: {bundle.data_path}")
    print(f"[Data] Label column: {bundle.label_column}")
    print(f"[Data] USE_EL_AS_INPUT={bundle.use_el_as_input}; EL {'included' if bundle.use_el_as_input else 'excluded'}")
    print(f"[Data] Standard input variables ({len(bundle.standard_node_names)}): {bundle.standard_node_names}")
    print(f"[Data] Graph nodes ({len(bundle.graph_node_names)}): {bundle.graph_node_names}")
    print(f"[Data] Column mapping: {bundle.column_mapping}")
    print(
        "[Data] Split "
        f"method={bundle.split_method}; "
        f"train={bundle.split_sizes['train']}, "
        f"val={bundle.split_sizes['val']}, "
        f"test={bundle.split_sizes['test']}"
    )


def create_dataloaders(
    data_path: str | Path | None = None,
    label_column: str | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    use_el_as_input: bool | None = None,
    save_scaler_path: str | Path | None = None,
    split_method: str | None = None,
    allow_synthetic: bool = False,
    split_seed: int | None = None,
) -> DataBundle:
    seed = config.SEED if seed is None else seed
    split_seed = int(getattr(config, "SPLIT_SEED", seed) if split_seed is None else split_seed)
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    split_method = config.SPLIT_METHOD if split_method is None else split_method
    use_el = config.USE_EL_AS_INPUT if use_el_as_input is None else use_el_as_input
    input_names = config.input_node_names(use_el)
    graph_node_names = config.active_node_names(use_el)

    source_snapshot: FileSnapshot | None = None
    if data_path:
        (
            x,
            y,
            feature_columns,
            resolved_label,
            mapping,
            source,
            source_snapshot,
        ) = _load_real_arrays_with_snapshot(data_path, input_names, label_column)
    elif allow_synthetic:
        print("[Data][WARN] Using synthetic CAPL data for development only.")
        x, y, feature_columns, resolved_label, mapping, source = _make_synthetic_arrays(
            input_names, config.SYNTHETIC_NUM_SAMPLES, seed
        )
    else:
        raise ValueError(
            "A real CAPL data file is required. Pass --data_path \"path\\to\\CAPL.xlsx\". "
            "Use --allow_synthetic only for code-level development smoke tests."
        )

    train_idx, val_idx, test_idx = _split_indices(y, split_seed, split_method)
    split_hashes, combined_split_hash = split_fingerprints(
        train_idx,
        val_idx,
        test_idx,
        total_rows=len(y),
    )
    source_hash = (
        source_snapshot.sha256
        if source_snapshot is not None
        else canonical_sha256({"source": source, "model_seed": int(seed), "rows": int(len(y))})
    )
    schema_hash = schema_sha256([*feature_columns, resolved_label])
    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    x_train, x_val, x_test = _fill_missing(x_train, x_val, x_test)
    y_train, y_val, y_test = _fill_missing(y_train, y_val, y_test)
    y_train_raw = np.array(y_train, copy=True)
    tail_lower = float(np.quantile(y_train.reshape(-1), config.TAIL_QUANTILE))
    tail_upper = float(np.quantile(y_train.reshape(-1), 1.0 - config.TAIL_QUANTILE))

    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    if config.STANDARDIZE_X:
        x_train, x_val, x_test = x_scaler.transform(x_train), x_scaler.transform(x_val), x_scaler.transform(x_test)
    if config.STANDARDIZE_Y:
        y_train, y_val, y_test = y_scaler.transform(y_train), y_scaler.transform(y_val), y_scaler.transform(y_test)

    split_sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    if source_snapshot is not None:
        assert_file_snapshot_current(source_snapshot, "CAPL source data")
    save_path = Path(save_scaler_path or config.SCALER_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump(
            {
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "standard_node_names": input_names,
                "graph_node_names": graph_node_names,
                "column_mapping": mapping,
                "label_column": resolved_label,
                "split_method": split_method,
                "split_seed": split_seed,
                "model_seed": int(seed),
                "run_seed": int(seed),
                "split_hashes": split_hashes,
                "combined_split_hash": combined_split_hash,
                "source_sha256": source_hash,
                "schema_hash": schema_hash,
                "split_sizes": split_sizes,
                "use_el_as_input": use_el,
            },
            f,
        )

    train_loader = DataLoader(
        CAPLDataset(x_train, y_train, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        CAPLDataset(x_val, y_val, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        CAPLDataset(x_test, y_test, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    bundle = DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        standard_node_names=input_names,
        graph_node_names=graph_node_names,
        feature_columns=feature_columns,
        column_mapping=mapping,
        label_column=resolved_label,
        tail_thresholds=(tail_lower, tail_upper),
        y_train_raw=y_train_raw,
        train_sample_ids=train_idx,
        val_sample_ids=val_idx,
        test_sample_ids=test_idx,
        split_sizes=split_sizes,
        use_el_as_input=use_el,
        split_method=split_method,
        data_path=source,
        model_seed=int(seed),
        run_seed=int(seed),
        split_seed=split_seed,
        split_hashes=split_hashes,
        combined_split_hash=combined_split_hash,
        source_sha256=source_hash,
        schema_hash=schema_hash,
    )
    _print_data_summary(bundle)
    return bundle
