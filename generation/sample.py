"""Run TabDiff sampling and collect the generated table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from generation.tabdiff import (
    check_tabdiff_dependencies,
    command_summary,
    copy_latest_sample,
    dataname,
    generation_seed,
    project_path,
    require_tabdiff_repo,
    run_command,
    sample_command,
    sample_output_snapshot,
    tabdiff_remote,
)
from protocol_integrity import assert_file_snapshot_current, capture_file_snapshot


def _raw_output_path() -> Path:
    return project_path(getattr(config, "TABDIFF_OUTPUT_DIR", "outputs/tabdiff")) / "synthetic_CAPL_raw.csv"


def run_tabdiff_sample(
    dry_run: bool = False,
    num_samples: int | None = None,
    checkpoint_path: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    repo = require_tabdiff_repo()
    ds_name = dataname()
    effective_checkpoint = checkpoint_path or getattr(config, "TABDIFF_CKPT_PATH", "") or None
    if not dry_run and effective_checkpoint is None:
        raise ValueError(
            "Sampling requires the exact checkpoint produced by the current TabDiff training phase; "
            "implicit checkpoint discovery is disabled."
        )
    checkpoint_snapshot = None
    if not dry_run:
        if expected_checkpoint_sha256 is None:
            raise ValueError("Sampling requires the checkpoint SHA-256 from the current training phase.")
        checkpoint_snapshot = capture_file_snapshot(project_path(effective_checkpoint))
        if checkpoint_snapshot.sha256 != str(expected_checkpoint_sha256).lower():
            raise RuntimeError("The sampling checkpoint does not match the current training phase.")
        effective_checkpoint = checkpoint_snapshot.path
    cmd = sample_command(
        repo,
        ds_name,
        num_samples=num_samples,
        checkpoint_path=effective_checkpoint,
    )
    deps_ok, deps_message = check_tabdiff_dependencies(repo)
    output_path = _raw_output_path()
    result = {
        "tabdiff_repo_path": str(repo),
        "tabdiff_remote": tabdiff_remote(repo),
        "dataset_name": ds_name,
        "dependencies_ok": deps_ok,
        "dependencies_message": deps_message,
        "commands": command_summary([cmd]),
        "raw_output_path": str(output_path),
        "sampling_mode": "unconditional",
        "generation_seed": generation_seed(),
        "deterministic": True,
        "checkpoint_path": str(effective_checkpoint or ""),
        "checkpoint_sha256": str(expected_checkpoint_sha256 or ""),
    }
    if dry_run:
        return result
    if not deps_ok:
        raise RuntimeError(deps_message)
    previous_snapshot = sample_output_snapshot(repo, ds_name)
    run_command(cmd)
    assert checkpoint_snapshot is not None
    assert_file_snapshot_current(checkpoint_snapshot, "TabDiff sampling checkpoint")
    copied = copy_latest_sample(
        repo,
        output_path,
        ds_name,
        previous_snapshot=previous_snapshot,
    )
    assert_file_snapshot_current(checkpoint_snapshot, "TabDiff sampling checkpoint")
    result["copied_from_official_output"] = str(copied)
    return result


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official TabDiff sampling.")
    parser.add_argument("--dry_run", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--checkpoint_sha256", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_tabdiff_sample(
        dry_run=args.dry_run,
        num_samples=args.num_samples,
        checkpoint_path=args.checkpoint_path or None,
        expected_checkpoint_sha256=args.checkpoint_sha256 or None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
