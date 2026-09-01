"""Run TabDiff preprocessing and mechanism-aware training."""

from __future__ import annotations

import argparse
import json

import config
from generation.tabdiff import (
    check_tabdiff_dependencies,
    checkpoint_output_snapshot,
    command_summary,
    dataname,
    find_fresh_checkpoint,
    generation_seed,
    process_command,
    processed_dataset_ready,
    project_path,
    require_tabdiff_repo,
    run_command,
    tabdiff_remote,
    train_command,
)
from protocol_integrity import (
    ValidatedPreparedInput,
    capture_file_snapshot,
    validate_prepared_tabdiff_input,
)


def run_tabdiff_train(dry_run: bool = False) -> dict[str, object]:
    repo = require_tabdiff_repo()
    ds_name = dataname()
    process_cmd = process_command(repo, ds_name)
    train_cmd = train_command(repo, ds_name)
    deps_ok, deps_message = check_tabdiff_dependencies(repo)
    ready_before_process = processed_dataset_ready(repo, ds_name)

    result = {
        "tabdiff_repo_path": str(repo),
        "tabdiff_remote": tabdiff_remote(repo),
        "readme_path": str(repo / "README.md"),
        "dataset_name": ds_name,
        "processed_dataset_ready": ready_before_process,
        "processed_dataset_ready_before_process": ready_before_process,
        "generation_seed": generation_seed(),
        "deterministic": True,
        "dependencies_ok": deps_ok,
        "dependencies_message": deps_message,
        "commands": command_summary([process_cmd, train_cmd]),
        "official_train_epochs_note": (
            "TabDiff does not expose an epoch CLI flag in the official entry point; "
            "training steps are controlled by third_party/TabDiff/tabdiff/configs/tabdiff_configs.toml."
        ),
    }
    if dry_run:
        return result
    if not deps_ok:
        raise RuntimeError(deps_message)
    metadata_path = project_path(getattr(config, "TABDIFF_DATA_DIR", "data/tabdiff")) / "capl_metadata.json"
    prepared_input = validate_prepared_tabdiff_input(metadata_path)
    previous_checkpoints = checkpoint_output_snapshot(repo, ds_name)
    run_command(process_cmd)
    if isinstance(prepared_input, ValidatedPreparedInput):
        prepared_input.assert_current()
    run_command(train_cmd)
    if isinstance(prepared_input, ValidatedPreparedInput):
        prepared_input.assert_current()
    checkpoint_path = find_fresh_checkpoint(repo, ds_name, previous_checkpoints)
    checkpoint_snapshot = capture_file_snapshot(checkpoint_path)
    result["checkpoint_path"] = str(checkpoint_path)
    result["checkpoint_sha256"] = checkpoint_snapshot.sha256
    return result


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official TabDiff training.")
    parser.add_argument("--dry_run", type=_str_to_bool, nargs="?", const=True, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_tabdiff_train(dry_run=args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
