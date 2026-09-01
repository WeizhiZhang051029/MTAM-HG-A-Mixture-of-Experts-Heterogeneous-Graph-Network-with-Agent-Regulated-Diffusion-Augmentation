"""Run TabDiff preprocessing and mechanism-aware training."""

from __future__ import annotations

import argparse
import json

import config
from generation.tabdiff import (
    base_exp_name,
    base_train_command,
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
    assert_file_snapshot_current,
    capture_file_snapshot,
    validate_prepared_tabdiff_input,
)


def run_tabdiff_train(dry_run: bool = False) -> dict[str, object]:
    repo = require_tabdiff_repo()
    ds_name = dataname()
    process_cmd = process_command(repo, ds_name)
    external_checkpoint = str(getattr(config, "TABDIFF_CKPT_PATH", "") or "").strip()
    base_cmd = None if external_checkpoint else base_train_command(repo, ds_name)
    planned_checkpoint = (
        project_path(external_checkpoint)
        if external_checkpoint
        else repo / "tabdiff" / "ckpt" / ds_name / base_exp_name() / "<fresh-base-checkpoint>"
    )
    finetune_cmd = train_command(repo, ds_name, planned_checkpoint)
    commands = [process_cmd]
    if base_cmd is not None:
        commands.append(base_cmd)
    commands.append(finetune_cmd)
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
        "commands": command_summary(commands),
        "training_stages": ["base", "mechanism_finetune"] if base_cmd is not None else ["mechanism_finetune"],
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
    run_command(process_cmd)
    if isinstance(prepared_input, ValidatedPreparedInput):
        prepared_input.assert_current()

    if base_cmd is not None:
        previous_base = checkpoint_output_snapshot(repo, ds_name, base_exp_name())
        run_command(base_cmd)
        if isinstance(prepared_input, ValidatedPreparedInput):
            prepared_input.assert_current()
        base_checkpoint = find_fresh_checkpoint(repo, ds_name, previous_base, base_exp_name())
        base_snapshot = capture_file_snapshot(base_checkpoint)
    else:
        base_snapshot = capture_file_snapshot(project_path(external_checkpoint))
        base_checkpoint = base_snapshot.path
    result["base_checkpoint_path"] = str(base_checkpoint)
    result["base_checkpoint_sha256"] = base_snapshot.sha256

    previous_finetune = checkpoint_output_snapshot(repo, ds_name)
    finetune_cmd = train_command(repo, ds_name, base_checkpoint)
    executed_commands = [process_cmd]
    if base_cmd is not None:
        executed_commands.append(base_cmd)
    executed_commands.append(finetune_cmd)
    result["commands"] = command_summary(executed_commands)
    run_command(finetune_cmd)
    if isinstance(prepared_input, ValidatedPreparedInput):
        prepared_input.assert_current()
    assert_file_snapshot_current(base_snapshot, "TabDiff base checkpoint")
    checkpoint_path = find_fresh_checkpoint(repo, ds_name, previous_finetune)
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
