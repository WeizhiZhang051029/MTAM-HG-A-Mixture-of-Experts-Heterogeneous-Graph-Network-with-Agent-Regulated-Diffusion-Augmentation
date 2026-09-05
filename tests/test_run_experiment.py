"""Run-specific data partitioning and aggregation."""

from pathlib import Path

import pytest

from run_experiment import (
    build_main_train_command,
    parse_args,
    run_experiments,
    seed_template_path,
)


def test_run_seed_also_controls_partition(tmp_path: Path):
    args = parse_args(["--seeds", "7", "19", "--dry_run"])
    first = build_main_train_command(args, 7, tmp_path / "7", "syn_7.csv", 32)
    second = build_main_train_command(args, 19, tmp_path / "19", "syn_19.csv", 32)
    for command, seed in ((first, "7"), (second, "19")):
        assert command[command.index("--seed") + 1] == seed
        assert command[command.index("--split_seed") + 1] == seed
    assert first[first.index("--synthetic_data_path") + 1] != second[second.index("--synthetic_data_path") + 1]


def test_duplicate_seeds_rejected():
    with pytest.raises(ValueError, match="distinct"):
        parse_args(["--seeds", "7", "7"])


def test_cli_overrides_are_recorded_without_confirmatory_claim():
    args = parse_args(["--seeds", "7", "--epochs", "2", "--dry_run"])
    summary = run_experiments(args)
    assert summary["arguments"]["epochs"] == 2
    assert summary["protocol"] == "per_run_split"
    assert summary["feedback_source"] == "real_training_set"
    assert summary["aggregate"] == {}
    assert len(summary["commands"]) == 1


def test_synthetic_paths_are_distinct():
    assert seed_template_path("data/syn.csv", 7) == str(Path("data/syn_seed_7.csv"))
    assert seed_template_path("data/syn_{seed}.csv", 19) == "data/syn_19.csv"
