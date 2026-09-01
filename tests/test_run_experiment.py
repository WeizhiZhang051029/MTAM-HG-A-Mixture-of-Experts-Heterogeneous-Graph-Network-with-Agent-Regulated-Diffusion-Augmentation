from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from config_loader import load_yaml_config  # noqa: E402
from generation.tabdiff import require_tabdiff_repo, train_command  # noqa: E402
from pipeline import load_config_overrides  # noqa: E402
from protocol import (  # noqa: E402
    DEFAULT_GENERATION_SEED,
    DEFAULT_SEEDS,
    DEFAULT_SPLIT_METHOD,
    DEFAULT_SPLIT_SEED,
    MR_LORA_ARG_SPECS,
)
from protocol_integrity import canonical_sha256, file_sha256  # noqa: E402
from run_experiment import (  # noqa: E402
    MAIN_TRAIN_ARG_SPECS,
    bind_metrics_to_effective_protocol,
    build_main_train_command,
    build_parser,
    build_tabdiff_generation_command,
    effective_protocol_sha256,
    effective_protocol_snapshot,
    load_seed_metrics,
    metrics_integrity_fields,
    metrics_output_snapshot,
    require_file_sha256,
    require_fresh_seed_metrics,
    require_loaded_config_sha256,
    scientific_producer_protocol_sha256,
    scientific_producer_protocol_snapshot,
    validate_args,
    validate_declared_protocol,
    validate_runner_integrity,
    write_runner_summary,
)

TEST_EFFECTIVE_PROTOCOL = {"format": "test_effective_protocol", "locked": True}
TEST_EFFECTIVE_PROTOCOL_SHA256 = canonical_sha256(TEST_EFFECTIVE_PROTOCOL)
TEST_PRODUCER_PROTOCOL_SHA256 = "6" * 64

TEST_INTEGRITY_FIELDS = {
    "Split_Seed": 42,
    "Split_Method": "stratified_random",
    "Combined_Split_SHA256": "1" * 64,
    "Source_Data_SHA256": "2" * 64,
    "Synthetic_SHA256": "3" * 64,
    "Generation_Seed": 0,
    "Config_SHA256": "4" * 64,
    "Effective_Protocol_SHA256": TEST_EFFECTIVE_PROTOCOL_SHA256,
    "CBTG_Cross_Run_Validation_STD_SHA256": "5" * 64,
    "Synthetic_Provenance_SHA256": "7" * 64,
}


def _with_integrity(metrics: dict[str, object]) -> dict[str, object]:
    return {"TAIL_MAE": 2.5, **metrics, **TEST_INTEGRITY_FIELDS}


def _summary_protocol_kwargs() -> dict[str, object]:
    return {
        "effective_protocol": TEST_EFFECTIVE_PROTOCOL,
        "effective_protocol_hash": TEST_EFFECTIVE_PROTOCOL_SHA256,
    }


class RunMainExperimentTest(unittest.TestCase):
    def test_yaml_protocol_must_match_locked_runner(self) -> None:
        valid = load_yaml_config(PROJECT_ROOT / "configs" / "mtam_hg.yaml")
        validate_declared_protocol(valid)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_declared_protocol({**valid, "dropout": 0.3})
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_declared_protocol({**valid, "model_seeds": DEFAULT_SEEDS[:-1]})

    def test_loaded_config_hash_must_match_runner_snapshot(self) -> None:
        with patch.object(config, "CONFIG_SHA256", "b" * 64):
            with self.assertRaisesRegex(RuntimeError, "changed while loading"):
                require_loaded_config_sha256("a" * 64)

    def test_config_hash_lock_rejects_mid_run_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("split_seed: 42\n", encoding="utf-8")
            expected = file_sha256(path)
            path.write_text("split_seed: 43\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                require_file_sha256(path, expected, "Experiment config")

    def test_effective_protocol_hash_covers_yaml_runner_runtime_and_cross_run_artifact(self) -> None:
        args = build_parser().parse_args([])
        cross_run = {
            "sha256": "5" * 64,
            "payload": {"source": "independent_validation_runs", "num_runs": 10},
        }
        first = effective_protocol_snapshot({"b": 2, "a": 1}, args, 5000, cross_run)
        reordered = effective_protocol_snapshot({"a": 1, "b": 2}, args, 5000, cross_run)

        self.assertEqual(effective_protocol_sha256(first), effective_protocol_sha256(reordered))
        changed_args = build_parser().parse_args(["--output_root", "outputs/alternate"])
        changed = effective_protocol_snapshot({"a": 1, "b": 2}, changed_args, 5000, cross_run)
        self.assertNotEqual(effective_protocol_sha256(first), effective_protocol_sha256(changed))

    def test_producer_protocol_excludes_operations_but_covers_scientific_runtime(self) -> None:
        base_args = build_parser().parse_args([])
        moved_args = build_parser().parse_args(
            [
                "--output_root",
                "outputs/alternate",
                "--cbtg_cross_run_validation_std_path",
                "data/alternate.json",
                "--tabdiff_gpu",
                "1",
            ]
        )
        config_values = {
            "epochs": 50,
            "output_base": "outputs/mtam_hg",
            "cbtg_cross_run_validation_std_path": "data/reference.json",
        }
        runtime = {"LR": 1.0e-3, "OUTPUT_DIR": "outputs/one"}
        first = scientific_producer_protocol_snapshot(
            config_values,
            base_args,
            5000,
            runtime,
            "a" * 64,
        )
        moved = scientific_producer_protocol_snapshot(
            {**config_values, "output_base": "outputs/two"},
            moved_args,
            5000,
            {"LR": 1.0e-3, "OUTPUT_DIR": "outputs/two"},
            "a" * 64,
        )
        changed = scientific_producer_protocol_snapshot(
            config_values,
            base_args,
            5000,
            {"LR": 5.0e-4, "OUTPUT_DIR": "outputs/one"},
            "a" * 64,
        )

        self.assertEqual(
            scientific_producer_protocol_sha256(first),
            scientific_producer_protocol_sha256(moved),
        )
        self.assertNotEqual(
            scientific_producer_protocol_sha256(first),
            scientific_producer_protocol_sha256(changed),
        )
        alternate_seeds = build_parser().parse_args(
            ["--seeds", *[str(seed) for seed in range(100, 110)]]
        )
        seed_independent = scientific_producer_protocol_snapshot(
            {**config_values, "model_seeds": list(range(100, 110))},
            alternate_seeds,
            5000,
            runtime,
            "a" * 64,
        )
        self.assertEqual(
            scientific_producer_protocol_sha256(first),
            scientific_producer_protocol_sha256(seed_independent),
        )

    def test_metrics_are_bound_to_effective_protocol_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            base_integrity = {
                key: value
                for key, value in TEST_INTEGRITY_FIELDS.items()
                if key != "Effective_Protocol_SHA256"
            }
            metrics_path.write_text(
                json.dumps(
                    {
                        "RMSE": 1.0,
                        "MAE": 0.8,
                        "MAPE": 3.4,
                        "R2": 0.9,
                        "TAIL_MAE": 2.5,
                        "Model": "mtam_hg",
                        "Experiment_Group": "mtam_hg_paper",
                        "Seed": 42,
                        **base_integrity,
                    }
                ),
                encoding="utf-8",
            )

            bound_hash = bind_metrics_to_effective_protocol(
                metrics_path,
                42,
                TEST_INTEGRITY_FIELDS,
            )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            current_hash = file_sha256(metrics_path)

        self.assertEqual(bound_hash, current_hash)
        self.assertEqual(payload["Effective_Protocol_SHA256"], TEST_EFFECTIVE_PROTOCOL_SHA256)
        self.assertEqual(payload["Synthetic_Provenance_SHA256"], "7" * 64)
        self.assertEqual(payload["CBTG_Cross_Run_Validation_STD_SHA256"], "5" * 64)

    def test_parent_rejects_metrics_without_child_provenance_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            payload = _with_integrity({"Seed": 42})
            payload.pop("Synthetic_Provenance_SHA256")
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Synthetic_Provenance_SHA256"):
                bind_metrics_to_effective_protocol(metrics_path, 42, TEST_INTEGRITY_FIELDS)

    def test_parent_rejects_metrics_without_child_cross_run_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            payload = _with_integrity(
                {
                    "RMSE": 1.0,
                    "MAE": 0.8,
                    "MAPE": 3.4,
                    "R2": 0.9,
                    "Model": "mtam_hg",
                    "Experiment_Group": "mtam_hg_paper",
                    "Seed": 42,
                }
            )
            payload.pop("Effective_Protocol_SHA256")
            payload.pop("CBTG_Cross_Run_Validation_STD_SHA256")
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "CBTG_Cross_Run_Validation_STD_SHA256"):
                bind_metrics_to_effective_protocol(metrics_path, 42, TEST_INTEGRITY_FIELDS)

    def test_parser_accepts_current_synthetic_agent_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "--synthetic_agent_epochs",
                "7",
                "--synthetic_agent_lr",
                "0.0003",
                "--synthetic_agent_hidden_dim",
                "96",
                "--synthetic_agent_attention_dim",
                "48",
                "--synthetic_agent_attention_heads",
                "4",
                "--synthetic_agent_dropout",
                "0.2",
                "--synthetic_confidence_threshold",
                "0.55",
                "--synthetic_pretrain_confidence_threshold",
                "0.0",
                "--use_dynamic_synthetic_agent",
                "--dynamic_synthetic_refresh_epochs",
                "4",
                "--dynamic_synthetic_warmup_epochs",
                "2",
                "--dynamic_synthetic_use_sampler",
                "False",
                "--dynamic_synthetic_use_loss_weight",
                "True",
                "--dynamic_synthetic_top_ratio",
                "0.6",
                "--dynamic_synthetic_weight_min",
                "0.1",
                "--dynamic_synthetic_weight_max",
                "2.5",
                "--dynamic_synthetic_ema",
                "0.5",
                "--dynamic_synthetic_real_feedback_weight",
                "0.4",
                "--dynamic_synthetic_quota_strength",
                "0.6",
                "--dynamic_synthetic_quota_min",
                "0.4",
                "--dynamic_synthetic_quota_max",
                "1.6",
                "--dynamic_synthetic_reliability_floor",
                "0.3",
                "--dynamic_synthetic_scarcity_bins",
                "6",
            ]
        )

        self.assertEqual(args.synthetic_agent_epochs, 7)
        self.assertEqual(args.synthetic_agent_lr, 0.0003)
        self.assertEqual(args.synthetic_agent_hidden_dim, 96)
        self.assertEqual(args.synthetic_agent_attention_dim, 48)
        self.assertEqual(args.synthetic_agent_attention_heads, 4)
        self.assertEqual(args.synthetic_agent_dropout, 0.2)
        self.assertEqual(args.synthetic_confidence_threshold, 0.55)
        self.assertEqual(args.synthetic_pretrain_confidence_threshold, 0.0)
        self.assertTrue(args.use_dynamic_synthetic_agent)
        self.assertEqual(args.dynamic_synthetic_refresh_epochs, 4)
        self.assertEqual(args.dynamic_synthetic_warmup_epochs, 2)
        self.assertFalse(args.dynamic_synthetic_use_sampler)
        self.assertTrue(args.dynamic_synthetic_use_loss_weight)
        self.assertEqual(args.dynamic_synthetic_top_ratio, 0.6)
        self.assertEqual(args.dynamic_synthetic_weight_min, 0.1)
        self.assertEqual(args.dynamic_synthetic_weight_max, 2.5)
        self.assertEqual(args.dynamic_synthetic_ema, 0.5)
        self.assertEqual(args.dynamic_synthetic_real_feedback_weight, 0.4)
        self.assertEqual(args.dynamic_synthetic_quota_strength, 0.6)
        self.assertEqual(args.dynamic_synthetic_quota_min, 0.4)
        self.assertEqual(args.dynamic_synthetic_quota_max, 1.6)
        self.assertEqual(args.dynamic_synthetic_reliability_floor, 0.3)
        self.assertEqual(args.dynamic_synthetic_scarcity_bins, 6)

        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.seeds, DEFAULT_SEEDS)
        self.assertEqual(defaults.split_seed, DEFAULT_SPLIT_SEED)
        self.assertEqual(defaults.split_method, DEFAULT_SPLIT_METHOD)
        self.assertEqual(defaults.generation_seed, DEFAULT_GENERATION_SEED)
        self.assertEqual(defaults.config, "configs/mtam_hg.yaml")
        self.assertEqual(defaults.synthetic_confidence_threshold, 0.5)
        self.assertEqual(defaults.synthetic_agent_epochs, 100)
        self.assertEqual(
            defaults.cbtg_cross_run_validation_std_path,
            "data/cbtg_cross_run_validation_std.json",
        )
        self.assertTrue(defaults.use_dynamic_synthetic_agent)
        self.assertTrue(defaults.dynamic_synthetic_use_sampler)
        self.assertTrue(defaults.dynamic_synthetic_use_loss_weight)
        self.assertEqual(defaults.mr_lora_scope, "graph_attention_routing")
        self.assertEqual(defaults.dynamic_synthetic_train_reward_metric, "rmse_tail")

    def test_mr_lora_specs_are_not_part_of_shared_main_train_passthrough(self) -> None:
        main_attrs = {attr for attr, _flag in MAIN_TRAIN_ARG_SPECS}
        mr_lora_attrs = {attr for attr, _flag in MR_LORA_ARG_SPECS}

        self.assertFalse(main_attrs & mr_lora_attrs)

    def test_tabdiff_generation_command_is_explicit_pipeline_phase(self) -> None:
        args = build_parser().parse_args(["--tabdiff_gpu", "0"])

        cmd = build_tabdiff_generation_command(
            args,
            tabdiff_num_samples=5000,
            scientific_code_hash="a" * 64,
            generation_protocol_hash="b" * 64,
        )

        self.assertIn("--mode", cmd)
        self.assertIn("generate_synthetic_tabdiff", cmd)
        self.assertIn("--tabdiff_num_samples", cmd)
        self.assertIn("5000", cmd)
        self.assertEqual(args.synthetic_data_path, "data/synthetic_CAPL_ma_tabdiff.xlsx")
        self.assertIn("--synthetic_data_path", cmd)
        self.assertIn("data/synthetic_CAPL_ma_tabdiff.xlsx", cmd)
        self.assertEqual(cmd[cmd.index("--seed") + 1], "0")
        self.assertEqual(cmd[cmd.index("--split_seed") + 1], "42")
        self.assertEqual(cmd[cmd.index("--split_method") + 1], "stratified_random")
        self.assertEqual(cmd[cmd.index("--generation_seed") + 1], "0")
        self.assertEqual(cmd[cmd.index("--tabdiff_gpu") + 1], "0")
        self.assertEqual(cmd[cmd.index("--scientific_code_sha256") + 1], "a" * 64)
        self.assertEqual(cmd[cmd.index("--generation_protocol_sha256") + 1], "b" * 64)

    def test_main_command_covers_paper_aligned_mtam_hg_training_chain(self) -> None:
        args = build_parser().parse_args(["--synthetic_pretrain_confidence_threshold", "0.0"])

        main_cmd = build_main_train_command(
            args,
            seed=42,
            run_dir=Path("outputs/mtam_hg/seed_42"),
            synthetic_path="data/synthetic_CAPL_ma_tabdiff.xlsx",
            tabdiff_num_samples=5000,
            scientific_code_hash="a" * 64,
            generation_protocol_hash="b" * 64,
        )

        self.assertIn("train_with_tabdiff_pretrain", main_cmd)
        self.assertIn("--synthetic_data_path", main_cmd)
        self.assertEqual(main_cmd[main_cmd.index("--synthetic_data_path") + 1], "data/synthetic_CAPL_ma_tabdiff.xlsx")
        self.assertIn("--synthetic_pretrain_epochs", main_cmd)
        self.assertEqual(main_cmd[main_cmd.index("--seed") + 1], "42")
        self.assertEqual(main_cmd[main_cmd.index("--split_seed") + 1], "42")
        self.assertEqual(main_cmd[main_cmd.index("--split_method") + 1], "stratified_random")
        self.assertEqual(main_cmd[main_cmd.index("--generation_seed") + 1], "0")
        self.assertIn("--synthetic_pretrain_confidence_threshold", main_cmd)
        self.assertIn("--dynamic_synthetic_use_sampler", main_cmd)
        self.assertEqual(main_cmd[main_cmd.index("--dynamic_synthetic_use_sampler") + 1], "True")
        self.assertIn("--dynamic_synthetic_use_loss_weight", main_cmd)
        self.assertEqual(main_cmd[main_cmd.index("--dynamic_synthetic_use_loss_weight") + 1], "True")
        self.assertIn("--dynamic_synthetic_real_feedback_weight", main_cmd)
        self.assertIn("--use_layerwise_finetune_lr", main_cmd)
        self.assertIn("--freeze_finetune_backbone", main_cmd)
        self.assertIn("--use_mr_lora", main_cmd)
        self.assertEqual(main_cmd[main_cmd.index("--mr_lora_scope") + 1], "graph_attention_routing")
        self.assertEqual(main_cmd[main_cmd.index("--mr_lora_rank_graph") + 1], "8")
        self.assertEqual(main_cmd[main_cmd.index("--mr_lora_rank_routing") + 1], "4")
        self.assertIn("--use_cluster_balance_reward", main_cmd)
        self.assertEqual(main_cmd[main_cmd.index("--synthetic_agent_epochs") + 1], "100")
        self.assertEqual(
            main_cmd[main_cmd.index("--cbtg_cross_run_validation_std_path") + 1],
            "data/cbtg_cross_run_validation_std.json",
        )
        self.assertNotIn("--skip_agent_filter", main_cmd)
        self.assertEqual(main_cmd[main_cmd.index("--scientific_code_sha256") + 1], "a" * 64)
        self.assertEqual(main_cmd[main_cmd.index("--generation_protocol_sha256") + 1], "b" * 64)
        self.assertNotIn("--no_synthetic_resampling", main_cmd)

    def test_runner_summary_collects_final_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            seed_run_dirs = {}
            seed_metrics_paths = {}
            for seed in DEFAULT_SEEDS:
                run_dir = output_root / f"seed_{seed}"
                seed_run_dirs[seed] = run_dir
                results_dir = run_dir / "20260603_000000" / "results"
                results_dir.mkdir(parents=True)
                metrics_path = results_dir / "metrics.json"
                metrics_path.write_text(
                    json.dumps(
                        _with_integrity({
                            "RMSE": 1.2 + seed / 1000,
                            "MAE": 0.8,
                            "MAPE": 3.4,
                            "R2": 0.9,
                            "TAIL_MAE": 2.5,
                            "Best_Epoch": 3,
                            "Epochs_Run": 4,
                            "Model": "mtam_hg",
                            "Experiment_Group": "mtam_hg_paper",
                            "Seed": seed,
                        })
                    ),
                    encoding="utf-8",
                )
                seed_metrics_paths[seed] = metrics_path

            summary_path = write_runner_summary(
                output_root,
                phases=[{"name": "seed 42 main-train"}],
                seed_run_dirs=seed_run_dirs,
                dry_run=False,
                use_dynamic_synthetic_agent=True,
                seed_metrics_paths=seed_metrics_paths,
                seed_metrics_sha256={seed: file_sha256(path) for seed, path in seed_metrics_paths.items()},
                integrity_fields=TEST_INTEGRITY_FIELDS,
                **_summary_protocol_kwargs(),
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["main_train_mode"], "train_with_tabdiff_pretrain")
        self.assertNotIn("use_synthetic_resampling", summary)
        self.assertNotIn("skip_agent_filter", summary)
        self.assertTrue(summary["use_dynamic_synthetic_agent"])
        self.assertEqual(len(summary["seeds"]), 10)
        self.assertEqual(summary["seeds"][0]["metrics_seed"], 42)
        self.assertEqual(summary["aggregate_mean_std"]["RMSE"]["n"], 10)
        self.assertEqual(summary["aggregate_mean_std"]["TAIL_MAE"]["n"], 10)
        self.assertEqual(summary["effective_protocol_sha256"], TEST_EFFECTIVE_PROTOCOL_SHA256)
        self.assertEqual(summary["evaluation_protocol"], "confirmatory_fixed_seed")
        self.assertEqual(summary["model_seeds"], DEFAULT_SEEDS)
        self.assertEqual(summary["split_seed"], DEFAULT_SPLIT_SEED)
        self.assertEqual(summary["split_method"], DEFAULT_SPLIT_METHOD)
        self.assertEqual(summary["generation_seed"], DEFAULT_GENERATION_SEED)
        self.assertTrue(summary["confirmatory_protocol"]["validated"])
        self.assertTrue(summary["confirmatory_protocol"]["shared_main_split"])
        self.assertEqual(summary["confirmatory_protocol"]["metrics_integrity"], TEST_INTEGRITY_FIELDS)
        self.assertIn("mechanism-aware TabDiff rows are used directly as the synthetic pretraining table", summary["pipeline"])
        self.assertIn(
            "MoE-IPOHGN pretraining on synthetic samples with the K-means cluster-balanced CBTG-Agent policy",
            summary["pipeline"],
        )

    def test_dry_run_summary_is_not_marked_as_validated_confirmatory_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            summary_path = write_runner_summary(
                output_root,
                phases=[],
                seed_run_dirs={42: output_root / "seed_42"},
                dry_run=True,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["evaluation_protocol"], "confirmatory_dry_run")
        self.assertFalse(summary["confirmatory_protocol"]["validated"])

    def test_load_seed_metrics_uses_current_main_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_42"
            final_results = run_dir / "20260603_000002" / "results"
            final_results.mkdir(parents=True)
            (final_results / "metrics.json").write_text(
                json.dumps(
                    _with_integrity({
                        "RMSE": 1.0,
                        "MAE": 0.8,
                        "MAPE": 3.4,
                        "R2": 0.9,
                        "Model": "mtam_hg",
                        "Experiment_Group": "mtam_hg_paper",
                        "Seed": 42,
                    })
                ),
                encoding="utf-8",
            )

            loaded = load_seed_metrics(
                final_results / "metrics.json",
                expected_seed=42,
                expected_integrity=TEST_INTEGRITY_FIELDS,
            )

        self.assertEqual(loaded["metrics"]["RMSE"], 1.0)

    def test_load_seed_metrics_never_falls_back_to_wrong_seed_or_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_42"
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True)
            (results_dir / "metrics.json").write_text(
                json.dumps(
                    _with_integrity({
                        "RMSE": 0.01,
                        "MAE": 0.01,
                        "MAPE": 0.01,
                        "R2": 0.99,
                        "Model": "mtam_hg",
                        "Experiment_Group": "mtam_hg_paper",
                        "Seed": 99,
                    })
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "No confirmatory MTAM-HG metrics matched"):
                load_seed_metrics(
                    results_dir / "metrics.json",
                    expected_seed=42,
                    expected_integrity=TEST_INTEGRITY_FIELDS,
                )

    def test_load_seed_metrics_requires_all_five_finite_paper_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_42"
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True)
            (results_dir / "metrics.json").write_text(
                json.dumps(
                    _with_integrity({
                        "RMSE": 1.0,
                        "MAE": 0.8,
                        "MAPE": float("nan"),
                        "R2": 0.9,
                        "Model": "mtam_hg",
                        "Experiment_Group": "mtam_hg_paper",
                        "Seed": 42,
                    })
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "finite metrics"):
                load_seed_metrics(
                    results_dir / "metrics.json",
                    expected_seed=42,
                    expected_integrity=TEST_INTEGRITY_FIELDS,
                )

    def test_load_seed_metrics_rejects_missing_tail_mae(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            payload = _with_integrity(
                {
                    "RMSE": 1.0,
                    "MAE": 0.8,
                    "MAPE": 3.4,
                    "R2": 0.9,
                    "Model": "mtam_hg",
                    "Experiment_Group": "mtam_hg_paper",
                    "Seed": 42,
                }
            )
            payload.pop("TAIL_MAE")
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "TAIL_MAE"):
                load_seed_metrics(
                    metrics_path,
                    expected_seed=42,
                    expected_integrity=TEST_INTEGRITY_FIELDS,
                )

    def test_fresh_metrics_rejects_success_without_a_new_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_42"
            old_metrics = run_dir / "20260801_000000" / "results" / "metrics.json"
            old_metrics.parent.mkdir(parents=True)
            old_metrics.write_text(
                json.dumps(
                    {
                        "RMSE": 1.0,
                        "MAE": 0.8,
                        "MAPE": 3.4,
                        "R2": 0.9,
                        "Model": "mtam_hg",
                        "Experiment_Group": "mtam_hg_paper",
                        "Seed": 42,
                    }
                ),
                encoding="utf-8",
            )
            before = metrics_output_snapshot(run_dir)

            with self.assertRaisesRegex(RuntimeError, "produce exactly one fresh metrics.json"):
                require_fresh_seed_metrics(
                    run_dir,
                    before,
                    expected_seed=42,
                    expected_integrity=TEST_INTEGRITY_FIELDS,
                )

    def test_summary_uses_exact_fresh_paths_even_when_old_metrics_have_newer_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            seed_run_dirs = {}
            seed_metrics_paths = {}
            for seed in DEFAULT_SEEDS:
                run_dir = output_root / f"seed_{seed}"
                seed_run_dirs[seed] = run_dir
                current_metrics = run_dir / "20260901_000000" / "results" / "metrics.json"
                old_metrics = run_dir / "20260801_000000" / "results" / "metrics.json"
                current_metrics.parent.mkdir(parents=True)
                old_metrics.parent.mkdir(parents=True)
                common = {
                    "MAE": 0.8,
                    "MAPE": 3.4,
                    "R2": 0.9,
                    "TAIL_MAE": 2.5,
                    "Model": "mtam_hg",
                    "Experiment_Group": "mtam_hg_paper",
                    "Seed": seed,
                    **TEST_INTEGRITY_FIELDS,
                }
                current_metrics.write_text(json.dumps({**common, "RMSE": 1.2}), encoding="utf-8")
                old_metrics.write_text(json.dumps({**common, "RMSE": 0.01}), encoding="utf-8")
                future = time.time() + 3600
                os.utime(old_metrics, (future, future))
                seed_metrics_paths[seed] = current_metrics

            summary_path = write_runner_summary(
                output_root,
                phases=[],
                seed_run_dirs=seed_run_dirs,
                dry_run=False,
                seed_metrics_paths=seed_metrics_paths,
                seed_metrics_sha256={seed: file_sha256(path) for seed, path in seed_metrics_paths.items()},
                integrity_fields=TEST_INTEGRITY_FIELDS,
                **_summary_protocol_kwargs(),
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["seeds"][0]["RMSE"], 1.2)
        self.assertIn("20260901_000000", summary["seeds"][0]["metrics_path"])

    def test_fresh_metrics_rejects_an_invalid_exact_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_42"
            before = metrics_output_snapshot(run_dir)
            metrics_path = run_dir / "20260901_000000" / "results" / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(
                json.dumps(
                    _with_integrity({
                        "RMSE": 1.0,
                        "MAE": 0.8,
                        "MAPE": 3.4,
                        "R2": 0.9,
                        "Model": "mtam_hg",
                        "Experiment_Group": "mtam_hg_paper",
                        "Seed": 99,
                    })
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                require_fresh_seed_metrics(
                    run_dir,
                    before,
                    expected_seed=42,
                    expected_integrity=TEST_INTEGRITY_FIELDS,
                )

    def test_summary_rejects_metrics_changed_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            seed_run_dirs: dict[int, Path] = {}
            seed_metrics_paths: dict[int, Path] = {}
            for seed in DEFAULT_SEEDS:
                run_dir = output_root / f"seed_{seed}"
                metrics_path = run_dir / "results" / "metrics.json"
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(
                    json.dumps(
                        _with_integrity(
                            {
                                "RMSE": 1.0,
                                "MAE": 0.8,
                                "MAPE": 3.4,
                                "R2": 0.9,
                                "Model": "mtam_hg",
                                "Experiment_Group": "mtam_hg_paper",
                                "Seed": seed,
                            }
                        )
                    ),
                    encoding="utf-8",
                )
                seed_run_dirs[seed] = run_dir
                seed_metrics_paths[seed] = metrics_path
            seed_metrics_sha256 = {
                seed: file_sha256(path) for seed, path in seed_metrics_paths.items()
            }
            changed_path = seed_metrics_paths[DEFAULT_SEEDS[0]]
            changed = json.loads(changed_path.read_text(encoding="utf-8"))
            changed["RMSE"] = 0.01
            changed_path.write_text(json.dumps(changed), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "changed after validation"):
                write_runner_summary(
                    output_root,
                    phases=[],
                    seed_run_dirs=seed_run_dirs,
                    dry_run=False,
                    seed_metrics_paths=seed_metrics_paths,
                    seed_metrics_sha256=seed_metrics_sha256,
                    integrity_fields=TEST_INTEGRITY_FIELDS,
                    **_summary_protocol_kwargs(),
                )

    def test_fresh_metrics_rejects_multiple_new_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_42"
            before = metrics_output_snapshot(run_dir)
            payload = {
                "RMSE": 1.0,
                "MAE": 0.8,
                "MAPE": 3.4,
                "R2": 0.9,
                "TAIL_MAE": 2.5,
                "Model": "mtam_hg",
                "Experiment_Group": "mtam_hg_paper",
                "Seed": 42,
                **TEST_INTEGRITY_FIELDS,
            }
            for timestamp in ("20260901_000000", "20260901_000001"):
                metrics_path = run_dir / timestamp / "results" / "metrics.json"
                metrics_path.parent.mkdir(parents=True)
                metrics_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "seed 42 produced 2"):
                require_fresh_seed_metrics(
                    run_dir,
                    before,
                    expected_seed=42,
                    expected_integrity=TEST_INTEGRITY_FIELDS,
                )

    def test_metrics_reject_wrong_split_source_and_synthetic_identity(self) -> None:
        wrong_values = {
            "Split_Seed": 43,
            "Split_Method": "chronological",
            "Combined_Split_SHA256": "4" * 64,
            "Source_Data_SHA256": "5" * 64,
            "Synthetic_SHA256": "6" * 64,
            "Generation_Seed": 1,
            "Config_SHA256": "7" * 64,
            "Effective_Protocol_SHA256": "8" * 64,
            "CBTG_Cross_Run_Validation_STD_SHA256": "9" * 64,
        }
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            for field, wrong_value in wrong_values.items():
                with self.subTest(field=field):
                    payload = _with_integrity(
                        {
                            "RMSE": 1.0,
                            "MAE": 0.8,
                            "MAPE": 3.4,
                            "R2": 0.9,
                            "Model": "mtam_hg",
                            "Experiment_Group": "mtam_hg_paper",
                            "Seed": 42,
                        }
                    )
                    payload[field] = wrong_value
                    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, field):
                        load_seed_metrics(
                            metrics_path,
                            expected_seed=42,
                            expected_integrity=TEST_INTEGRITY_FIELDS,
                        )

    @patch("run_experiment.validate_cbtg_cross_run_validation_std")
    @patch("run_experiment.validate_synthetic_provenance_for_runner")
    def test_runner_integrity_rejects_runtime_source_or_synthetic_change(
        self,
        mock_validate,
        mock_cross_run,
    ) -> None:
        mock_cross_run.return_value = {"sha256": "5" * 64}
        initial_provenance = {
            "combined_split_sha256": "1" * 64,
            "source_sha256": "2" * 64,
            "synthetic_sha256": "3" * 64,
            "provenance_sha256": "7" * 64,
        }
        for provenance_field, metrics_field in (
            ("source_sha256", "Source_Data_SHA256"),
            ("synthetic_sha256", "Synthetic_SHA256"),
            ("provenance_sha256", "Synthetic_Provenance_SHA256"),
        ):
            with self.subTest(field=metrics_field):
                changed_provenance = {**initial_provenance, provenance_field: "4" * 64}
                mock_validate.side_effect = [initial_provenance, changed_provenance]
                expected = validate_runner_integrity(
                    Path("synthetic.xlsx"),
                    Path("source.xlsx"),
                    42,
                    "stratified_random",
                    0,
                    "4" * 64,
                    TEST_EFFECTIVE_PROTOCOL_SHA256,
                    TEST_PRODUCER_PROTOCOL_SHA256,
                    Path("cross_run.json"),
                    "5" * 64,
                    label_col="屈服强度",
                    scientific_code_hash="a" * 64,
                )

                with self.assertRaisesRegex(RuntimeError, metrics_field):
                    validate_runner_integrity(
                        Path("synthetic.xlsx"),
                        Path("source.xlsx"),
                        42,
                        "stratified_random",
                        0,
                        "4" * 64,
                        TEST_EFFECTIVE_PROTOCOL_SHA256,
                        TEST_PRODUCER_PROTOCOL_SHA256,
                        Path("cross_run.json"),
                        "5" * 64,
                        label_col="屈服强度",
                        scientific_code_hash="a" * 64,
                        expected_fields=expected,
                    )

    @patch("run_experiment.validate_cbtg_cross_run_validation_std")
    @patch("run_experiment.validate_synthetic_provenance_for_runner")
    def test_runner_integrity_passes_label_and_input_mode(self, mock_validate, mock_cross_run) -> None:
        mock_validate.return_value = {
            "combined_split_sha256": "1" * 64,
            "source_sha256": "2" * 64,
            "synthetic_sha256": "3" * 64,
            "provenance_sha256": "7" * 64,
        }
        mock_cross_run.return_value = {"sha256": "5" * 64}

        validate_runner_integrity(
            Path("synthetic.xlsx"),
            Path("source.xlsx"),
            42,
            "stratified_random",
            0,
            "4" * 64,
            TEST_EFFECTIVE_PROTOCOL_SHA256,
            TEST_PRODUCER_PROTOCOL_SHA256,
            Path("cross_run.json"),
            "5" * 64,
            label_col="target_column",
            scientific_code_hash="a" * 64,
        )

        mock_validate.assert_called_once_with(
            Path("synthetic.xlsx"),
            Path("source.xlsx"),
            42,
            "stratified_random",
            0,
            label_col="target_column",
            use_el_as_input=False,
            validate_current_generation_config=True,
            expected_scientific_code_sha256="a" * 64,
            expected_generation_protocol_sha256=TEST_PRODUCER_PROTOCOL_SHA256,
        )
        mock_cross_run.assert_called_once_with(
            Path("cross_run.json"),
            expected_split_sha256="1" * 64,
            expected_source_data_sha256="2" * 64,
            expected_config_sha256="4" * 64,
            expected_producer_protocol_sha256=TEST_PRODUCER_PROTOCOL_SHA256,
        )

    def test_summary_rejects_one_seed_with_a_different_integrity_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            seed_run_dirs = {}
            seed_metrics_paths = {}
            for seed in DEFAULT_SEEDS:
                run_dir = output_root / f"seed_{seed}"
                seed_run_dirs[seed] = run_dir
                metrics_path = run_dir / "results" / "metrics.json"
                metrics_path.parent.mkdir(parents=True)
                payload = _with_integrity(
                    {
                        "RMSE": 1.0,
                        "MAE": 0.8,
                        "MAPE": 3.4,
                        "R2": 0.9,
                        "Model": "mtam_hg",
                        "Experiment_Group": "mtam_hg_paper",
                        "Seed": seed,
                    }
                )
                if seed == DEFAULT_SEEDS[-1]:
                    payload["Synthetic_SHA256"] = "4" * 64
                metrics_path.write_text(json.dumps(payload), encoding="utf-8")
                seed_metrics_paths[seed] = metrics_path

            with self.assertRaisesRegex(ValueError, "Synthetic_SHA256"):
                write_runner_summary(
                    output_root,
                    phases=[],
                    seed_run_dirs=seed_run_dirs,
                    dry_run=False,
                    seed_metrics_paths=seed_metrics_paths,
                    seed_metrics_sha256={seed: file_sha256(path) for seed, path in seed_metrics_paths.items()},
                    integrity_fields=TEST_INTEGRITY_FIELDS,
                    **_summary_protocol_kwargs(),
                )

    def test_metrics_integrity_fields_use_runner_provenance_hashes(self) -> None:
        fields = metrics_integrity_fields(
            {
                "combined_split_sha256": "1" * 64,
                "source_sha256": "2" * 64,
                "synthetic_sha256": "3" * 64,
                "provenance_sha256": "7" * 64,
            },
            42,
            "stratified_random",
            0,
            "4" * 64,
            TEST_EFFECTIVE_PROTOCOL_SHA256,
            "5" * 64,
        )

        self.assertEqual(fields, TEST_INTEGRITY_FIELDS)

    def test_non_dry_summary_requires_all_ten_confirmatory_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "requires ten distinct model seeds"):
                write_runner_summary(
                    output_root,
                    phases=[],
                    seed_run_dirs={42: output_root / "seed_42"},
                    dry_run=False,
                )

    def test_validate_args_rejects_invalid_current_controls(self) -> None:
        validate_args(build_parser().parse_args([]))
        for invalid_seeds in (
            [42],
            [*DEFAULT_SEEDS, 52],
            [42, 42, *DEFAULT_SEEDS[2:]],
        ):
            with self.subTest(invalid_seeds=invalid_seeds), self.assertRaisesRegex(ValueError, "ten distinct"):
                validate_args(build_parser().parse_args(["--seeds", *map(str, invalid_seeds)]))
        validate_args(build_parser().parse_args(["--seeds", *map(str, reversed(DEFAULT_SEEDS))]))
        validate_args(build_parser().parse_args(["--seeds", *map(str, range(1, 11))]))
        with self.assertRaisesRegex(ValueError, "split seed is fixed"):
            validate_args(build_parser().parse_args(["--split_seed", "43"]))
        with self.assertRaisesRegex(ValueError, "split method is fixed"):
            validate_args(build_parser().parse_args(["--split_method", "chronological"]))
        with self.assertRaisesRegex(ValueError, "generation seed is fixed"):
            validate_args(build_parser().parse_args(["--generation_seed", "42"]))
        with self.assertRaisesRegex(ValueError, "main_experiment_name"):
            validate_args(build_parser().parse_args(["--main_experiment_name", "alternate_run"]))
        with self.assertRaisesRegex(ValueError, "--epochs"):
            validate_args(build_parser().parse_args(["--epochs", "51"]))
        with self.assertRaisesRegex(ValueError, "--lr"):
            validate_args(build_parser().parse_args(["--lr", "0.0005"]))
        with self.assertRaisesRegex(ValueError, "--use_dynamic_synthetic_agent"):
            validate_args(build_parser().parse_args(["--no_dynamic_synthetic_agent"]))
        with self.assertRaisesRegex(ValueError, "--tabdiff_num_samples"):
            validate_args(build_parser().parse_args(["--tabdiff_num_samples", "4999"]))
        with self.assertRaises(ValueError):
            validate_args(build_parser().parse_args(["--dynamic_synthetic_refresh_epochs", "0"]))
        with self.assertRaises(ValueError):
            validate_args(build_parser().parse_args(["--dynamic_synthetic_scarcity_bins", "1"]))
        with self.assertRaises(ValueError):
            validate_args(build_parser().parse_args(["--dynamic_synthetic_top_ratio", "0"]))
        with self.assertRaises(ValueError):
            validate_args(build_parser().parse_args(["--dynamic_synthetic_weight_max", "0.01", "--dynamic_synthetic_weight_min", "0.2"]))
        with self.assertRaises(ValueError):
            validate_args(build_parser().parse_args(["--dynamic_synthetic_quota_strength", "1.1"]))
        with self.assertRaises(ValueError):
            validate_args(build_parser().parse_args(["--dynamic_synthetic_quota_max", "0.4", "--dynamic_synthetic_quota_min", "0.5"]))
        with self.assertRaises(ValueError):
            validate_args(
                build_parser().parse_args(
                    ["--synthetic_agent_attention_dim", "10", "--synthetic_agent_attention_heads", "4"]
                )
            )

    def test_tabdiff_train_command_includes_mechanism_finetune_args(self) -> None:
        repo = require_tabdiff_repo()
        original = {
            "TABDIFF_MECHANISM_CONSTRAINT": getattr(config, "TABDIFF_MECHANISM_CONSTRAINT", False),
            "TABDIFF_MECHANISM_LAMBDA": getattr(config, "TABDIFF_MECHANISM_LAMBDA", 0.05),
            "TABDIFF_GUIDANCE_SCALE": getattr(config, "TABDIFF_GUIDANCE_SCALE", 0.0),
            "TABDIFF_MECHANISM_TEMPERATURE_HOLD_TOLERANCE": getattr(
                config,
                "TABDIFF_MECHANISM_TEMPERATURE_HOLD_TOLERANCE",
                10.0,
            ),
            "TABDIFF_TRAINABLE_SCOPE": getattr(config, "TABDIFF_TRAINABLE_SCOPE", "mlp_detokenizer"),
        }
        try:
            config.TABDIFF_MECHANISM_CONSTRAINT = True
            config.TABDIFF_MECHANISM_LAMBDA = 0.1
            config.TABDIFF_GUIDANCE_SCALE = 0.05
            config.TABDIFF_MECHANISM_TEMPERATURE_HOLD_TOLERANCE = 12.5
            config.TABDIFF_TRAINABLE_SCOPE = "mlp"

            cmd = train_command(repo).command

            self.assertIn("--mechanism_constraint", cmd)
            self.assertIn("--mechanism_lambda", cmd)
            self.assertIn("0.1", cmd)
            self.assertIn("--guidance_scale", cmd)
            self.assertIn("0.05", cmd)
            self.assertIn("--mechanism_temperature_hold_tolerance", cmd)
            self.assertEqual(cmd[cmd.index("--mechanism_temperature_hold_tolerance") + 1], "12.5")
            self.assertIn("--trainable_scope", cmd)
            self.assertIn("mlp", cmd)
            self.assertIn("--reset_train_epoch", cmd)
        finally:
            for key, value in original.items():
                setattr(config, key, value)

    def test_full_pipeline_yaml_uses_direct_dynamic_sampler_pipeline(self) -> None:
        original = {
            "TABDIFF_EXP_NAME": getattr(config, "TABDIFF_EXP_NAME", "capl_tabdiff"),
            "SYNTHETIC_DATA_PATH": getattr(config, "SYNTHETIC_DATA_PATH", ""),
            "SYNTHETIC_CONFIDENCE_THRESHOLD": getattr(config, "SYNTHETIC_CONFIDENCE_THRESHOLD", 0.9),
            "USE_DYNAMIC_SYNTHETIC_AGENT": getattr(config, "USE_DYNAMIC_SYNTHETIC_AGENT", True),
            "DYNAMIC_SYNTHETIC_REFRESH_EPOCHS": getattr(config, "DYNAMIC_SYNTHETIC_REFRESH_EPOCHS", 0),
            "DYNAMIC_SYNTHETIC_WARMUP_EPOCHS": getattr(config, "DYNAMIC_SYNTHETIC_WARMUP_EPOCHS", 0),
            "DYNAMIC_SYNTHETIC_USE_SAMPLER": getattr(config, "DYNAMIC_SYNTHETIC_USE_SAMPLER", False),
            "DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT": getattr(config, "DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT", False),
            "DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC": getattr(config, "DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC", ""),
            "USE_MR_LORA": getattr(config, "USE_MR_LORA", False),
            "MR_LORA_SCOPE": getattr(config, "MR_LORA_SCOPE", ""),
            "MR_LORA_RANK_GRAPH": getattr(config, "MR_LORA_RANK_GRAPH", 0),
            "MR_LORA_RANK_ROUTING": getattr(config, "MR_LORA_RANK_ROUTING", 0),
            "USE_CLUSTER_BALANCE_REWARD": getattr(config, "USE_CLUSTER_BALANCE_REWARD", False),
        }
        try:
            load_config_overrides("configs/mtam_hg.yaml")
            self.assertEqual(config.TABDIFF_EXP_NAME, "capl_mtamhg")
            self.assertEqual(config.SYNTHETIC_DATA_PATH, "data/synthetic_CAPL_ma_tabdiff.xlsx")
            self.assertEqual(config.SYNTHETIC_CONFIDENCE_THRESHOLD, 0.5)
            self.assertFalse(hasattr(config, "FILTERED_SYNTHETIC_DATA_PATH"))
            self.assertFalse(hasattr(config, "USE_SYNTHETIC_RESAMPLING"))
            self.assertTrue(config.USE_DYNAMIC_SYNTHETIC_AGENT)
            self.assertEqual(config.DYNAMIC_SYNTHETIC_REFRESH_EPOCHS, 5)
            self.assertEqual(config.DYNAMIC_SYNTHETIC_WARMUP_EPOCHS, 5)
            self.assertTrue(config.DYNAMIC_SYNTHETIC_USE_SAMPLER)
            self.assertTrue(config.DYNAMIC_SYNTHETIC_USE_LOSS_WEIGHT)
            self.assertEqual(config.DYNAMIC_SYNTHETIC_TRAIN_REWARD_METRIC, "rmse_tail")
            self.assertTrue(config.USE_MR_LORA)
            self.assertEqual(config.MR_LORA_SCOPE, "graph_attention_routing")
            self.assertEqual(config.MR_LORA_RANK_GRAPH, 8)
            self.assertEqual(config.MR_LORA_RANK_ROUTING, 4)
            self.assertTrue(config.USE_CLUSTER_BALANCE_REWARD)
        finally:
            for key, value in original.items():
                setattr(config, key, value)


if __name__ == "__main__":
    unittest.main()
