"""Build and validate TabDiff commands and artifacts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import config
from protocol_integrity import TABDIFF_DETERMINISTIC_SEED, file_sha256


@dataclass(frozen=True)
class TabDiffCommand:
    command: list[str]
    cwd: Path

    def as_display(self) -> str:
        return f"(cd {self.cwd} && {' '.join(self.command)})"


def project_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (config.PROJECT_ROOT / raw)


def tabdiff_repo_path(path: str | Path | None = None) -> Path:
    return project_path(path or getattr(config, "TABDIFF_REPO_PATH", "third_party/TabDiff"))


def require_tabdiff_repo(path: str | Path | None = None) -> Path:
    repo = tabdiff_repo_path(path)
    required = [repo / "README.md", repo / "main.py", repo / "process_dataset.py", repo / "tabdiff"]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        clone_hint = f"git clone https://github.com/MinkaiXu/TabDiff.git {repo}"
        raise FileNotFoundError(
            "Official TabDiff repository is not available or is incomplete.\n"
            f"Missing: {missing}\n"
            f"Install it with: {clone_hint}"
        )
    return repo


def tabdiff_remote(repo: Path) -> str:
    git_dir = repo / ".git"
    if not git_dir.exists():
        return "not_a_git_checkout"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "remote", "get-url", "origin"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip()


def dataname() -> str:
    return str(getattr(config, "TABDIFF_DATANAME", "capl"))


def exp_name() -> str:
    return str(getattr(config, "TABDIFF_EXP_NAME", "capl_tabdiff"))


def base_exp_name() -> str:
    return f"{exp_name()}_base"


def generation_seed() -> int:
    """Return the seed implemented by official TabDiff deterministic mode."""
    seed = int(getattr(config, "TABDIFF_GENERATION_SEED", TABDIFF_DETERMINISTIC_SEED))
    if seed != TABDIFF_DETERMINISTIC_SEED:
        raise ValueError(
            "Official TabDiff --deterministic mode fixes its backend seed to 0; "
            f"TABDIFF_GENERATION_SEED must be 0, got {seed}."
        )
    return seed


def process_command(repo: Path, name: str | None = None) -> TabDiffCommand:
    return TabDiffCommand(
        command=[sys.executable, "process_dataset.py", "--dataname", name or dataname()],
        cwd=repo,
    )


def _training_command(
    repo: Path,
    name: str | None,
    *,
    experiment: str,
    checkpoint_path: str | Path | None,
    mechanism_finetune: bool,
) -> TabDiffCommand:
    generation_seed()
    cmd = [
        sys.executable,
        "main.py",
        "--dataname",
        name or dataname(),
        "--mode",
        "train",
        "--no_wandb",
        "--exp_name",
        experiment,
        "--gpu",
        str(getattr(config, "TABDIFF_GPU", -1)),
        "--deterministic",
    ]
    if checkpoint_path:
        cmd.extend(["--ckpt_path", str(project_path(checkpoint_path))])
    if mechanism_finetune:
        _append_mechanism_args(cmd, for_sampling=False)
    return TabDiffCommand(command=cmd, cwd=repo)


def base_train_command(repo: Path, name: str | None = None) -> TabDiffCommand:
    return _training_command(
        repo,
        name,
        experiment=base_exp_name(),
        checkpoint_path=None,
        mechanism_finetune=False,
    )


def train_command(
    repo: Path,
    name: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> TabDiffCommand:
    effective_checkpoint = checkpoint_path or getattr(config, "TABDIFF_CKPT_PATH", "") or None
    return _training_command(
        repo,
        name,
        experiment=exp_name(),
        checkpoint_path=effective_checkpoint,
        mechanism_finetune=True,
    )


def sample_command(
    repo: Path,
    name: str | None = None,
    num_samples: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> TabDiffCommand:
    generation_seed()
    cmd = [
        sys.executable,
        "main.py",
        "--dataname",
        name or dataname(),
        "--mode",
        "test",
        "--num_samples_to_generate",
        str(num_samples or int(getattr(config, "TABDIFF_NUM_SAMPLES", 1000))),
        "--no_wandb",
        "--exp_name",
        exp_name(),
        "--gpu",
        str(getattr(config, "TABDIFF_GPU", -1)),
        "--deterministic",
    ]
    ckpt_path = str(checkpoint_path or getattr(config, "TABDIFF_CKPT_PATH", "") or "")
    if ckpt_path:
        cmd.extend(["--ckpt_path", str(project_path(ckpt_path))])
    _append_mechanism_args(cmd, for_sampling=True)
    return TabDiffCommand(command=cmd, cwd=repo)


def _append_mechanism_args(cmd: list[str], *, for_sampling: bool) -> None:
    if not bool(getattr(config, "TABDIFF_MECHANISM_CONSTRAINT", False)):
        return
    cmd.append("--mechanism_constraint")
    cmd.extend(["--mechanism_lambda", str(getattr(config, "TABDIFF_MECHANISM_LAMBDA", 0.0 if for_sampling else 0.05))])
    cmd.extend(["--guidance_scale", str(getattr(config, "TABDIFF_GUIDANCE_SCALE", 0.0))])
    cmd.extend([
        "--mechanism_temperature_hold_tolerance",
        str(getattr(config, "TABDIFF_MECHANISM_TEMPERATURE_HOLD_TOLERANCE", 10.0)),
    ])
    cmd.extend(["--mechanism_window_margin", "0.0"])
    cmd.extend([
        "--mechanism_yield_tolerance",
        str(getattr(config, "TABDIFF_MECHANISM_YIELD_TOLERANCE", 0.0)),
    ])
    cmd.extend(["--trainable_scope", str(getattr(config, "TABDIFF_TRAINABLE_SCOPE", "all" if for_sampling else "mlp_detokenizer"))])
    cmd.extend(["--min_save_epoch", str(getattr(config, "TABDIFF_MIN_SAVE_EPOCH", 1))])
    finetune_lr = getattr(config, "TABDIFF_FINETUNE_LR", None)
    if finetune_lr is not None:
        cmd.extend(["--finetune_lr", str(finetune_lr)])
    finetune_steps = getattr(config, "TABDIFF_FINETUNE_STEPS", None)
    if finetune_steps is not None:
        cmd.extend(["--finetune_steps", str(finetune_steps)])
    num_timesteps = getattr(config, "TABDIFF_NUM_TIMESTEPS_OVERRIDE", None)
    if num_timesteps is not None:
        cmd.extend(["--num_timesteps_override", str(num_timesteps)])
    stochastic_sampler = getattr(config, "TABDIFF_STOCHASTIC_SAMPLER", None)
    if stochastic_sampler is not None:
        cmd.extend(["--stochastic_sampler", "true" if bool(stochastic_sampler) else "false"])
    if not for_sampling:
        cmd.append("--reset_train_epoch")


def check_tabdiff_dependencies(repo: Path) -> tuple[bool, str]:
    code = "import torch, numpy, pandas, sklearn, category_encoders, wandb; import tabdiff, src"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return True, "ok"
    message = completed.stderr.strip() or completed.stdout.strip()
    return False, (
        "TabDiff Python dependencies are not importable in the current environment. "
        "Use the official environment first: conda env create -f third_party/TabDiff/tabdiff.yaml. "
        f"Import error: {message}"
    )


def run_command(command: TabDiffCommand) -> None:
    subprocess.run(command.command, cwd=command.cwd, check=True)


def processed_dataset_ready(repo: Path, name: str | None = None) -> bool:
    data_dir = repo / "data" / (name or dataname())
    synthetic_dir = repo / "synthetic" / (name or dataname())
    required = [
        "X_num_train.npy",
        "X_cat_train.npy",
        "y_train.npy",
        "X_num_test.npy",
        "X_cat_test.npy",
        "y_test.npy",
        "info.json",
    ]
    synthetic_required = ["real.csv", "test.csv"]
    return all((data_dir / item).exists() for item in required) and all(
        (synthetic_dir / item).exists() for item in synthetic_required
    )


def expected_sample_patterns(repo: Path, name: str | None = None) -> list[Path]:
    ds = name or dataname()
    exp = exp_name()
    return [
        repo / "tabdiff" / "result" / ds / exp,
        repo / "eval" / "report_runs" / exp / ds / "all_samples",
    ]


def checkpoint_output_snapshot(
    repo: Path,
    name: str | None = None,
    experiment: str | None = None,
) -> dict[str, tuple[int, int, str]]:
    """Capture existing official TabDiff checkpoint outputs."""
    checkpoint_dir = repo / "tabdiff" / "ckpt" / (name or dataname()) / (experiment or exp_name())
    snapshot: dict[str, tuple[int, int, str]] = {}
    if not checkpoint_dir.exists():
        return snapshot
    for path in checkpoint_dir.glob("best_ema_model*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        stat = resolved.stat()
        snapshot[str(resolved)] = (int(stat.st_mtime_ns), int(stat.st_size), file_sha256(resolved))
    return snapshot


def find_fresh_checkpoint(
    repo: Path,
    name: str | None,
    previous_snapshot: dict[str, tuple[int, int, str]],
    experiment: str | None = None,
) -> Path:
    """Return a checkpoint created or updated by the current train command."""
    current = checkpoint_output_snapshot(repo, name, experiment)
    changed = [Path(path) for path, signature in current.items() if previous_snapshot.get(path) != signature]
    if len(changed) != 1:
        raise RuntimeError(
            "The TabDiff training command must produce exactly one new or updated "
            f"best_ema_model checkpoint; found {len(changed)}. Refusing to select "
            "an ambiguous or stale checkpoint."
        )
    return changed[0]


def _sample_candidates(repo: Path, name: str | None = None) -> list[Path]:
    candidates: set[Path] = set()
    for root in expected_sample_patterns(repo, name):
        if root.exists():
            candidates.update(path.resolve() for path in root.rglob("samples.csv"))
            candidates.update(path.resolve() for path in root.rglob("samples_*.csv"))
    return sorted(candidates)


def sample_output_snapshot(repo: Path, name: str | None = None) -> dict[str, tuple[int, int, str]]:
    """Capture timestamp, size and content hash for existing sample outputs."""
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in _sample_candidates(repo, name):
        stat = path.stat()
        snapshot[str(path)] = (int(stat.st_mtime_ns), int(stat.st_size), file_sha256(path))
    return snapshot


def find_latest_sample(
    repo: Path,
    name: str | None = None,
    previous_snapshot: dict[str, tuple[int, int, str]] | None = None,
) -> Path | None:
    candidates = _sample_candidates(repo, name)
    if previous_snapshot is not None:
        fresh = []
        for path in candidates:
            stat = path.stat()
            signature = (int(stat.st_mtime_ns), int(stat.st_size), file_sha256(path))
            if previous_snapshot.get(str(path)) != signature:
                fresh.append(path)
        if len(fresh) != 1:
            raise RuntimeError(
                "The TabDiff sampling command must produce exactly one new or updated "
                f"sample CSV; found {len(fresh)}; refusing to reuse a stale output or "
                "select an ambiguous output."
            )
        return fresh[0]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def copy_latest_sample(
    repo: Path,
    output_path: str | Path,
    name: str | None = None,
    previous_snapshot: dict[str, tuple[int, int, str]] | None = None,
) -> Path:
    sample = find_latest_sample(repo, name, previous_snapshot=previous_snapshot)
    if sample is None:
        roots = [str(path) for path in expected_sample_patterns(repo, name)]
        raise FileNotFoundError(f"TabDiff sample CSV was not found under: {roots}")
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample, output)
    return output


def command_summary(commands: Sequence[TabDiffCommand]) -> list[str]:
    return [command.as_display() for command in commands]
