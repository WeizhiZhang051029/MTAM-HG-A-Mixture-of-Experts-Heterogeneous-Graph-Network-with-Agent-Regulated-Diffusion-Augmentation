"""Fail closed when a Git release contains private data or credentials."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".feather",
    ".npy", ".npz", ".pkl", ".pickle", ".joblib", ".pt", ".pth",
    ".ckpt", ".h5", ".hdf5", ".onnx",
}
FORBIDDEN_PREFIXES = (
    "outputs/", "output/", "logs/", "checkpoints/", ".venv/",
    ".venv_tabdiff/", ".codex_sync_backups/",
    "third_party/TabDiff/data/", "third_party/TabDiff/synthetic/",
    "third_party/TabDiff/tabdiff/ckpt/", "third_party/TabDiff/tabdiff/result/",
)
FORBIDDEN_RUNTIME_SUFFIXES = (".provenance.json", ".postprocess.json")
ALLOWED_DATA_FILES: set[str] = set()

PRIVATE_TOKEN_HASHES = {
    "61957f71a9897c4104200f3511935848c4edeeaefd13893d71fabe5e5bada8f1",
    "994c643d0350818318a4863d0f48b6b5af97ba3d5fa358437e5fe18b5e8a218c",
    "3cc3bf0c950d21d454751ff7fcee115011e788e312336ec9fd1331543828d495",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_.:@-]{4,}")
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def tracked_files() -> list[str]:
    if not (ROOT / ".git").exists():
        return [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file() and not any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"} for part in p.parts)]
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def audit() -> list[str]:
    failures: list[str] = []
    for relative in tracked_files():
        normalized = relative.replace("\\", "/")
        path = ROOT / relative
        if path.is_symlink():
            failures.append(f"symbolic link is not allowed in the release: {normalized}")
            continue
        if not path.exists():
            continue
        if normalized.startswith("data/") and normalized not in ALLOWED_DATA_FILES:
            failures.append(f"private data path is tracked: {normalized}")
        if normalized.startswith(FORBIDDEN_PREFIXES):
            failures.append(f"runtime/private path is tracked: {normalized}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact type is tracked: {normalized}")
        if normalized.endswith(FORBIDDEN_RUNTIME_SUFFIXES):
            failures.append(f"runtime metadata is tracked: {normalized}")
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file exceeds 10 MiB release limit: {normalized}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label} pattern found in {normalized}")
        for token in TOKEN_RE.findall(text):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            if digest in PRIVATE_TOKEN_HASHES:
                failures.append(f"private host/login indicator found in {normalized}")
                break
    return failures


def main() -> int:
    failures = audit()
    if failures:
        print("release audit failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("release audit passed: tracked files contain no private data or credential indicators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
