from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tabdiff_rmse_does_not_use_removed_squared_keyword() -> None:
    source_path = PROJECT_ROOT / "third_party" / "TabDiff" / "eval" / "mle" / "mle.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "mean_squared_error":
            continue

        squared_keywords = [keyword for keyword in node.keywords if keyword.arg == "squared"]
        assert not squared_keywords
