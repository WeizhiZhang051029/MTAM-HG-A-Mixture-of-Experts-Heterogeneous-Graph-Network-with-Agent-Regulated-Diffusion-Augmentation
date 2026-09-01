"""Configuration loading for the MTAM-HG main experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _fallback_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                continue
            item = content[2:].strip()
            if ":" in item:
                key, value = item.split(":", 1)
                obj: dict[str, Any] = {key.strip(): _parse_scalar(value)}
                parent.append(obj)
                stack.append((indent, obj))
            else:
                parent.append(_parse_scalar(item))
            continue
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            next_obj: dict[str, Any] = {}
            if isinstance(parent, dict):
                parent[key] = next_obj
            stack.append((indent, next_obj))
        else:
            if isinstance(parent, dict):
                parent[key] = _parse_scalar(value)
    return root


def load_yaml_config_text(text: str) -> dict[str, Any]:
    """Parse YAML text with the bundled fallback."""
    try:
        import yaml  # type: ignore
    except ImportError:
        return _fallback_yaml(text)
    loaded = yaml.safe_load(text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML configuration root must be a mapping.")
    return loaded


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    return load_yaml_config_text(Path(path).read_text(encoding="utf-8"))
