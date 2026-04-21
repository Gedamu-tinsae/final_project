"""Lightweight config/env loader for notebook and scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_dotenv_file(dotenv_path: Path) -> None:
    """Load KEY=VALUE pairs from .env into process env if not already set."""
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _substitute_project_root(value: Any, project_root: Path) -> Any:
    token = "${PROJECT_ROOT}"
    if isinstance(value, str):
        return value.replace(token, str(project_root))
    if isinstance(value, list):
        return [_substitute_project_root(v, project_root) for v in value]
    if isinstance(value, dict):
        return {k: _substitute_project_root(v, project_root) for k, v in value.items()}
    return value


def load_json_config(config_path: Path, project_root: Path) -> dict:
    """Load JSON config and expand ${PROJECT_ROOT} placeholders."""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return _substitute_project_root(data, project_root)
