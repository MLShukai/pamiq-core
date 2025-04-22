from __future__ import annotations

import tomllib
from pathlib import Path

import pamiq_core

PROJECT_ROOT = Path(__file__).parent.parent


def test_version() -> None:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    assert pamiq_core.__version__ == pyproject["project"]["version"]
