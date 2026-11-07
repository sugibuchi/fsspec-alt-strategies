"""Tests that the shared interface stays light: fsspec and the standard library only."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import fsspec_alt_strategies_common

PACKAGE_DIR = Path(fsspec_alt_strategies_common.__file__).parent
ALLOWED_IMPORTS = sys.stdlib_module_names | {"fsspec"}


def _sources() -> list[Path]:
    return sorted(p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("source", _sources(), ids=lambda p: p.name)
def test_imports_only_fsspec_and_stdlib(source: Path) -> None:
    # Every copy of the strategies imports it: it must import none of them.
    imported = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported |= {alias.name.partition(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.partition(".")[0])
    assert imported <= ALLOWED_IMPORTS, imported - ALLOWED_IMPORTS


@pytest.mark.parametrize("source", _sources(), ids=lambda p: p.name)
def test_license_header(source: Path) -> None:
    head = source.read_text(encoding="utf-8").splitlines()[:5]
    assert "# SPDX-License-Identifier: Apache-2.0" in head
