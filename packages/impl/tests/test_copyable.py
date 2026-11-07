"""Tests that the package works when copied into another package.

Users are advised to copy the modules into their own package, e.g. as
``_contrib/strategy``, rather than depend on releases. These tests keep that
possible: they check the rules of DEVELOPMENT.md's "Copied, not only installed".
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import fsspec_alt_strategies

if TYPE_CHECKING:
    from collections.abc import Iterator

PACKAGE = fsspec_alt_strategies.__name__
PACKAGE_DIR = Path(fsspec_alt_strategies.__file__).parent
# The shared interface is installed, not copied: a copy imports it by its name.
COMMON = "fsspec_alt_strategies_common"
ALLOWED_IMPORTS = sys.stdlib_module_names | {"fsspec", COMMON}


def _sources() -> list[Path]:
    return sorted(p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _absolute_imports(source: Path) -> Iterator[str]:
    """The top-level names of the modules ``source`` imports, relative ones left out."""
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.partition(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.partition(".")[0]


@pytest.mark.parametrize("source", _sources(), ids=lambda p: p.name)
def test_imports_only_fsspec_the_interface_and_stdlib(source: Path) -> None:
    # The package's own modules must be imported relatively, so that a copy
    # works under any name.
    imported = set(_absolute_imports(source))
    assert PACKAGE not in imported
    assert imported <= ALLOWED_IMPORTS, imported - ALLOWED_IMPORTS


@pytest.mark.parametrize("source", _sources(), ids=lambda p: p.name)
def test_license_header(source: Path) -> None:
    # Copies must keep the license notice, which the header carries with the file.
    head = source.read_text(encoding="utf-8").splitlines()[:5]
    assert "# SPDX-License-Identifier: Apache-2.0" in head


def test_copy_imports_under_another_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The static checks above miss imports built at run time, so import a copy
    # while the original package cannot be imported.
    target = tmp_path / "_contrib" / "strategy"
    shutil.copytree(PACKAGE_DIR, target, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "_contrib" / "__init__.py").touch()

    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [n for n in sys.modules if n == PACKAGE or n.startswith(f"{PACKAGE}.")]:
        monkeypatch.setitem(sys.modules, name, None)

    try:
        copy = importlib.import_module("_contrib.strategy")
        modules = [
            info.name
            for info in pkgutil.walk_packages(copy.__path__, "_contrib.strategy.")
        ]
        for name in modules:
            importlib.import_module(name)
        assert copy.__version__ == fsspec_alt_strategies.__version__
    finally:
        for name in [n for n in sys.modules if n.partition(".")[0] == "_contrib"]:
            del sys.modules[name]
