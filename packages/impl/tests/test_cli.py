"""Tests of the command line: copying the installed strategies into a package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

import fsspec_alt_strategies_impl
from fsspec_alt_strategies_impl_cli import main

if TYPE_CHECKING:
    from pathlib import Path


def test_copy_carries_the_version(tmp_path: Path) -> None:
    target = tmp_path / "_contrib" / "strategy"
    result = CliRunner().invoke(main, ["copy", str(target)])
    assert result.exit_code == 0, result.output
    assert fsspec_alt_strategies_impl.__version__ in result.output
    assert (target / "_version.py").is_file()
    assert (target / "tree" / "rm.py").is_file()
    assert not list(target.rglob("__pycache__"))


def test_copy_carries_the_license(tmp_path: Path) -> None:
    target = tmp_path / "strategy"
    result = CliRunner().invoke(main, ["copy", str(target)])
    assert result.exit_code == 0, result.output
    assert "Apache License" in (target / "LICENSE").read_text(encoding="utf-8")


def test_copy_replaces_a_previous_copy(tmp_path: Path) -> None:
    target = tmp_path / "strategy"
    CliRunner().invoke(main, ["copy", str(target)])
    (target / "removed.py").write_text("")
    result = CliRunner().invoke(main, ["copy", str(target)])
    assert result.exit_code == 0, result.output
    assert not (target / "removed.py").exists()


def test_copy_refuses_a_dir_that_is_no_copy(tmp_path: Path) -> None:
    (tmp_path / "mine.py").write_text("")
    result = CliRunner().invoke(main, ["copy", str(tmp_path)])
    assert result.exit_code != 0
    assert "not replaced" in result.output
    assert (tmp_path / "mine.py").exists()
