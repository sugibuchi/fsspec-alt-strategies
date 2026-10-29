"""Tests of `tree_du`, checked against fsspec's own ``_du``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.control import Stop
from fsspec_alt_strategies.du import DuState, tree_du

if TYPE_CHECKING:
    from fsspec_alt_strategies.testing import SimulatedFileSystem


def _tree(fs: SimulatedFileSystem) -> None:
    for year, size in (("2023", 10), ("2024", 100)):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * size)
    fs.add_file("data/README.md", b"x")


@pytest.mark.parametrize("maxdepth", [None, 1, 3])
@pytest.mark.parametrize("path", ["data", "data/2024", "data/README.md", "missing"])
async def test_answers_as_fsspec(
    any_fs: SimulatedFileSystem, path: str, maxdepth: int | None
) -> None:
    _tree(any_fs)
    result = await tree_du(any_fs, path, maxdepth=maxdepth)
    expected = await AsyncFileSystem._du(any_fs, path, maxdepth=maxdepth)
    assert (result.complete, result.size) == (True, expected)


async def test_sends_no_info_per_file(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    result = await tree_du(any_fs, "data")
    assert (result.size, result.files, result.folders) == (221, 5, 7)
    assert any_fs.stats.calls == {"list": 7}


async def test_stop_returns_the_size_so_far(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    states: list[DuState] = []

    def first_only(state: DuState) -> Stop:
        states.append(state)
        return Stop()

    result = await tree_du(any_fs, "data", callback=first_only)
    assert (result.complete, result.size, result.files) == (False, 1, 1)
    assert (states[0].size, states[0].folders_listed) == (1, 1)
