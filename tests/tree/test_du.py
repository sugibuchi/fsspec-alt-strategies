"""Tests of `tree_du`, checked against fsspec's own ``_du``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.control import collect
from fsspec_alt_strategies.tree.du import DuState, tree_du

if TYPE_CHECKING:
    from fsspec_alt_strategies_testing import SimulatedFileSystem


def _du(fs: SimulatedFileSystem, path: Any, **kwargs: Any) -> Any:
    """`tree_du` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "ls": fs._ls,
        "info": fs._info,
    }
    return tree_du(path, **{**operations, **kwargs})


def _tree(fs: SimulatedFileSystem) -> None:
    for year, size in (("2023", 10), ("2024", 100)):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * size)
    fs.add_file("data/README.md", b"x")


@pytest.mark.parametrize("maxdepth", [None, 1, 3])
@pytest.mark.parametrize("path", ["data", "data/2024", "data/README.md", "missing"])
async def test_measures_as_fsspec(
    any_fs: SimulatedFileSystem, path: str, maxdepth: int | None
) -> None:
    _tree(any_fs)
    op = _du(any_fs, path, maxdepth=maxdepth)
    entries, errors = await collect(op)
    assert errors == []
    total = await AsyncFileSystem._du(any_fs, path, maxdepth=maxdepth)
    assert op.state.size == total
    each = await AsyncFileSystem._du(any_fs, path, total=False, maxdepth=maxdepth)
    assert {e["name"]: e["size"] for e in entries} == each


async def test_sends_no_info_per_file(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    op = _du(any_fs, "data")
    await collect(op)
    assert op.state == DuState(
        current=op.state.current, folders_listed=7, files=5, size=221
    )
    assert any_fs.stats.calls == {"list": 7}


async def test_cancel_counts_what_was_listed(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    op = _du(any_fs, "data", concurrency_limit=asyncio.Semaphore(1))
    async for _ in op:
        op.cancel()
    # data, and data/2023 listed with it: only the README is counted.
    assert (op.state.size, op.state.files) == (1, 1)
    assert op.cancelled()
