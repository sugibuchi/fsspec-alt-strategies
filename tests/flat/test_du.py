"""Tests of `flat_du`, checked against fsspec's own ``_du``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.control import collect
from fsspec_alt_strategies.flat.du import DuState, flat_du
from fsspec_alt_strategies_testing import SimulatedFileSystem


def _du(fs: SimulatedFileSystem, path: str, **kwargs: Any) -> Any:
    """`flat_du` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "ils": fs._ils,
        "info": fs._info,
    }
    return flat_du(path, **{**operations, **kwargs})


def _tree(fs: SimulatedFileSystem) -> None:
    for year, size in (("2023", 10), ("2024", 100)):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * size)
    fs.add_file("data/README.md", b"x")


@pytest.mark.parametrize("maxdepth", [None, 1, 3])
@pytest.mark.parametrize("path", ["data", "data/2024", "data/README.md", "missing"])
async def test_measures_as_fsspec(
    flat_fs: SimulatedFileSystem, path: str, maxdepth: int | None
) -> None:
    _tree(flat_fs)
    op = _du(flat_fs, path, maxdepth=maxdepth)
    entries, errors = await collect(op)
    assert errors == []
    total = await AsyncFileSystem._du(flat_fs, path, maxdepth=maxdepth)
    assert op.state.size == total
    each = await AsyncFileSystem._du(flat_fs, path, total=False, maxdepth=maxdepth)
    assert {e["name"]: e["size"] for e in entries} == each


async def test_sends_only_the_listing() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _du(fs, "data")
    await collect(op)
    assert op.state == DuState(current=op.state.current, files=5, size=221)
    assert fs.stats.calls == {"list": 3}
