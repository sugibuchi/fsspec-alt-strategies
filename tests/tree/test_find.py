"""Tests of `tree_find`, checked against fsspec's own ``_find``."""

from __future__ import annotations

from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.control import collect
from fsspec_alt_strategies.testing import SimulatedFileSystem
from fsspec_alt_strategies.tree.find import FindState, tree_find


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * 10)
    fs.add_file("data/README.md")


def _sorted(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda e: e["name"])


def _names(entries: list[dict[str, Any]]) -> list[str]:
    return sorted(e["name"].rstrip("/") for e in entries)


async def _fsspec_find(fs: SimulatedFileSystem, path: str, **kwargs: Any) -> Any:
    # fsspec's own walk, as the reference answer.
    return await AsyncFileSystem._find(fs, path, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"withdirs": True}, {"maxdepth": 1}, {"maxdepth": 2, "withdirs": True}],
)
@pytest.mark.parametrize("path", ["data", "data/2024", "data/README.md", "missing"])
async def test_finds_as_fsspec(
    any_fs: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    _tree(any_fs)
    entries, errors = await collect(tree_find(any_fs, path, **kwargs))
    assert errors == []
    expected = await _fsspec_find(any_fs, path, detail=True, **kwargs)
    assert _names(entries) == list(expected)
    assert _sorted(entries) == list(expected.values())


async def test_entries_are_the_listings(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    async with tree_find(any_fs, "data") as op:
        entries = [e async for e in op]
    assert all(e["size"] == 10 for e in entries if e["name"].endswith(".parquet"))
    # One listing per folder, and no _info.
    assert any_fs.stats.calls == {"list": 7}
    assert op.state == FindState(
        current=op.state.current, folders_listed=7, found=len(entries)
    )


async def test_the_whole_filesystem(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    entries, _ = await collect(tree_find(any_fs, "", withdirs=True))
    assert _names(entries) == await _fsspec_find(any_fs, "", withdirs=True)


async def test_lists_folders_concurrently() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(20):
        fs.add_file(f"root/{i:02}/file")
    entries, _ = await collect(tree_find(fs, "root", max_concurrency=5))
    assert len(entries) == 20
    assert fs.stats.max_concurrency == 5


def test_maxdepth_is_checked_at_once(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        tree_find(any_fs, "data", maxdepth=0)


class _Forbidding(SimulatedFileSystem):
    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


async def test_a_failed_folder_is_yielded() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    async with tree_find(fs, "data") as op:
        entries, errors = await collect(op)
    assert [type(e) for e in errors] == [PermissionError]
    assert _names(entries) == [
        "data/2024/01/part-0.parquet",
        "data/2024/02/part-0.parquet",
        "data/README.md",
    ]
    assert op.state.errors == 1


async def test_cancel_after_the_first_listing(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    op = tree_find(any_fs, "data", max_concurrency=1)
    found = []
    async for entry in op:
        found.append(entry["name"])
        op.cancel()
    # data's listing, then the one in flight with it, data/2023, which has no file.
    assert found == ["data/README.md"]
    assert any_fs.stats.calls["list"] <= 2
    assert op.cancelled()


async def test_closing_stops_the_search() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    async with tree_find(fs, "root", max_concurrency=2) as found:
        async for _ in found:
            break  # The first file: other listings are now in flight.
    assert fs._in_flight == 0
    assert fs.stats.calls["list"] < 11
