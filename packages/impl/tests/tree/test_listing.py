"""Tests of `_tree_ils`, a recursive listing made of one listing per directory."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import Any

import pytest

from fsspec_alt_strategies_impl.tree.listing import _tree_ils
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView


def _tree(fs: SimulatedFileSystem) -> None:
    for path in ("data/a/f", "data/a-x/g", "data/a.txt", "data/b/c/h", "data/b/i"):
        fs.add_file(path)


async def _names(view: TreeView, path: str) -> list[str]:
    return [e["name"] async for e in _tree_ils(view, path)]


async def test_lists_depth_first_in_name_order(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    # Each directory's entries in name order, all below one right after it.
    assert await _names(TreeView(any_fs), "data") == [
        "data/a",
        "data/a/f",
        "data/a-x",
        "data/a-x/g",
        "data/a.txt",
        "data/b",
        "data/b/c",
        "data/b/c/h",
        "data/b/i",
    ]
    # One listing per directory: data, a, a-x, b and b/c.
    assert any_fs.stats.calls == {"list": 5}


async def test_lists_what_a_recursive_listing_lists(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    any_fs.add_file("data/b/c/d/j")
    native = [e async for e in any_fs._ils_recursive("data")]
    listed = [e async for e in _tree_ils(TreeView(any_fs), "data")]
    # A flat storage lists no directory recursively: the walk tells them apart.
    files = [e for e in listed if e["type"] == "file"]
    assert sorted(files, key=lambda e: e["name"]) == [
        e for e in native if e["type"] == "file"
    ]


async def test_a_file_lists_itself(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    assert await _names(TreeView(any_fs), "data/a.txt") == ["data/a.txt"]


async def test_a_missing_root_raises(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(FileNotFoundError) as raised:
        await _names(TreeView(any_fs), "missing")
    assert not getattr(raised.value, "__notes__", [])


async def test_a_failed_directory_ends_the_listing() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)

    async def ls(path: str, detail: bool = True, **kwargs: Any) -> Any:
        if path == "data/a-x":
            raise PermissionError(path)
        return await fs._ls(path, detail=detail, **kwargs)

    view = TreeView(fs, ils=_dir_ils(ls))
    listed: list[str] = []
    with pytest.raises(PermissionError) as raised:
        # A loop, not a comprehension: what came before the error is kept.
        async for entry in _tree_ils(view, "data"):
            listed.append(entry["name"])  # noqa: PERF401
    assert raised.value.__notes__ == ["Listing 'data/a-x' failed."]
    assert listed == ["data/a", "data/a/f", "data/a-x"]
    assert fs._in_flight == 0


def _dir_ils(ls: Any) -> Any:
    async def ils(path: str, **kwargs: Any) -> Any:
        for entry in sorted(await ls(path, **kwargs), key=lambda e: e["name"]):
            yield entry

    return ils


async def test_a_directory_listed_twice_is_listed_once() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)

    async def ls(path: str, detail: bool = True, **kwargs: Any) -> Any:
        # As a prefix, and as its marker object.
        entries = await fs._ls(path, detail=detail, **kwargs)
        return entries + [
            {**e, "name": e["name"] + "/"} for e in entries if e["type"] == "directory"
        ]

    names = await _names(TreeView(fs, ils=_dir_ils(ls)), "data")
    assert names.count("data/b/c/h") == 1
    assert names.count("data/b/") == 1
    assert fs.stats.calls == {"list": 5}


async def test_lists_ahead_within_the_limit() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(20):
        fs.add_file(f"root/{i:02}/file")
    limit = asyncio.Semaphore(4)

    # The caller holds a permit, as the strategies hold one for each _ils.
    async with limit:
        names = await _names(TreeView(fs, operation_limit=limit), "root")

    assert len(names) == 40
    assert fs.stats.calls == {"list": 21}
    # Its 3 permits ahead, and the caller's.
    assert fs.stats.max_concurrency == 4


async def test_goes_on_under_the_callers_permit_alone() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    limit = asyncio.Semaphore(1)

    async with limit:  # The only permit: none is left to list ahead.
        names = await asyncio.wait_for(
            _names(TreeView(fs, operation_limit=limit), "data"), timeout=5
        )

    assert len(names) == 9
    assert fs.stats.max_concurrency == 1


async def test_closing_stops_the_listings_ahead() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    limit = asyncio.Semaphore(4)

    async with aclosing(_tree_ils(TreeView(fs, operation_limit=limit), "root")) as it:
        await anext(it)  # The root's first entry: listings ahead are in flight.

    assert fs._in_flight == 0
    assert fs.stats.calls["list"] < 11
    for _ in range(4):  # Every permit is back.
        await asyncio.wait_for(limit.acquire(), 0.1)
