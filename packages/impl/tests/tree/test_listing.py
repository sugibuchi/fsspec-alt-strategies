"""Tests of `common.listing`'s `implied_directories`, and of the testing package's `ils_by_directory`, a recursive listing made of one listing per directory."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import Any

import pytest

from fsspec_alt_strategies_impl.common.listing import implied_directories
from fsspec_alt_strategies_impl_testing import (
    SimulatedFileSystem,
    TreeView,
    ils_by_directory,
)


def _tree(fs: SimulatedFileSystem) -> None:
    for path in ("data/a/f", "data/a-x/g", "data/a.txt", "data/b/c/h", "data/b/i"):
        fs.add_file(path)


async def _names(view: TreeView, path: str) -> list[str]:
    return [e["name"] async for e in ils_by_directory(view, view.entry(path))]


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
    listed = [e async for e in ils_by_directory(TreeView(any_fs), {"name": "data"})]
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
        async for entry in ils_by_directory(view, view.entry("data")):
            listed.append(entry["name"])  # noqa: PERF401
    assert not getattr(raised.value, "__notes__", [])  # As the storage raised it.
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


async def test_goes_on_under_the_callers_permit_alone() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    limit = asyncio.Semaphore(1)

    async with limit:  # The only permit: it lists one directory at a time.
        names = await asyncio.wait_for(
            _names(TreeView(fs, operation_limit=limit), "data"), timeout=5
        )

    assert len(names) == 9
    assert fs.stats.max_concurrency == 1


async def test_closing_closes_each_listing_open() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    async with aclosing(ils_by_directory(TreeView(fs), {"name": "data"})) as it:
        await anext(it)
        await anext(it)  # Inside data/a: two listings open.
    assert fs._in_flight == 0


async def _implied(names: list[str], top: str) -> list[tuple[str, str]]:
    async def keys() -> Any:
        for name in names:
            yield {"name": name, "type": "directory" if name.endswith("/") else "file"}

    return [(e["name"], e["type"]) async for e in implied_directories(keys(), top)]


async def test_implied_directories_make_a_tree_in_preorder() -> None:
    # Name order: "data/a-x" comes between "data/a" and "data/a/f"'s directory.
    keys = ["data/a-x/g", "data/a.txt", "data/a/f", "data/b/c/h", "data/b/i"]
    assert await _implied(keys, "data") == [
        ("data/a-x", "directory"),
        ("data/a-x/g", "file"),
        ("data/a.txt", "file"),
        ("data/a", "directory"),
        ("data/a/f", "file"),
        ("data/b", "directory"),
        ("data/b/c", "directory"),
        ("data/b/c/h", "file"),
        ("data/b/i", "file"),
    ]


async def test_implied_directories_take_markers_as_directories() -> None:
    keys = ["data/", "data/a/", "data/a/f", "data/a/f", "data/e/"]
    # The top's marker left out, a key listed twice once, an empty directory kept.
    assert await _implied(keys, "data") == [
        ("data/a", "directory"),
        ("data/a/f", "file"),
        ("data/e", "directory"),
    ]


async def test_implied_directories_below_a_prefix() -> None:
    # "data/2024-" lists the keys starting with it: their directory is "data".
    keys = ["data/2024-01/a", "data/2024-02.csv"]
    assert await _implied(keys, "data") == [
        ("data/2024-01", "directory"),
        ("data/2024-01/a", "file"),
        ("data/2024-02.csv", "file"),
    ]
    assert await _implied(["a/b"], "") == [("a", "directory"), ("a/b", "file")]
    assert await _implied(["data/f"], "data/f") == [("data/f", "file")]
