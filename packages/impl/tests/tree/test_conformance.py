"""Tests of `check_recursive_order`: a handler's `ils_recursive` against the order it must keep."""

from __future__ import annotations

from typing import Any

import pytest

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView
from fsspec_alt_strategies_impl_testing.conformance import (
    check_recursive_order,
    recursive_order_errors,
)


class _BreadthFirst(SimulatedFileSystem):
    """Lists a tree level by level: each directory before what is below it, and nothing more."""

    async def _ils_recursive(self, path: str) -> Any:
        entries = [e async for e in super()._ils_recursive(path)]
        entries.sort(key=lambda e: (e["name"].count("/"), e["name"]))
        for entry in entries:
            yield entry


def _view(fs: SimulatedFileSystem) -> TreeView:
    profile = StorageProfile(recursive_ls=True)
    return TreeView(fs, recursive_listing=True, storage_profile=profile)


async def test_a_listing_by_name_keeps_the_order(any_fs: SimulatedFileSystem) -> None:
    if any_fs.hierarchical:
        any_fs.add_dir("scratch")  # A flat one has no empty directory: none is needed.
    await check_recursive_order(_view(any_fs), "scratch")
    # The tree it wrote is gone.
    assert await any_fs._find("scratch") == []


async def test_a_breadth_first_listing_fails() -> None:
    fs = _BreadthFirst(hierarchical=True, asynchronous=True)
    fs.add_dir("scratch")
    with pytest.raises(AssertionError, match="after the listing passed"):
        await check_recursive_order(_view(fs), "scratch")
    assert await fs._find("scratch") == []


def test_the_errors_name_the_entry_out_of_order() -> None:
    def entry(name: str, kind: str = "file") -> dict[str, Any]:
        return {"name": name, "type": kind}

    # Coming back below a directory passed.
    listing = [entry("r/a", "directory"), entry("r/a/x"), entry("r/b", "directory")]
    listing += [entry("r/b/z"), entry("r/a/y")]
    assert recursive_order_errors(listing, "r") == [
        "'r/a/y' comes after the listing passed 'r/a', below which it is."
    ]
    # A file before its directory, a directory left out, and a marker listed.
    listing = [entry("r/a/y"), entry("r/a", "directory"), entry("r/b/", "directory")]
    assert recursive_order_errors(listing, "r") == [
        "'r/a/y' comes before its directory 'r/a'.",
        "'r/b/' is a marker, not its directory's entry.",
    ]
