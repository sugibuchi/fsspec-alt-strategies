"""Tests of `check_recursive_order`: a handler's recursive listing against the order its profile declares."""

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


def _view(fs: SimulatedFileSystem, preorder: bool) -> TreeView:
    profile = StorageProfile(recursive_ls_depth_first_preorder=preorder)
    return TreeView(fs, recursive_listing=True, storage_profile=profile)


@pytest.mark.parametrize("preorder", [True, False])
async def test_a_listing_by_name_keeps_any_order(
    any_fs: SimulatedFileSystem, preorder: bool
) -> None:
    if any_fs.hierarchical:
        any_fs.add_dir("scratch")  # A flat one has no empty directory: none is needed.
    await check_recursive_order(_view(any_fs, preorder), "scratch")
    # The tree it wrote is gone.
    assert await any_fs._find("scratch") == []


async def test_a_breadth_first_listing_keeps_parents_first() -> None:
    fs = _BreadthFirst(hierarchical=True, asynchronous=True)
    fs.add_dir("scratch")
    await check_recursive_order(_view(fs, False), "scratch")


async def test_a_breadth_first_listing_declared_depth_first_fails() -> None:
    fs = _BreadthFirst(hierarchical=True, asynchronous=True)
    fs.add_dir("scratch")
    with pytest.raises(AssertionError, match="after the listing passed"):
        await check_recursive_order(_view(fs, True), "scratch")
    assert await fs._find("scratch") == []


def test_the_errors_name_the_entry_out_of_order() -> None:
    def entry(name: str, kind: str = "file") -> dict[str, Any]:
        return {"name": name, "type": kind}

    # Depth first, coming back below a directory passed. Its own entry apart, as in
    # name order, is no break: only what is below it, before and after another.
    listing = [entry("r/a/x"), entry("r/b/z"), entry("r/a/y")]
    assert recursive_order_errors(listing, "r", True) == [
        "'r/a/y' comes after the listing passed 'r/a', below which it is."
    ]
    # Parents first, a file before its directory's entry; a directory never listed breaks nothing.
    listing = [entry("r/a/y"), entry("r/a", "directory"), entry("r/b/z")]
    assert recursive_order_errors(listing, "r", False) == [
        "'r/a/y' comes before its directory 'r/a'."
    ]
