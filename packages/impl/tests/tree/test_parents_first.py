"""Tests of the tree strategies on a recursive listing declared parents first, e.g. breadth first."""

from __future__ import annotations

from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.find import _tree_find
from fsspec_alt_strategies_impl.tree.mv import _tree_mv
from fsspec_alt_strategies_impl.tree.rm import _tree_rm
from fsspec_alt_strategies_impl.tree.walk import _tree_walk
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView

_FILES = [
    "data/a/1",
    "data/a/b/2",
    "data/a/b/c/3",
    "data/a/b-x/4",
    "data/d/5",
    "data/6",
]


class _BreadthFirst(SimulatedFileSystem):
    """Lists a tree in one request, level by level: each directory before what is below it, and nothing more.

    It breaks off after ``cut_after`` entries, if given.
    """

    cut_after: int | None = None

    async def _ils_recursive(self, path: str) -> Any:
        entries = [e async for e in super()._ils_recursive(path)]
        entries.sort(key=lambda e: (e["name"].count("/"), e["name"]))
        for count, entry in enumerate(entries):
            if count == self.cut_after:
                raise ConnectionResetError(path)
            yield entry


def _view(fs: SimulatedFileSystem) -> TreeView:
    # Declared parents first; left unknown, the order is taken so too.
    profile = StorageProfile(recursive_ls=False)
    return TreeView(fs, recursive_listing=True, storage_profile=profile)


def _fs(hierarchical: bool = True) -> _BreadthFirst:
    fs = _BreadthFirst(hierarchical=hierarchical, asynchronous=True)
    for path in _FILES:
        fs.add_file(path)
    return fs


@pytest.mark.parametrize("withdirs", [False, True])
@pytest.mark.parametrize("hierarchical", [True, False])
async def test_find_as_fsspec_a_directory_per_listing(
    withdirs: bool, hierarchical: bool
) -> None:
    fs = _fs(hierarchical)
    found, errors = await _collect_with_errors(
        _tree_find(_view(fs), "data", withdirs=withdirs)
    )
    assert errors == []
    # Its own listing is not streamed: one per directory, and with withdirs
    # the root's own info, read before fsspec's find adds its own.
    assert fs.stats.calls == {"list": 6, **({"info": 1} if withdirs else {})}
    expected = await AsyncFileSystem._find(fs, "data", withdirs=withdirs)
    assert sorted(e["name"] for e in found) == expected


async def test_a_parents_first_listing_declared_depth_first_goes_wrong() -> None:
    # Why the declaration: taken as depth first, a directory is done as soon as
    # the listing passes to another, though more of it comes later.
    fs = _fs()
    profile = StorageProfile(recursive_ls=True)
    view = TreeView(fs, recursive_listing=True, storage_profile=profile)
    walked, _ = await _collect_with_errors(_tree_walk(view, "data", topdown=False))
    # An rm plans data/a twice too. Its consumer orders a batch by names, so a
    # tree this small, read in one batch, is still deleted bottom up.
    assert [d for d, _, _ in walked].count("data/a") > 1


async def test_bottom_up_each_directory_after_those_below_it() -> None:
    fs = _fs()
    walked, errors = await _collect_with_errors(
        _tree_walk(_view(fs), "data", topdown=False)
    )
    assert errors == []
    order = [d for d, _, _ in walked]
    assert sorted(order) == sorted(
        ["data", "data/a", "data/a/b", "data/a/b/c", "data/a/b-x", "data/d"]
    )
    for directory in order:
        below = [o for o in order if o.startswith(directory + "/")]
        assert all(order.index(b) < order.index(directory) for b in below)


async def test_rm_deletes_each_directory_once_the_listing_ends() -> None:
    fs = _fs()
    deleted, errors = await _collect_with_errors(_tree_rm(_view(fs), "data"))
    assert errors == []
    assert deleted[-1] == "data"
    assert not await fs._exists("data")


async def test_mv_moves_the_whole_tree() -> None:
    fs = _fs()
    fs.add_file("out/kept")
    view = _view(fs)
    _, errors = await _collect_with_errors(_tree_mv(view, view, "data/", "out"))
    assert errors == []
    assert sorted(await fs._find("out")) == sorted(
        ["out/kept", *(p.replace("data", "out", 1) for p in _FILES)]
    )
    assert not await fs._exists("data/a")
