"""Tests of `tree.cp`'s `_Plan` and `tree.rm`'s roots: the placing of a copy's roots.

These test what no strategy shows: each permit given back, however a queue
ends, and the edges of a pattern that no copy reaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies.monitor import State
from fsspec_alt_strategies_impl.common.operation import Operation
from fsspec_alt_strategies_impl.tree.cp import _Plan, _Roots, _roots_of
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def test_roots_of_leave_out_what_a_matching_dir_has() -> None:
    matches = [
        {"name": "a/b", "type": "file"},
        {"name": "a", "type": "directory"},
        {"name": "c/", "type": "directory"},
        {"name": "d", "type": "file"},
    ]
    roots = _roots_of(matches, [], "*")
    assert (
        [d["name"] for d in roots.directories],
        [f["name"] for f in roots.files],
    ) == (
        ["a", "c"],
        ["d"],
    )


def test_roots_of_nothing_raise_unless_errors_were_found() -> None:
    with pytest.raises(FileNotFoundError):
        _roots_of([], [], "x*")
    error = OSError("x")
    assert _roots_of([], [error], "x*") == _Roots([], [])


# The placing of a copy.


async def _missing(path: str) -> dict[str, Any]:
    raise FileNotFoundError(path)


async def _plan(path1: Any, path2: str, found: list[Any]) -> _Plan:
    """A copy's plan, placed, whose glob yields the entries and errors given."""
    storage = TreeView(
        SimulatedFileSystem(hierarchical=True, asynchronous=True), info=_missing
    )

    def glob(pattern: Any) -> Operation[Any, Any]:
        async def entries() -> AsyncGenerator[Any]:
            for item in found:
                yield item

        return Operation(lambda op: entries(), State())

    plan = _Plan(storage, storage, move=False, same_storage=True, maxdepth=None)
    await plan.read_profile(batching=False)
    await plan.place(path1, path2, None, glob)
    return plan


def _file(name: str) -> dict[str, Any]:
    return {"name": name, "type": "file"}


def _dir(name: str) -> dict[str, Any]:
    return {"name": name, "type": "directory"}


async def test_a_pattern_matching_only_errors_gives_them() -> None:
    error = OSError("x")
    plan = await _plan("x*", "dest", [error])
    assert (plan.directories, plan.files, plan.errors) == ([], [], [error])


async def test_errors_are_given_when_nothing_else_is_to_do() -> None:
    # The first root is in place already, and the second's name is taken.
    plan = await _plan(["a/x", "b/*"], "a", [_file("a/x"), _file("b/x")])
    assert (plan.directories, plan.files) == ([], [])
    assert [type(e) for e in plan.errors] == [FileExistsError]


async def test_a_single_match_already_in_place_is_left() -> None:
    plan = await _plan("a/*.txt", "a", [_file("a/x.txt")])
    assert (plan.directories, plan.files) == ([], [])


async def test_matches_without_a_common_dir_go_under_the_destination() -> None:
    plan = await _plan("*1", "dest", [_dir("x1"), _file("y1")])
    assert plan.prefix == ""
    assert (plan.target("x1/f"), plan.target("y1")) == ("dest/x1/f", "dest/y1")
