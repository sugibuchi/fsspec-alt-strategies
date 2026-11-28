"""Tests of `tree.cp`'s `_Plan`: where a copy's matches go, as the traversal finds them."""

from __future__ import annotations

from typing import Any

import pytest

from fsspec_alt_strategies_impl.tree.cp import _Plan
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView


async def _missing(path: str) -> dict[str, Any]:
    raise FileNotFoundError(path)


async def _plan(path1: Any, path2: str) -> _Plan:
    """A copy's plan within one storage, placed, whose destination is missing."""
    storage = TreeView(
        SimulatedFileSystem(hierarchical=True, asynchronous=True), info=_missing
    )
    plan = _Plan(storage, storage, move=False, same_storage=True, maxdepth=None)
    await plan.read_profile(batching=False)
    assert await plan.place(path1, path2, None)
    return plan


async def test_a_patterns_matches_go_below_its_literal_root() -> None:
    plan = await _plan("data/*/x", "dest")
    assert plan.prefix == "data"
    assert plan.top("data/2023/x") == "dest/2023/x"


async def test_matches_without_a_literal_root_go_under_the_destination() -> None:
    plan = await _plan("*1", "dest")
    assert (plan.top("x1"), plan.top("y1")) == ("dest/x1", "dest/y1")


async def test_a_lists_second_of_a_name_is_refused() -> None:
    # A literal path claims its target before any match is found.
    plan = await _plan(["b/*", "a/x"], "dest")
    with pytest.raises(FileExistsError):
        plan.top("b/x")
    assert plan.top("a/x") == "dest/x"


async def test_a_match_onto_itself_is_refused() -> None:
    # As fsspec's copy refuses it, unless it is the only match: one pass
    # cannot know that.
    plan = await _plan("a/*.txt", "a")
    with pytest.raises(ValueError, match="into itself"):
        plan.top("a/x.txt")


async def test_a_match_into_itself_is_refused() -> None:
    plan = await _plan("a*", "a/b")
    with pytest.raises(ValueError, match="into itself"):
        plan.top("a")
