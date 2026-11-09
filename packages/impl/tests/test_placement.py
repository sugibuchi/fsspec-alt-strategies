"""Tests of `common.placement`: where `_Placement` puts the paths of a pattern's copy.

The copies of both families test the placing of fsspec's cases. These test the
edges of a pattern that none of them reaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies_impl.common.jobs import _Roots
from fsspec_alt_strategies_impl.common.placement import _Placement

if TYPE_CHECKING:
    from collections.abc import Sequence


def _placement(path1: str, path2: str, roots: _Roots) -> _Placement:
    """A copy's placement whose glob finds the roots given."""

    async def info(path: str) -> dict[str, Any]:
        raise FileNotFoundError(path)

    async def expand(pattern: str | Sequence[str]) -> _Roots:
        return roots

    return _Placement(
        path1,
        path2,
        destination_info=info,
        same_storage=True,
        expand=expand,
        found=[],
    )


async def test_a_pattern_matching_only_errors_yields_them() -> None:
    roots = _Roots([], [])
    assert await _placement("x*", "dest", roots).prepare() is roots


async def test_a_single_match_already_in_place_is_left() -> None:
    placement = _placement("a/*.txt", "a", _Roots([], ["a/x.txt"]))
    assert await placement.prepare() is None


async def test_matches_without_a_common_folder_go_under_the_destination() -> None:
    placement = _placement("*1", "dest", _Roots(["x1"], ["y1"]))
    assert await placement.prepare() is not None
    assert placement.prefix == ""
    assert (placement.target("x1/f"), placement.target("y1")) == (
        "dest/x1/f",
        "dest/y1",
    )
