"""Tests of `common.placement`: where `_Placing` puts the paths of a pattern's copy.

The copies of both families test the placing of fsspec's cases. These test the
edges of a pattern that none of them reaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies_impl.common.jobs import _Roots
from fsspec_alt_strategies_impl.common.placement import _Placing

if TYPE_CHECKING:
    from collections.abc import Sequence


class _Copy(_Placing):
    """A copy whose glob finds the roots given."""

    def __init__(self, path1: str, path2: str, roots: _Roots) -> None:
        self.path1 = path1
        self.path2 = path2
        self.roots = roots
        self.found = []
        self._placing(self.info, same_storage=True)

    async def info(self, path: str) -> dict[str, Any]:
        raise FileNotFoundError(path)

    async def expand(self, pattern: str | Sequence[str]) -> _Roots:
        return self.roots


async def test_a_pattern_matching_only_errors_yields_them() -> None:
    roots = _Roots([], [])
    assert await _Copy("x*", "dest", roots).prepare() is roots


async def test_a_single_match_already_in_place_is_left() -> None:
    copy = _Copy("a/*.txt", "a", _Roots([], ["a/x.txt"]))
    assert await copy.prepare() is None


async def test_matches_without_a_common_folder_go_under_the_destination() -> None:
    copy = _Copy("*1", "dest", _Roots(["x1"], ["y1"]))
    assert await copy.prepare() is not None
    assert copy.prefix == ""
    assert (copy.target("x1/f"), copy.target("y1")) == ("dest/x1/f", "dest/y1")
