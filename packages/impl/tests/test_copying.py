"""Tests of `common.copying`: where `_Copying` puts the paths of a pattern's copy.

The copies of both families test the placing of fsspec's cases. These test the
edges of a pattern that none of them reaches.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from fsspec_alt_strategies.states import CpState
from fsspec_alt_strategies_impl.common.copying import _Copying
from fsspec_alt_strategies_impl.common.jobs import _Roots
from fsspec_alt_strategies_impl.common.operation import Operation


async def _missing(path: str) -> dict[str, Any]:
    raise FileNotFoundError(path)


class _Copy(_Copying):
    """A copy whose glob finds the roots given, and which copies nothing."""

    def __init__(self, path1: str, path2: str, roots: _Roots) -> None:
        storage = SimpleNamespace(operation_limit=asyncio.Semaphore(1), _info=_missing)
        super().__init__(
            fs=storage,
            destination=storage,
            path1=path1,
            path2=path2,
            same_storage=True,
            op=Operation(lambda op: self.outcomes(), CpState()),
        )
        self.roots = roots

    async def glob_roots(self, pattern: Any) -> _Roots:
        return self.roots

    # The jobs, never run here: only the placing is.

    def source(self, roots: _Roots) -> Any:
        raise NotImplementedError

    def fed(self, item: Any) -> list[Any]:
        raise NotImplementedError

    def file_root(self, path: str) -> None:
        raise NotImplementedError

    def run(self, job: Any) -> Any:
        raise NotImplementedError

    def done(self, job: Any) -> list[Any]:
        raise NotImplementedError

    def failed(self, job: Any, error: Exception) -> list[Any]:
        raise NotImplementedError


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
