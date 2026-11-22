"""Tests that every flat strategy passes its ``ils_kwargs`` on to each recursive ``ils``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.flat.cp import _flat_cp
from fsspec_alt_strategies_impl.flat.du import _flat_du
from fsspec_alt_strategies_impl.flat.find import _flat_find
from fsspec_alt_strategies_impl.flat.glob import _flat_glob
from fsspec_alt_strategies_impl.flat.mv import _flat_mv
from fsspec_alt_strategies_impl.flat.rm import _flat_rm
from fsspec_alt_strategies_impl.flat.walk import _flat_walk
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


class _Recording(FlatView):
    """A view noting the options of each recursive ``ils``."""

    def __init__(self, fs: Any) -> None:
        super().__init__(fs)
        self.options: list[dict[str, Any]] = []

    def ils(
        self,
        entry: dict[str, Any],
        recursive: bool = False,
        *,
        prefix: str = "",
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        if recursive:
            self.options.append(kwargs)
            return super().ils(entry, recursive=True, prefix=prefix)
        else:
            return super().ils(entry, **kwargs)


_STRATEGIES: dict[str, Callable[..., Any]] = {
    "walk": lambda h, **kw: _flat_walk(h, "data", **kw),
    "find": lambda h, **kw: _flat_find(h, "data", **kw),
    "du": lambda h, **kw: _flat_du(h, "data", **kw),
    "glob": lambda h, **kw: _flat_glob(h, "data/*/*", **kw),
    "rm": lambda h, **kw: _flat_rm(h, "data", **kw),
    "rm a pattern": lambda h, **kw: _flat_rm(h, "data/*", **kw),
    "cp": lambda h, **kw: _flat_cp(h, h, "data", "copy", **kw),
    "mv": lambda h, **kw: _flat_mv(h, h, "data", "moved", **kw),
}


@pytest.mark.parametrize("strategy", list(_STRATEGIES))
async def test_ils_kwargs_go_to_each_listing(
    flat_fs: SimulatedFileSystem, strategy: str
) -> None:
    flat_fs.add_file("data/a/x", b"x")
    flat_fs.add_file("data/b/y", b"y")
    handler = _Recording(flat_fs)

    _, errors = await _collect_with_errors(
        _STRATEGIES[strategy](handler, ils_kwargs={"versions": True})
    )

    assert errors == []
    assert handler.options
    assert all(o == {"versions": True} for o in handler.options)


async def test_the_listings_own_prefix_is_kept(flat_fs: SimulatedFileSystem) -> None:
    flat_fs.add_file("data/2024-01", b"x")
    flat_fs.add_file("data/2025-01", b"x")
    handler = _Recording(flat_fs)

    matches, _ = await _collect_with_errors(
        _flat_glob(handler, "data/2024-*", ils_kwargs={"prefix": "2025-"})
    )

    assert [m["name"] for m in matches] == ["data/2024-01"]
