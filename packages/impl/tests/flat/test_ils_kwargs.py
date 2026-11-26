"""Tests that every strategy on a storage listing keys passes its ``ils_kwargs`` on to each listing, which takes them or fails."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.cp import _tree_cp
from fsspec_alt_strategies_impl.tree.du import _tree_du
from fsspec_alt_strategies_impl.tree.find import _tree_find
from fsspec_alt_strategies_impl.tree.glob import _tree_glob
from fsspec_alt_strategies_impl.tree.mv import _tree_mv
from fsspec_alt_strategies_impl.tree.rm import _tree_rm
from fsspec_alt_strategies_impl.tree.walk import _tree_walk
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


class _Recording(FlatView):
    """A view taking options of its own, noting those of each listing."""

    def __init__(self, fs: Any) -> None:
        super().__init__(fs)
        self.options: list[dict[str, Any]] = []

    def ils(
        self, entry: dict[str, Any], *, versions: bool = False
    ) -> AsyncIterator[dict[str, Any]]:
        self.options.append({"versions": versions})
        return super().ils(entry)

    def ils_recursive(
        self, entry: dict[str, Any], prefix: bool = False, *, versions: bool = False
    ) -> AsyncIterator[dict[str, Any]]:
        self.options.append({"versions": versions})
        return super().ils_recursive(entry, prefix)


_STRATEGIES: dict[str, Callable[..., Any]] = {
    "walk": lambda h, **kw: _tree_walk(h, "data", **kw),
    "find": lambda h, **kw: _tree_find(h, "data", **kw),
    "du": lambda h, **kw: _tree_du(h, "data", **kw),
    "glob": lambda h, **kw: _tree_glob(h, "data/*/*", **kw),
    "rm": lambda h, **kw: _tree_rm(h, "data", **kw),
    "rm a pattern": lambda h, **kw: _tree_rm(h, "data/*", **kw),
    "cp": lambda h, **kw: _tree_cp(h, h, "data", "copy", **kw),
    "mv": lambda h, **kw: _tree_mv(h, h, "data", "moved", **kw),
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


@pytest.mark.parametrize("strategy", ["find", "glob"])
async def test_an_option_the_handler_takes_not_fails(
    flat_fs: SimulatedFileSystem, strategy: str
) -> None:
    flat_fs.add_file("data/a/x", b"x")
    # FlatView's listings take no option: the strategy passes it on, and its
    # root fails, as the listing raised.
    found, errors = await _collect_with_errors(
        _STRATEGIES[strategy](FlatView(flat_fs), ils_kwargs={"versions": True})
    )
    assert found == []
    assert [type(e) for e in errors] == [TypeError]
    assert "versions" in str(errors[0])
