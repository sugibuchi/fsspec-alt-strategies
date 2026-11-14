"""Tests that every tree strategy passes its ``ils_kwargs`` on to each ``ils``."""

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
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView, dir_ils

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


_STRATEGIES: dict[str, Callable[..., Any]] = {
    "walk": lambda h, **kw: _tree_walk(h, "data", **kw),
    "find": lambda h, **kw: _tree_find(h, "data", maxdepth=2, **kw),
    "du": lambda h, **kw: _tree_du(h, "data", maxdepth=2, **kw),
    "glob": lambda h, **kw: _tree_glob(h, "data/*/*", **kw),
    "rm": lambda h, **kw: _tree_rm(h, "data", **kw),
    "rm a pattern": lambda h, **kw: _tree_rm(h, "data/*", **kw),
    "cp": lambda h, **kw: _tree_cp(h, h, "data", "copy", **kw),
    "mv": lambda h, **kw: _tree_mv(h, h, "data", "moved", **kw),
}


@pytest.mark.parametrize("strategy", list(_STRATEGIES))
async def test_ils_kwargs_go_to_each_listing(strategy: str) -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    fs.add_file("data/a/x", b"x")
    fs.add_file("data/b/y", b"y")
    options: list[dict[str, Any]] = []
    listed = dir_ils(fs)

    def ils(path: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        options.append(kwargs)
        return listed(path)

    _, errors = await _collect_with_errors(
        _STRATEGIES[strategy](TreeView(fs, ils=ils), ils_kwargs={"versions": True})
    )

    assert errors == []
    assert options
    assert all(o == {"versions": True} for o in options)
