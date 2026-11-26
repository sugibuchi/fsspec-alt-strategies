"""Tests of `_tree_find` on a storage listing keys, checked against fsspec's own ``_find``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.states import FindState
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.find import _tree_find
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _find(fs: SimulatedFileSystem, path: str, **kwargs: Any) -> Any:
    """`_tree_find` on ``fs``, whose ``list_prefix`` or ``info`` ``kwargs`` may replace."""
    primitives = {k: kwargs.pop(k) for k in ("list_prefix", "info") if k in kwargs}
    return _tree_find(FlatView(fs, **primitives), path, **kwargs)


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * 10)
    fs.add_file("data/README.md")
    fs.add_file("database.csv")  # Shares the text "data", not the directory.


def _sorted(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda e: e["name"])


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"withdirs": True}, {"maxdepth": 1}, {"maxdepth": 2, "withdirs": True}],
)
@pytest.mark.parametrize(
    "path", ["data", "data/", "data/2024", "data/README.md", "missing", ""]
)
async def test_finds_as_fsspec(
    flat_fs: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    _tree(flat_fs)
    entries, errors = await _collect_with_errors(_find(flat_fs, path, **kwargs))
    assert errors == []
    expected = await AsyncFileSystem._find(flat_fs, path, detail=True, **kwargs)
    assert sorted(e["name"] for e in entries) == list(expected)
    assert _sorted(entries) == list(expected.values())


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _find(fs, "data")
    entries, _ = await _collect_with_errors(op)
    assert len(entries) == 5
    # The 5 keys under data/, 2 a page: no info, and no listing per directory.
    assert fs.stats.calls == {"list": 3}
    assert op.state == FindState(
        current=op.state.current, found=5, directories_listed=7
    )

    fs.reset_stats()
    await _collect_with_errors(_find(fs, "data", withdirs=True))
    assert fs.stats.calls == {"list": 3, "info": 1}  # The info of data itself.


def _listing(*names: str) -> Any:
    """An ``list_prefix`` of these keys, as a storage with directory markers lists them."""

    async def list_prefix(prefix: str) -> AsyncIterator[dict[str, Any]]:
        for name in names:
            if name.startswith(prefix):
                size = 0 if name.endswith("/") else 1
                yield {"name": name, "size": size, "type": "file"}

    return list_prefix


async def _info(path: str) -> dict[str, Any]:
    return {"name": path, "size": 0, "type": "directory"}


async def test_markers_are_dirs() -> None:
    list_prefix = _listing("data/", "data/empty/", "data/a/b.txt")
    entries, _ = await _collect_with_errors(
        _tree_find(
            FlatView(None, list_prefix=list_prefix, info=_info), "data", withdirs=True
        )
    )
    assert _sorted(entries) == [
        {"name": "data", "size": 0, "type": "directory"},
        {"name": "data/a", "size": 0, "type": "directory"},
        {"name": "data/a/b.txt", "size": 1, "type": "file"},
        # A marker: its directory, with an object of its own.
        {"name": "data/empty", "size": 0, "type": "directory", "marker": True},
    ]
    files, _ = await _collect_with_errors(
        _tree_find(FlatView(None, list_prefix=list_prefix, info=_info), "data")
    )
    assert [e["name"] for e in files] == ["data/a/b.txt"]


async def test_an_error_of_the_listing_ends_it() -> None:
    async def list_prefix(prefix: str) -> AsyncIterator[dict[str, Any]]:
        yield {"name": "data/a", "size": 1, "type": "file"}
        raise PermissionError(prefix)

    op = _tree_find(FlatView(None, list_prefix=list_prefix, info=_info), "data")
    found, errors = await _collect_with_errors(op)
    # The root's failure, yielded: what was listed before it is found.
    assert [e["name"] for e in found] == ["data/a"]
    assert [type(e) for e in errors] == [PermissionError]


def test_maxdepth_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _find(flat_fs, "data", maxdepth=0)
