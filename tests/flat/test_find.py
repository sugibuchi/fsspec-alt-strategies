"""Tests of `flat_find`, checked against fsspec's own ``_find``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.control import collect
from fsspec_alt_strategies.flat.find import FlatFindState, flat_find
from fsspec_alt_strategies_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _find(fs: SimulatedFileSystem, path: str, **kwargs: Any) -> Any:
    """`flat_find` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {"ils": fs._ils, "info": fs._info}
    return flat_find(path, **{**operations, **kwargs})


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * 10)
    fs.add_file("data/README.md")
    fs.add_file("database.csv")  # Shares the text "data", not the folder.


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
    entries, errors = await collect(_find(flat_fs, path, **kwargs))
    assert errors == []
    expected = await AsyncFileSystem._find(flat_fs, path, detail=True, **kwargs)
    assert sorted(e["name"] for e in entries) == list(expected)
    assert _sorted(entries) == list(expected.values())


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _find(fs, "data")
    entries, _ = await collect(op)
    assert len(entries) == 5
    # The 5 keys under data/, 2 a page: no info, and no listing per folder.
    assert fs.stats.calls == {"list": 3}
    assert op.state == FlatFindState(current=op.state.current, found=5)

    fs.reset_stats()
    await collect(_find(fs, "data", withdirs=True))
    assert fs.stats.calls == {"list": 3, "info": 1}  # The info of data itself.


async def test_yields_as_the_pages_come() -> None:
    fs = SimulatedFileSystem(page_size=1, asynchronous=True)
    _tree(fs)
    async with _find(fs, "data") as op:
        await anext(op)
        # The first path, after the first page only.
        assert fs.stats.calls == {"list": 1}
        op.cancel()
        assert [e async for e in op] == []
    # Cancelled, it asked for no more pages.
    assert fs.stats.calls == {"list": 1}
    assert op.cancelled()


def _listing(*names: str) -> Any:
    """An ``ils`` of these keys, as a storage with folder markers lists them."""

    async def ils(prefix: str) -> AsyncIterator[dict[str, Any]]:
        for name in names:
            if name.startswith(prefix):
                size = 0 if name.endswith("/") else 1
                yield {"name": name, "size": size, "type": "file"}

    return ils


async def _info(path: str) -> dict[str, Any]:
    return {"name": path, "size": 0, "type": "directory"}


async def test_markers_are_folders() -> None:
    ils = _listing("data/", "data/empty/", "data/a/b.txt")
    entries, _ = await collect(flat_find("data", ils=ils, info=_info, withdirs=True))
    assert _sorted(entries) == [
        {"name": "data", "size": 0, "type": "directory"},
        {"name": "data/a", "size": 0, "type": "directory"},
        {"name": "data/a/b.txt", "size": 1, "type": "file"},
        {"name": "data/empty", "size": 0, "type": "directory"},
    ]
    files, _ = await collect(flat_find("data", ils=ils, info=_info))
    assert [e["name"] for e in files] == ["data/a/b.txt"]


async def test_an_error_of_the_listing_ends_it() -> None:
    async def ils(prefix: str) -> AsyncIterator[dict[str, Any]]:
        yield {"name": "data/a", "size": 1, "type": "file"}
        raise PermissionError(prefix)

    op = flat_find("data", ils=ils, info=_info)
    with pytest.raises(PermissionError):
        await collect(op)


def test_maxdepth_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _find(flat_fs, "data", maxdepth=0)
