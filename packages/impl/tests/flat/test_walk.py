"""Tests of `_flat_walk`, checked against `_tree_walk` on the same flat storage."""

from __future__ import annotations

from typing import Any

import pytest

from fsspec_alt_strategies.states import WalkState
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.flat.walk import _flat_walk
from fsspec_alt_strategies_impl.tree.walk import _tree_walk
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem, TreeView


def _walk(fs: SimulatedFileSystem, path: str, **kwargs: Any) -> Any:
    """`_flat_walk` on ``fs``, whose ``list_prefix`` or ``info`` ``kwargs`` may replace."""
    primitives = {k: kwargs.pop(k) for k in ("list_prefix", "info") if k in kwargs}
    return _flat_walk(FlatView(fs, **primitives), path, **kwargs)


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x")
    fs.add_file("data/README.md")
    fs.add_file("database.csv")  # Shares the text "data", not the directory.


def _not_2023(entry: dict[str, Any]) -> bool:
    return not entry["name"].endswith("2023")


@pytest.mark.parametrize("topdown", [True, False])
@pytest.mark.parametrize(
    "kwargs", [{}, {"maxdepth": 1}, {"maxdepth": 2}, {"descend": _not_2023}]
)
@pytest.mark.parametrize("path", ["data", "data/2024", ""])
async def test_finds_what_tree_walk_finds(
    flat_fs: SimulatedFileSystem, path: str, kwargs: dict[str, Any], topdown: bool
) -> None:
    _tree(flat_fs)
    directories, errors = await _collect_with_errors(
        _walk(flat_fs, path, topdown=topdown, **kwargs)
    )
    assert errors == []
    expected, _ = await _collect_with_errors(
        _tree_walk(TreeView(flat_fs), path, topdown=topdown, **kwargs)
    )
    key = lambda f: f[0]  # noqa: E731
    assert sorted(directories, key=key) == sorted(expected, key=key)


def _before(names: list[str], first: str, then: str) -> bool:
    return names.index(first) < names.index(then)


@pytest.mark.parametrize("topdown", [True, False])
async def test_dirs_come_before_or_after_those_below_them(
    flat_fs: SimulatedFileSystem, topdown: bool
) -> None:
    _tree(flat_fs)
    directories, _ = await _collect_with_errors(_walk(flat_fs, "data", topdown=topdown))
    names = [f[0] for f in directories]
    for directory in names:
        for below in (n for n in names if n.startswith(directory + "/")):
            assert _before(names, directory, below) is topdown


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _walk(fs, "data")
    directories, _ = await _collect_with_errors(op)
    # 5 keys, 2 a page: no listing per directory, and no info.
    assert fs.stats.calls == {"list": 3}
    assert len(directories) == 7  # data and its 6 directories.
    assert op.state == WalkState(current=op.state.current, found=11)


async def test_bottom_up_yields_each_dir_once_the_listing_passes_it() -> None:
    fs = SimulatedFileSystem(page_size=1, asynchronous=True)
    _tree(fs)
    seen = []
    async for directory, _, _ in _walk(fs, "data", topdown=False):
        seen.append((directory, fs.stats.calls["list"]))
    # data/2023/01 is done once the second key, data/2023/02's, is listed.
    assert seen[0] == ("data/2023/01", 2)
    assert seen[-1] == ("data", 5)


@pytest.mark.parametrize(
    ("path", "error"),
    [("missing", FileNotFoundError), ("data/README.md", NotADirectoryError)],
)
async def test_a_root_not_a_dir_raises(
    flat_fs: SimulatedFileSystem, path: str, error: type[Exception]
) -> None:
    _tree(flat_fs)
    with pytest.raises(error):
        await _collect_with_errors(_walk(flat_fs, path))


async def test_cancel_takes_no_more_from_the_listing() -> None:
    fs = SimulatedFileSystem(page_size=1, asynchronous=True)
    _tree(fs)
    op = _walk(fs, "data", topdown=False)
    names = []
    async for directory, _, _ in op:
        names.append(directory)
        op.cancel()
    assert op.cancelled()
    # Cancelled once the second key closed the first directory: the directories still
    # open are left out.
    assert fs.stats.calls == {"list": 2}
    assert names == ["data/2023/01"]


def test_maxdepth_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _walk(flat_fs, "data", maxdepth=0)


async def test_the_roots_own_marker_is_not_yielded(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    without, _ = await _collect_with_errors(_walk(flat_fs, "data"))

    async def list_prefix(prefix: str) -> Any:
        yield {"name": "data/", "size": 0, "type": "file"}
        async for entry in flat_fs._list_prefix(prefix):
            yield entry

    directories, errors = await _collect_with_errors(
        _walk(flat_fs, "data", list_prefix=list_prefix)
    )
    assert errors == []
    assert directories == without
