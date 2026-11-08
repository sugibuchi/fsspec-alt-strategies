"""Tests of `flat_walk`, checked against `tree_walk` on the same flat storage."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fsspec_alt_strategies.common.operation import _collect_with_errors
from fsspec_alt_strategies.flat.walk import WalkState, flat_walk
from fsspec_alt_strategies.tree.walk import tree_walk
from fsspec_alt_strategies_testing import SimulatedFileSystem


def _walk(fs: SimulatedFileSystem, path: str, **kwargs: Any) -> Any:
    """`flat_walk` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "ils": fs._ils,
        "info": fs._info,
    }
    return flat_walk(path, **{**operations, **kwargs})


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x")
    fs.add_file("data/README.md")
    fs.add_file("database.csv")  # Shares the text "data", not the folder.


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
    entries, errors = await _collect_with_errors(
        _walk(flat_fs, path, topdown=topdown, **kwargs)
    )
    assert errors == []
    folders, _ = await _collect_with_errors(
        tree_walk(
            path,
            ls=flat_fs._ls,
            topdown=topdown,
            concurrency_limit=asyncio.Semaphore(32),
            **kwargs,
        )
    )
    # tree_walk yields each folder's entries together: all of them, here.
    expected = [
        e for _, dirs, files in folders for e in [*dirs.values(), *files.values()]
    ]
    key = lambda e: e["name"]  # noqa: E731
    assert sorted(entries, key=key) == sorted(expected, key=key)


def _before(names: list[str], first: str, then: str) -> bool:
    return names.index(first) < names.index(then)


@pytest.mark.parametrize("topdown", [True, False])
async def test_folders_come_before_or_after_all_in_them(
    flat_fs: SimulatedFileSystem, topdown: bool
) -> None:
    _tree(flat_fs)
    entries, _ = await _collect_with_errors(_walk(flat_fs, "data", topdown=topdown))
    names = [e["name"] for e in entries]
    for folder in (e["name"] for e in entries if e["type"] == "directory"):
        for inside in (n for n in names if n.startswith(folder + "/")):
            assert _before(names, folder, inside) is topdown


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _walk(fs, "data")
    entries, _ = await _collect_with_errors(op)
    # 5 keys, 2 a page: no listing per folder, and no info.
    assert fs.stats.calls == {"list": 3}
    assert len(entries) == 5 + 6  # The files, and data's 6 folders.
    assert op.state == WalkState(current=op.state.current, found=11)


@pytest.mark.parametrize(
    ("path", "error"),
    [("missing", FileNotFoundError), ("data/README.md", NotADirectoryError)],
)
async def test_a_root_not_a_folder_raises(
    flat_fs: SimulatedFileSystem, path: str, error: type[Exception]
) -> None:
    _tree(flat_fs)
    with pytest.raises(error):
        await _collect_with_errors(_walk(flat_fs, path))


@pytest.mark.parametrize("topdown", [True, False])
async def test_cancel_takes_no_more_from_the_listing(topdown: bool) -> None:
    fs = SimulatedFileSystem(page_size=1, asynchronous=True)
    _tree(fs)
    op = _walk(fs, "data", topdown=topdown)
    names = []
    async for entry in op:
        names.append(entry["name"])
        op.cancel()
    assert op.cancelled()
    assert fs.stats.calls == {"list": 1}
    # Bottom up, only the first key's file: its folders were not known done.
    assert names == (
        ["data/2023", "data/2023/01", "data/2023/01/part-0.parquet"]
        if topdown
        else ["data/2023/01/part-0.parquet"]
    )


def test_maxdepth_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _walk(flat_fs, "data", maxdepth=0)


async def test_the_roots_own_marker_is_not_yielded(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    without, _ = await _collect_with_errors(_walk(flat_fs, "data"))

    async def ils(prefix: str) -> Any:
        yield {"name": "data/", "size": 0, "type": "file"}
        async for entry in flat_fs._ils(prefix):
            yield entry

    entries, errors = await _collect_with_errors(_walk(flat_fs, "data", ils=ils))
    assert errors == []
    assert entries == without
