"""Tests of `flat_cp`, placing paths as fsspec's ``copy`` does."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import other_paths

from fsspec_alt_strategies.common.operation import _collect
from fsspec_alt_strategies.flat.cp import CpState, flat_cp
from fsspec_alt_strategies_testing import SimulatedFileSystem


def _cp(fs: SimulatedFileSystem, path1: Any, path2: Any, **kwargs: Any) -> Any:
    """`flat_cp` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "ils": fs._ils,
        "info": fs._info,
        "cp_file": fs._cp_file,
    }
    return flat_cp(path1, path2, **{**operations, **kwargs})


_FILES = [
    "data/2023/01/part-0.parquet",
    "data/2023/02/part-0.parquet",
    "data/2024/01/part-0.parquet",
    "data/README.md",
]


def _tree(fs: SimulatedFileSystem, prefix: str = "") -> None:
    for path in _FILES:
        fs.add_file(prefix + path, path.encode())
    fs.add_file(prefix + "archive/old.txt")


@pytest.mark.parametrize(
    ("path1", "path2"),
    [
        ("data", "copied"),  # Onto a new path.
        ("data", "archive"),  # Into an existing folder.
        ("data", "new/"),  # Into a folder, by its trailing "/".
        ("data/", "archive"),  # Its contents onto an existing folder.
        ("data/README.md", "copied.md"),  # A file, onto a new path.
        ("data/README.md", "archive"),  # A file, into a folder.
    ],
)
async def test_places_as_fsspec(
    flat_fs: SimulatedFileSystem, path1: str, path2: str
) -> None:
    # In a bucket: fsspec's other_paths cannot place paths at the top level of
    # a filesystem whose root is "", as it finds no "/" to split them at.
    _tree(flat_fs, "bucket/")
    path1, path2 = f"bucket/{path1}", f"bucket/{path2}"
    root = path1.rstrip("/")
    files = sorted(p for p in flat_fs._file_data if p.startswith(root + "/"))
    paths = [root, *files] if files else [root]
    dest_is_dir = path2.endswith("/") or path2.rstrip("/") in flat_fs._folders
    exists = dest_is_dir and not path1.endswith("/")
    expected = dict(zip(paths, other_paths(paths, path2, exists=exists), strict=True))
    data = {p: flat_fs._file_data[p].data for p in paths if p in flat_fs._file_data}

    copied, errors = await _collect(_cp(flat_fs, path1, path2))
    assert errors == []
    assert sorted(copied) == sorted((p, expected[p]) for p in data)
    for source, content in data.items():
        assert flat_fs._file_data[expected[source]].data == content
        assert flat_fs._file_data[source].data == content  # The source is kept.


@pytest.mark.parametrize(
    ("pattern", "path2"),
    [
        ("data/*/01", "copied"),  # Folders, relative to their common prefix.
        ("data/20*", "copied"),
        ("data/**/*.parquet", "copied"),  # Files only.
        ("data/*.md", "archive"),  # A single file goes into the destination.
        ("data/2024/*", "copied/"),  # A single folder goes onto it.
    ],
)
async def test_a_pattern_places_as_fsspec(
    flat_fs: SimulatedFileSystem, pattern: str, path2: str
) -> None:
    _tree(flat_fs, "bucket/")
    pattern, path2 = f"bucket/{pattern}", f"bucket/{path2}"
    paths = await AsyncFileSystem._expand_path(flat_fs, pattern, recursive=True)
    destinations = other_paths(paths, path2, exists=len(paths) == 1)
    expected = dict(zip(paths, destinations, strict=True))
    files = [p for p in paths if p in flat_fs._file_data]

    copied, errors = await _collect(_cp(flat_fs, pattern, path2))
    assert errors == []
    assert sorted(copied) == sorted((p, expected[p]) for p in files)


async def test_a_list_keeps_the_trees_of_its_folders(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    copied, errors = await _collect(
        _cp(flat_fs, ["data/2023", "data/2024/*", "data/README.md"], "copied")
    )
    assert errors == []
    # As cp -r places them, as tree_cp does.
    assert sorted(copied) == [
        ("data/2023/01/part-0.parquet", "copied/2023/01/part-0.parquet"),
        ("data/2023/02/part-0.parquet", "copied/2023/02/part-0.parquet"),
        ("data/2024/01/part-0.parquet", "copied/01/part-0.parquet"),
        ("data/README.md", "copied/README.md"),
    ]


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _cp(fs, "data", "copied/")
    copied, _ = await _collect(op)
    assert len(copied) == 4
    # 2 pages, a copy per key: no info for "copied/", and no mkdir.
    assert fs.stats.calls == {"list": 2, "copy": 4}
    assert op.state == CpState(current=op.state.current, files_found=4, files_copied=4)


async def test_paths_it_refuses(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    with pytest.raises(ValueError, match="into itself"):
        await _collect(_cp(flat_fs, "data", "data/inner"))
    with pytest.raises(FileNotFoundError):
        await _collect(_cp(flat_fs, "missing", "copied"))
