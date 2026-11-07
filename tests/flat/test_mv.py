"""Tests of `flat_mv`, copying each key, then deleting them in batches."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fsspec_alt_strategies.control import collect
from fsspec_alt_strategies.flat.mv import MvState, flat_mv
from fsspec_alt_strategies_testing import SimulatedFileSystem


def _mv(fs: SimulatedFileSystem, path1: Any, path2: Any, **kwargs: Any) -> Any:
    """`flat_mv` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "ils": fs._ils,
        "info": fs._info,
        "cp_file": fs._cp_file,
        "rm_files": fs._rm_files,
        "batch_size": 1000,
    }
    return flat_mv(path1, path2, **{**operations, **kwargs})


_FILES = [
    "data/2023/01/part-0.parquet",
    "data/2023/02/part-0.parquet",
    "data/2024/01/part-0.parquet",
    "data/README.md",
]


def _tree(fs: SimulatedFileSystem) -> None:
    for path in _FILES:
        fs.add_file(path, path.encode())
    fs.add_file("other.txt")


async def test_moves_the_keys(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    moved, errors = await collect(_mv(flat_fs, "data", "moved"))
    assert errors == []
    assert sorted(moved) == [(p, "moved" + p[len("data") :]) for p in _FILES]
    assert sorted(flat_fs._file_data) == [
        "moved" + p[len("data") :] for p in _FILES
    ] + ["other.txt"]
    assert flat_fs._file_data["moved/README.md"].data == b"data/README.md"


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _mv(fs, "data", "moved/", batch_size=3)
    await collect(op)
    # 2 pages, a copy per key, and the 4 sources deleted in 2 batches.
    assert fs.stats.calls == {"list": 2, "copy": 4, "delete": 2}
    assert op.state == MvState(current=op.state.current, files_found=4, files_moved=4)


async def test_a_pattern_moves_its_matches(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    moved, errors = await collect(_mv(flat_fs, "data/202*", "moved"))
    assert errors == []
    # Relative to the matches' common prefix, "data", as fsspec places them.
    assert sorted(dst for _, dst in moved) == [
        "moved/2023/01/part-0.parquet",
        "moved/2023/02/part-0.parquet",
        "moved/2024/01/part-0.parquet",
    ]
    # Only the matches went: the README stays.
    assert [p for p in flat_fs._file_data if p.startswith("data/")] == [
        "data/README.md"
    ]


async def test_a_failed_batch_leaves_its_keys_in_both_places(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)

    async def rm_files(paths: list[str]) -> None:
        raise PermissionError("refused")

    moved, errors = await collect(_mv(flat_fs, "data", "moved", rm_files=rm_files))
    assert moved == []
    assert [type(e) for e in errors] == [PermissionError]
    assert "they are in both places" in errors[0].__notes__[0]
    assert all(p in flat_fs._file_data for p in _FILES)
    assert "moved/README.md" in flat_fs._file_data


async def test_a_failed_copy_keeps_its_source(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)

    async def cp_file(source: str, destination: str) -> None:
        if source == "data/README.md":
            raise PermissionError(source)
        await flat_fs._cp_file(source, destination)

    moved, errors = await collect(_mv(flat_fs, "data", "moved", cp_file=cp_file))
    assert len(moved) == 3
    assert errors[0].__notes__ == [
        "Moving 'data/README.md' to 'moved/README.md' failed."
    ]
    assert sorted(p for p in flat_fs._file_data if p.startswith("data/")) == [
        "data/README.md"
    ]


async def test_cancel_still_deletes_the_keys_copied() -> None:
    fs = SimulatedFileSystem(page_size=1, latency=0.001, asynchronous=True)
    _tree(fs)
    copied = []

    async def cp_file(source: str, destination: str) -> None:
        await fs._cp_file(source, destination)
        copied.append(source)
        op.cancel()

    op = _mv(fs, "data", "moved", cp_file=cp_file)
    moved, _ = await collect(op)
    assert op.cancelled()
    # Each key copied was deleted too: none is in both places.
    assert copied
    assert sorted(src for src, _ in moved) == sorted(copied)
    for source in copied:
        assert source not in fs._file_data


def test_batch_size_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _mv(flat_fs, "data", "moved", batch_size=0)
