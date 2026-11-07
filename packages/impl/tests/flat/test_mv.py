"""Tests of `flat_mv`, moving each key, or copying it and deleting in batches."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies.common.monitors import _collect
from fsspec_alt_strategies.flat.mv import MvState, flat_mv
from fsspec_alt_strategies_common.profile import StorageProfile
from fsspec_alt_strategies_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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
    moved, errors = await _collect(_mv(flat_fs, "data", "moved"))
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
    await _collect(op)
    # 2 pages, a copy per key, and the 4 sources deleted in 2 batches.
    assert fs.stats.calls == {"list": 2, "copy": 4, "delete": 2}
    assert op.state == MvState(current=op.state.current, files_found=4, files_moved=4)


async def test_a_pattern_moves_its_matches(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    moved, errors = await _collect(_mv(flat_fs, "data/202*", "moved"))
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

    moved, errors = await _collect(_mv(flat_fs, "data", "moved", rm_files=rm_files))
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

    moved, errors = await _collect(_mv(flat_fs, "data", "moved", cp_file=cp_file))
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
    moved, _ = await _collect(op)
    assert op.cancelled()
    # Each key copied was deleted too: none is in both places.
    assert copied
    assert sorted(src for src, _ in moved) == sorted(copied)
    for source in copied:
        assert source not in fs._file_data


def test_batch_size_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _mv(flat_fs, "data", "moved", batch_size=0)


_BY_MV_FILE = {"cp_file": None, "rm_files": None, "batch_size": None}


async def test_mv_file_moves_each_key(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    sent: list[tuple[str, str]] = []

    async def mv_file(source: str, destination: str) -> None:
        sent.append((source, destination))  # One request on a renaming storage.
        await flat_fs._mv_file(source, destination)

    op = _mv(flat_fs, "data", "moved", mv_file=mv_file, **_BY_MV_FILE)
    moved, errors = await _collect(op)
    assert errors == []
    expected = [(p, "moved" + p[len("data") :]) for p in _FILES]
    assert sorted(moved) == sorted(sent) == expected
    assert op.state == MvState(current=op.state.current, files_found=4, files_moved=4)
    assert flat_fs._file_data["moved/README.md"].data == b"data/README.md"
    assert not await flat_fs._exists("data")


async def test_mv_file_moves_markers_without_yielding_them() -> None:
    keys = {"data/": b"", "data/empty/": b"", "data/a.txt": b"a"}

    async def ils(prefix: str) -> AsyncIterator[dict[str, Any]]:
        for name in sorted(keys):
            if name.startswith(prefix):
                yield {"name": name, "size": len(keys[name]), "type": "file"}

    async def info(path: str) -> dict[str, Any]:
        raise FileNotFoundError(path)

    async def mv_file(source: str, destination: str) -> None:
        keys[destination] = keys.pop(source)

    op = flat_mv(
        "data",
        "moved",
        ils=ils,
        info=info,
        mv_file=mv_file,
        concurrency_limit=asyncio.Semaphore(4),
    )
    assert await _collect(op) == ([("data/a.txt", "moved/a.txt")], [])
    assert sorted(keys) == ["moved/", "moved/a.txt", "moved/empty/"]


@pytest.mark.parametrize(
    "operations",
    [
        {"mv_file": None, "cp_file": None},  # Neither way.
        {"mv_file": None, "batch_size": None},  # Batches, of no size.
        {"rm_files": None},  # Batches, of nothing to delete with, besides mv_file.
    ],
)
def test_a_way_to_move_is_whole(
    flat_fs: SimulatedFileSystem, operations: dict[str, Any]
) -> None:
    operations = {"mv_file": flat_fs._mv_file, **operations}
    with pytest.raises(ValueError, match="cp_file, rm_files and batch_size"):
        _mv(flat_fs, "data", "moved", **operations)


@pytest.mark.parametrize(
    ("renames_files", "by_mv_file"), [(True, True), (False, False), (None, False)]
)
async def test_the_profile_chooses_between_both_ways(
    flat_fs: SimulatedFileSystem, renames_files: bool | None, by_mv_file: bool
) -> None:
    _tree(flat_fs)
    renamed: list[str] = []

    async def mv_file(source: str, destination: str) -> None:
        renamed.append(source)
        await flat_fs._mv_file(source, destination)

    profile = StorageProfile(renames_files=renames_files)
    op = _mv(flat_fs, "data", "moved", mv_file=mv_file, profile=profile)
    moved, errors = await _collect(op)
    assert errors == []
    assert len(moved) == len(_FILES)
    # Unknown, a move copies, and deletes in batches: right on any storage.
    assert bool(renamed) == by_mv_file
    assert flat_fs.stats.calls["delete"] == (len(_FILES) if by_mv_file else 1)
