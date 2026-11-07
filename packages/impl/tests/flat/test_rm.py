"""Tests of `flat_rm`, the delete that deletes a prefix's keys in batches as it lists."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.common.operation import _collect
from fsspec_alt_strategies.flat.rm import RmState, flat_rm
from fsspec_alt_strategies_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _rm(fs: SimulatedFileSystem, path: Any, **kwargs: Any) -> Any:
    """`flat_rm` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "ils": fs._ils,
        "info": fs._info,
        "rm_files": fs._rm_files,
        "batch_size": 1000,
    }
    return flat_rm(path, **{**operations, **kwargs})


_FILES = [
    "data/2023/01/part-0.parquet",
    "data/2023/02/part-0.parquet",
    "data/2024/01/part-0.parquet",
    "data/2024/02/part-0.parquet",
    "data/README.md",
]


def _tree(fs: SimulatedFileSystem) -> None:
    for path in _FILES:
        fs.add_file(path)
    fs.add_file("database.csv")  # Shares the text "data", not the folder.
    fs.add_file("other/kept")


async def test_deletes_the_keys_under_the_prefix(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    deleted, errors = await _collect(_rm(flat_fs, "data"))
    assert errors == []
    assert sorted(deleted) == _FILES
    assert sorted(flat_fs._file_data) == ["database.csv", "other/kept"]


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _rm(fs, "data", batch_size=2)
    await _collect(op)
    # 5 keys: 3 pages of 2, and 3 batches of 2. No info, and no request per key.
    assert fs.stats.calls == {"list": 3, "delete": 3}
    assert op.state == RmState(current=op.state.current, files_found=5, files_deleted=5)


async def test_deletes_while_the_listing_goes_on(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    events: list[str] = []

    async def ils(prefix: str) -> AsyncIterator[dict[str, Any]]:
        async for entry in flat_fs._ils(prefix):
            events.append("listed")
            yield entry

    async def rm_files(paths: list[str]) -> None:
        events.append("deleted")
        await flat_fs._rm_files(paths)

    await _collect(_rm(flat_fs, "data", ils=ils, rm_files=rm_files, batch_size=2))
    # The first batch is deleted before the listing has given the last key.
    assert events.index("deleted") < len(events) - 1 - events[::-1].index("listed")
    assert events.count("deleted") == 3


async def test_a_file(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    assert await _collect(_rm(flat_fs, "data/README.md")) == (["data/README.md"], [])
    assert "data/README.md" not in flat_fs._file_data


@pytest.mark.parametrize(
    "path",
    ["data/*/01", "data/20*", "data/**/*.parquet", "*/kept", ["data/2023", "other"]],
)
async def test_patterns_delete_what_fsspec_expands(
    flat_fs: SimulatedFileSystem, path: Any
) -> None:
    _tree(flat_fs)
    expanded: set[str] = set()
    for p in [path] if isinstance(path, str) else path:
        with contextlib.suppress(FileNotFoundError):
            expanded |= set(
                await AsyncFileSystem._expand_path(flat_fs, p, recursive=True)
            )
    before = set(flat_fs._file_data)
    _, errors = await _collect(_rm(flat_fs, path))
    assert errors == []
    assert set(flat_fs._file_data) == before - expanded


@pytest.mark.parametrize("path", ["missing", "data/*/13", ["missing", "data/*/13"]])
async def test_nothing_to_delete_raises(
    flat_fs: SimulatedFileSystem, path: Any
) -> None:
    _tree(flat_fs)
    with pytest.raises(FileNotFoundError):
        await _collect(_rm(flat_fs, path))


async def test_a_failed_batch_is_yielded_and_the_others_go(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)

    async def rm_files(paths: list[str]) -> None:
        if "data/2023/01/part-0.parquet" in paths:
            raise PermissionError("refused")
        await flat_fs._rm_files(paths)

    deleted, errors = await _collect(
        _rm(flat_fs, "data", rm_files=rm_files, batch_size=2)
    )
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == [
        "Deleting 'data/2023/01/part-0.parquet', 'data/2023/02/part-0.parquet' failed."
    ]
    assert len(deleted) == 3
    assert "data/2023/01/part-0.parquet" in flat_fs._file_data


async def test_markers_are_deleted_as_keys() -> None:
    keys = {"data/": 0, "data/empty/": 0, "data/a.txt": 1}
    deleted_keys: list[str] = []

    async def ils(prefix: str) -> AsyncIterator[dict[str, Any]]:
        for name in sorted(keys):
            if name.startswith(prefix):
                yield {"name": name, "size": keys[name], "type": "file"}

    async def rm_files(paths: list[str]) -> None:
        deleted_keys.extend(paths)

    async def info(path: str) -> dict[str, Any]:
        raise AssertionError("not asked")

    op = flat_rm(
        "data",
        ils=ils,
        info=info,
        rm_files=rm_files,
        batch_size=10,
        concurrency_limit=asyncio.Semaphore(4),
    )
    await _collect(op)
    assert sorted(deleted_keys) == ["data/", "data/a.txt", "data/empty/"]


async def test_cancel_sends_no_new_batch() -> None:
    fs = SimulatedFileSystem(page_size=1, latency=0.001, asynchronous=True)
    _tree(fs)
    op = _rm(fs, "data", batch_size=1)
    async for _ in op:
        op.cancel()
    assert op.cancelled()
    # The batches sent before the stop went; the rest of data is kept.
    assert any(p.startswith("data/") for p in fs._file_data)


def test_batch_size_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        _rm(flat_fs, "data", batch_size=0)
