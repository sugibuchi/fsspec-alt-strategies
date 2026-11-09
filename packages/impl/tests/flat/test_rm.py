"""Tests of `_flat_rm`, the delete that deletes a prefix's keys in batches as it lists."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.states import RmState
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.flat.rm import _flat_rm
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _view(fs: Any, kwargs: dict[str, Any]) -> FlatView:
    """``fs`` as a `FlatView`, with the primitives, limit and batch size ``kwargs`` give."""
    view = {k: kwargs.pop(k) for k in ("iexpand", "info", "rm_files") if k in kwargs}
    if "concurrency_limit" in kwargs:
        view["operation_limit"] = kwargs.pop("concurrency_limit")
    if "batch_size" in kwargs:
        view["rm_batch_size"] = kwargs.pop("batch_size")
    return FlatView(fs, **view)


def _rm(fs: SimulatedFileSystem, path: Any, **kwargs: Any) -> Any:
    """`_flat_rm` on ``fs``, whose primitives ``kwargs`` may replace."""
    return _flat_rm(_view(fs, kwargs), path, **kwargs)


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
    deleted, errors = await _collect_with_errors(_rm(flat_fs, "data"))
    assert errors == []
    assert sorted(deleted) == _FILES
    assert sorted(flat_fs._file_data) == ["database.csv", "other/kept"]


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _rm(fs, "data", batch_size=2)
    await _collect_with_errors(op)
    # 5 keys: 3 pages of 2, and 3 batches of 2. No info, and no request per key.
    assert fs.stats.calls == {"list": 3, "delete": 3}
    assert op.state == RmState(current=op.state.current, files_found=5, files_deleted=5)


async def test_deletes_while_the_listing_goes_on(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    events: list[str] = []

    async def iexpand(prefix: str) -> AsyncIterator[dict[str, Any]]:
        async for entry in flat_fs._iexpand(prefix):
            events.append("listed")
            yield entry

    async def rm_files(paths: list[str]) -> None:
        events.append("deleted")
        await flat_fs._rm_files(paths)

    await _collect_with_errors(
        _rm(flat_fs, "data", iexpand=iexpand, rm_files=rm_files, batch_size=2)
    )
    # The first batch is deleted before the listing has given the last key.
    assert events.index("deleted") < len(events) - 1 - events[::-1].index("listed")
    assert events.count("deleted") == 3


async def test_a_file(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    assert await _collect_with_errors(_rm(flat_fs, "data/README.md")) == (
        ["data/README.md"],
        [],
    )
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
    _, errors = await _collect_with_errors(_rm(flat_fs, path))
    assert errors == []
    assert set(flat_fs._file_data) == before - expanded


@pytest.mark.parametrize("path", ["missing", "data/*/13", ["missing", "data/*/13"]])
async def test_nothing_to_delete_raises(
    flat_fs: SimulatedFileSystem, path: Any
) -> None:
    _tree(flat_fs)
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(_rm(flat_fs, path))


async def test_a_failed_batch_is_yielded_and_the_others_go(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)

    async def rm_files(paths: list[str]) -> None:
        if "data/2023/01/part-0.parquet" in paths:
            raise PermissionError("refused")
        await flat_fs._rm_files(paths)

    deleted, errors = await _collect_with_errors(
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

    async def iexpand(prefix: str) -> AsyncIterator[dict[str, Any]]:
        for name in sorted(keys):
            if name.startswith(prefix):
                yield {"name": name, "size": keys[name], "type": "file"}

    async def rm_files(paths: list[str]) -> None:
        deleted_keys.extend(paths)

    async def info(path: str) -> dict[str, Any]:
        raise AssertionError("not asked")

    view = _view(
        None,
        {
            "iexpand": iexpand,
            "info": info,
            "rm_files": rm_files,
            "batch_size": 10,
            "concurrency_limit": asyncio.Semaphore(4),
        },
    )
    op = _flat_rm(view, "data")
    await _collect_with_errors(op)
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


@pytest.mark.parametrize("maxdepth", [1, 2, 3])
@pytest.mark.parametrize(
    "path", ["data", "data/*", "data/20*/*", ["data/2023", "data/README.md"]]
)
async def test_maxdepth_deletes_what_fsspecs_rm_does(
    flat_fs: SimulatedFileSystem, path: Any, maxdepth: int
) -> None:
    _tree(flat_fs)
    expanded = await AsyncFileSystem._expand_path(
        flat_fs, path, recursive=True, maxdepth=maxdepth
    )
    files = sorted(p for p in expanded if p in flat_fs._file_data)
    flat_fs.reset_stats()

    deleted, errors = await _collect_with_errors(_rm(flat_fs, path, maxdepth=maxdepth))
    assert errors == []
    assert sorted(deleted) == files
    assert not set(files) & set(flat_fs._file_data)


async def test_maxdepth_requests(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    op = _rm(flat_fs, "data", maxdepth=1)
    deleted, _ = await _collect_with_errors(op)
    assert deleted == ["data/README.md"]
    # The prefix lists every key under it: those deeper are left out.
    assert flat_fs.stats.calls == {"list": 1, "delete": 1}
    assert op.state == RmState(current=op.state.current, files_found=1, files_deleted=1)


async def test_maxdepth_1_of_a_pattern_lists_no_folder_matched(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    deleted, _ = await _collect_with_errors(_rm(flat_fs, "data/*", maxdepth=1))
    # fsspec's _expand_path takes the matches only: the folders are not expanded.
    assert deleted == ["data/README.md"]
    assert await flat_fs._exists("data/2023/01/part-0.parquet")


async def test_maxdepth_below_1_is_refused(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="maxdepth"):
        _rm(flat_fs, "data", maxdepth=0)
