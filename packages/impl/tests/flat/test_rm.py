"""Tests of `_flat_rm` on a storage listing keys: it deletes them in batches as it lists."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies.states import RmState
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.rm import _flat_rm
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _view(fs: Any, kwargs: dict[str, Any]) -> FlatView:
    """``fs`` as a `FlatView`, with the primitives, limit and batch size ``kwargs`` give."""
    view = {
        k: kwargs.pop(k) for k in ("list_prefix", "info", "rm_files") if k in kwargs
    }
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
    fs.add_file("database.csv")  # Shares the text "data", not the directory.
    fs.add_file("other/kept")


async def test_deletes_the_keys_under_the_prefix(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    deleted, errors = await _collect_with_errors(_rm(flat_fs, "data"))
    assert errors == []
    # The root too, by rmdir: it may have a marker.
    assert sorted(deleted) == ["data", *_FILES]
    assert sorted(flat_fs._file_data) == ["database.csv", "other/kept"]


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _rm(fs, "data", batch_size=2)
    await _collect_with_errors(op)
    # 5 keys: 3 pages of 2, and 3 batches of 2. No info, and no request per key.
    assert fs.stats.calls == {"list": 3, "delete": 3}
    assert op.state == RmState(
        current=op.state.current,
        files_found=5,
        files_deleted=5,
        directories_deleted=1,  # The root, which may have a marker.
    )


async def test_deletes_while_the_listing_goes_on(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    for i in range(40):  # More than the events the traversal holds unread.
        flat_fs.add_file(f"data/many/{i:02}")
    events: list[str] = []

    async def list_prefix(prefix: str) -> AsyncIterator[dict[str, Any]]:
        async for entry in flat_fs._list_prefix(prefix):
            events.append("listed")
            yield entry

    async def rm_files(paths: list[str]) -> None:
        events.append("deleted")
        await flat_fs._rm_files(paths)

    await _collect_with_errors(
        _rm(flat_fs, "data", list_prefix=list_prefix, rm_files=rm_files, batch_size=2)
    )
    # The first batch is deleted before the listing has given the last key.
    assert events.index("deleted") < len(events) - 1 - events[::-1].index("listed")
    assert events.count("deleted") == 23


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
    # As the storage raised it, with no note.
    assert [type(e) for e in errors] == [PermissionError]
    assert not getattr(errors[0], "__notes__", [])
    assert len(deleted) == 3
    assert "data/2023/01/part-0.parquet" in flat_fs._file_data


async def test_a_key_its_batch_refused_is_yielded_and_the_others_counted(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    refused = "data/2023/01/part-0.parquet"

    async def rm_files(paths: list[str]) -> dict[str, Exception]:
        # As S3's DeleteObjects answers: the batch done, but for its Errors.
        await flat_fs._rm_files([p for p in paths if p != refused])
        return {refused: PermissionError("retention")} if refused in paths else {}

    op = _rm(flat_fs, "data", rm_files=rm_files, batch_size=2)
    deleted, errors = await _collect_with_errors(op)
    assert [type(e) for e in errors] == [PermissionError]
    assert len(deleted) == 4 and refused not in deleted
    assert refused in flat_fs._file_data
    assert (op.state.files_deleted, op.state.errors) == (4, 1)


async def test_markers_are_deleted_by_rmdir() -> None:
    keys = {"data/": 0, "data/empty/": 0, "data/a.txt": 1, "data/b/c.txt": 1}
    deleted_keys: list[str] = []
    directories: list[str] = []

    async def list_prefix(prefix: str) -> AsyncIterator[dict[str, Any]]:
        for name in sorted(keys):
            if name.startswith(prefix):
                yield {"name": name, "size": keys[name], "type": "file"}

    async def rm_files(paths: list[str]) -> None:
        deleted_keys.extend(paths)

    async def rm_dir(path: str) -> None:
        directories.append(path)

    async def info(path: str) -> dict[str, Any]:
        raise AssertionError("not asked")

    view = FlatView(
        None,
        list_prefix=list_prefix,
        info=info,
        rm_files=rm_files,
        rm_dir=rm_dir,
        rm_batch_size=10,
        operation_limit=asyncio.Semaphore(4),
        storage_profile=StorageProfile(implicit_directories=True),
    )
    _, errors = await _collect_with_errors(_flat_rm(view, "data"))
    assert errors == []
    assert sorted(deleted_keys) == ["data/a.txt", "data/b/c.txt"]
    # A directory with a marker, and the root, which may have one: not data/b.
    assert sorted(directories) == ["data", "data/empty"]


async def test_cancel_sends_no_new_batch() -> None:
    fs = SimulatedFileSystem(page_size=1, latency=0.001, asynchronous=True)
    _tree(fs)
    op = _rm(fs, "data", batch_size=1)
    async for _ in op:
        op.cancel()
    assert op.cancelled()
    # The batches sent before the stop went; the rest of data is kept.
    assert any(p.startswith("data/") for p in fs._file_data)


async def test_batch_size_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    # Read once the handler is open, as the operation starts.
    with pytest.raises(ValueError, match="batch_size"):
        await _collect_with_errors(_rm(flat_fs, "data", batch_size=0))


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
    # Within maxdepth, a directory per listing: data alone.
    assert flat_fs.stats.calls == {"list": 1, "delete": 1}
    assert op.state == RmState(
        current=op.state.current, files_found=1, files_deleted=1, directories_deleted=0
    )


async def test_maxdepth_1_of_a_pattern_lists_no_dir_matched(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    deleted, _ = await _collect_with_errors(_rm(flat_fs, "data/*", maxdepth=1))
    # fsspec's _expand_path takes the matches only: the directories are not expanded.
    assert deleted == ["data/README.md"]
    assert await flat_fs._exists("data/2023/01/part-0.parquet")


async def test_maxdepth_below_1_is_refused(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="maxdepth"):
        _rm(flat_fs, "data", maxdepth=0)


async def test_files_matched_or_listed_are_batched_too(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    batches: list[list[str]] = []

    async def rm_files(paths: list[str]) -> None:
        batches.append(sorted(paths))
        await flat_fs._rm_files(paths)

    files = ["data/README.md", "data/2023/01/part-0.parquet"]
    op = _rm(flat_fs, files, rm_files=rm_files, batch_size=10)
    deleted, errors = await _collect_with_errors(op)
    assert errors == []
    # One request for both, rather than one per file.
    assert batches == [sorted(files)]
    assert set(files) <= set(deleted)
    assert op.state.files_found == op.state.files_deleted == 2


async def test_a_single_file_is_deleted_by_a_batch(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    batches: list[list[str]] = []

    async def rm_files(paths: list[str]) -> None:
        batches.append(paths)
        await flat_fs._rm_files(paths)

    path = "data/README.md"
    deleted, errors = await _collect_with_errors(
        _rm(flat_fs, path, rm_files=rm_files, batch_size=10)
    )
    assert (deleted, errors, batches) == ([path], [], [[path]])
    assert path not in flat_fs._file_data
