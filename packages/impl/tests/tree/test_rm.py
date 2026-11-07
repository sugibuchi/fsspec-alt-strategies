"""Tests of `tree_rm`, the delete that walks and deletes at once."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.common.operation import _collect
from fsspec_alt_strategies.tree.rm import RmState, tree_rm
from fsspec_alt_strategies_common.profile import StorageProfile
from fsspec_alt_strategies_testing import SimulatedFileSystem


def _rm(fs: SimulatedFileSystem, path: Any, **kwargs: Any) -> Any:
    """`tree_rm` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "profile": StorageProfile(),
        "ls": fs._ls,
        "info": fs._info,
        "rm_file": fs._rm_file,
        "rm_folder": fs._rmdir,
    }
    return tree_rm(path, **{**operations, **kwargs})


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
    fs.add_file("other/kept")


async def test_deletes_the_tree(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    deleted, errors = await _collect(_rm(any_fs, "data"))
    assert errors == []
    assert set(_FILES) <= set(deleted)
    assert not await any_fs._exists("data")
    assert await any_fs._exists("other/kept")


async def test_requests_on_a_hierarchical_storage(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/empty")
    op = _rm(hierarchical_fs, "data")
    deleted, _ = await _collect(op)
    # The 5 files, then the 8 folders, data included.
    assert len(deleted) == 13
    assert deleted[-1] == "data"
    assert op.state == RmState(
        current="data", files_found=5, files_deleted=5, folders_deleted=8
    )
    # A listing per folder, then a deletion per file and per folder.
    assert hierarchical_fs.stats.calls == {"list": 8, "delete": 13}


async def test_requests_on_an_object_store(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    deleted, _ = await _collect(_rm(flat_fs, "data"))
    # Object stores have no folders to delete: _rmdir sends nothing.
    assert len(deleted) == 5 + 7
    assert flat_fs.stats.calls == {"list": 7, "delete": 5}


async def test_implicit_folders_save_the_folder_deletions(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    folders: list[str] = []

    async def rm_folder(path: str) -> None:
        folders.append(path)  # A request a real object store may send.
        await flat_fs._rmdir(path)

    profile = StorageProfile(implicit_folders=True)
    op = _rm(flat_fs, "data", rm_folder=rm_folder, profile=profile)
    deleted, errors = await _collect(op)
    assert (folders, errors) == ([], [])
    # As a folder found already gone: not yielded.
    assert sorted(deleted) == sorted(_FILES)
    assert op.state.folders_deleted == 0
    assert not await flat_fs._exists("data")
    assert await flat_fs._exists("other/kept")


async def test_a_file(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    assert await _collect(_rm(any_fs, "data/README.md")) == (["data/README.md"], [])
    assert not await any_fs._exists("data/README.md")


@pytest.mark.parametrize(
    "pattern",
    ["data/*/01", "data/20*", "data/**/*.parquet", "data/*", "*/kept", "data/2023/0?/"],
)
async def test_a_pattern_deletes_what_fsspec_expands(
    any_fs: SimulatedFileSystem, pattern: str
) -> None:
    _tree(any_fs)
    expanded = set(await AsyncFileSystem._expand_path(any_fs, pattern, recursive=True))
    before = set(any_fs._file_data)
    _, errors = await _collect(_rm(any_fs, pattern))
    assert errors == []
    assert set(any_fs._file_data) == before - expanded
    for path in expanded:
        assert not await any_fs._exists(path)


@pytest.mark.parametrize(
    "paths",
    [
        ["data/*/01", "data/2024/*"],
        ["data/README.md", "data/2023/**/*.parquet", "other"],
        ["data/2023", "data/2023/01"],  # One inside the other.
        ["data/*/13", "data/README.md"],  # One matching nothing.
    ],
)
async def test_a_list_deletes_what_fsspec_expands(
    any_fs: SimulatedFileSystem, paths: list[str]
) -> None:
    _tree(any_fs)
    expanded: set[str] = set()
    for path in paths:
        # Path by path: fsspec's expansion of a whole list raises if one pattern
        # in it matches nothing, as its recursion expands that empty match alone.
        with contextlib.suppress(FileNotFoundError):
            expanded |= set(
                await AsyncFileSystem._expand_path(any_fs, path, recursive=True)
            )
    before = set(any_fs._file_data)
    _, errors = await _collect(_rm(any_fs, paths))
    assert errors == []
    assert set(any_fs._file_data) == before - expanded
    for path in expanded:
        assert not await any_fs._exists(path)


async def test_a_list_matching_nothing_raises(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await _collect(_rm(any_fs, ["data/*/13", "missing"]))


async def test_a_pattern_walks_its_matches_at_once() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(8):
        fs.add_file(f"root/{i}/sub/file")
    await _collect(_rm(fs, "root/*", concurrency_limit=asyncio.Semaphore(8)))
    assert fs.stats.max_concurrency == 8
    # The glob's listing of root, then each match and its sub folder.
    assert fs.stats.calls["list"] == 1 + 8 + 8


async def test_a_pattern_matching_nothing_raises(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await _collect(_rm(any_fs, "data/*/13"))


async def test_missing_root_raises(any_fs: SimulatedFileSystem) -> None:
    op = _rm(any_fs, "missing")
    with pytest.raises(FileNotFoundError):
        await _collect(op)
    assert op.done() and not op.cancelled()


class _Locked(SimulatedFileSystem):
    """Refuses to delete one file, as a storage without permission would."""

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        if self._strip_protocol(path) == "data/2023/01/part-0.parquet":
            raise PermissionError(path)
        await super()._rm_file(path, **kwargs)


async def test_a_failure_keeps_its_folders_and_the_rest_goes() -> None:
    fs = _Locked(hierarchical=True, asynchronous=True)
    _tree(fs)
    deleted, errors = await _collect(_rm(fs, "data"))
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == ["Deleting 'data/2023/01/part-0.parquet' failed."]
    assert sum(p in _FILES for p in deleted) == 4
    # The folders above the failure are kept; the others are gone.
    assert await fs._exists("data/2023/01")
    assert not await fs._exists("data/2023/02")
    assert not await fs._exists("data/2024")


class _Forbidding(SimulatedFileSystem):
    """Refuses to list one folder, as a storage without permission would."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


async def test_a_failed_listing_keeps_its_folders_and_the_rest_goes() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    op = _rm(fs, "data")
    deleted, errors = await _collect(op)
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == ["Listing 'data/2023' failed."]
    assert op.state.errors == 1
    # data, 2024 and its 01 and 02 listed: the refusal sends nothing. Then
    # README.md, 2024's 2 files, and 01, 02 and 2024 deleted.
    assert fs.stats.calls == {"list": 4, "delete": 1 + 2 + 3}
    # Nothing under data/2023 is touched, and data stays above it.
    assert not any(p.startswith("data/2023") for p in deleted)
    assert await fs._exists("data/2023/01/part-0.parquet")
    assert not await fs._exists("data/2024")


class _Doubling(SimulatedFileSystem):
    """Lists each folder twice, as a prefix and as its marker object."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        entries = await super()._ls(path, detail=detail, **kwargs)
        return entries + [
            {**e, "name": e["name"] + "/"} for e in entries if e["type"] == "directory"
        ]


async def test_a_folder_listed_twice_is_deleted() -> None:
    fs = _Doubling(hierarchical=True, asynchronous=True)
    _tree(fs)
    deleted, errors = await _collect(_rm(fs, "data"))
    assert errors == []
    assert deleted[-1] == "data"
    assert not await fs._exists("data")


async def test_rm_folder_is_used_for_folders(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    removed = []

    async def rm_folder(path: str) -> None:
        removed.append(path)
        await hierarchical_fs._rmdir(path)

    await _collect(_rm(hierarchical_fs, "data", rm_folder=rm_folder))
    assert len(removed) == 7
    # A folder goes only after everything in it.
    assert removed[-1] == "data"
    assert removed.index("data/2023") > removed.index("data/2023/01")


async def test_deletes_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(40):
        fs.add_file(f"root/file-{i:02}")
    await _collect(_rm(fs, "root", concurrency_limit=asyncio.Semaphore(8)))
    assert fs.stats.calls == {"list": 1, "delete": 41}
    assert fs.stats.max_concurrency == 8


async def test_state_shows_progress(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    async with _rm(any_fs, "data") as op:
        states = [op.state async for _ in op]
    deleted = [s.files_deleted for s in states]
    assert deleted == sorted(deleted)
    assert op.state.files_found == op.state.files_deleted == 5


async def test_cancel_yields_the_deletions_sent() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/file-{i}")
    op = _rm(fs, "root", concurrency_limit=asyncio.Semaphore(10))
    deleted = []
    async for path in op:
        deleted.append(path)
        op.cancel()
    # All ten were sent together: each is still yielded, and none after.
    assert len(deleted) == fs.stats.calls["delete"] == 10
    assert await fs._exists("root")
    assert op.cancelled()


async def test_cancel_after_the_listing() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    fs.add_file("root/sub/file")
    fs.add_file("root/file")
    op = _rm(fs, "root", concurrency_limit=asyncio.Semaphore(10))
    async for path in op:
        op.cancel()
        assert path == "root/file"
    # The listing of root/sub, sent with the deletion, was waited for, but its
    # file was never deleted.
    assert fs.stats.calls == {"list": 2, "delete": 1}
    assert op.cancelled()


async def test_cancel_before_iterating_sends_nothing() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    fs.add_file("root/file")
    op = _rm(fs, "root")
    op.cancel()
    assert await _collect(op) == ([], [])
    assert fs.stats.calls == {}
    assert op.cancelled()


async def test_closing_waits_for_the_deletions_sent() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/file-{i}")
    async with _rm(fs, "root", concurrency_limit=asyncio.Semaphore(10)) as op:
        async for _ in op:
            break  # The first file: the nine others were sent with it.
    assert fs.stats.calls["delete"] == 10
    assert fs._in_flight == 0
    # Not yielded, but counted.
    assert op.state.files_deleted == 10
    assert await fs._exists("root")


async def test_cancelling_the_task_leaves_nothing_running() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    task = asyncio.ensure_future(
        _collect(_rm(fs, "root", concurrency_limit=asyncio.Semaphore(10)))
    )
    await asyncio.sleep(0.07)  # The root is listed; its subfolders are in flight.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fs._in_flight == 0


async def test_a_file_found_already_gone_is_not_yielded() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)

    async def rm_file(path: str) -> None:
        await fs._rm_file(path)
        if path == "data/README.md":
            # Deleted by another meanwhile, as the storage tells it.
            raise FileNotFoundError(path)

    op = _rm(fs, "data", rm_file=rm_file)
    deleted, errors = await _collect(op)
    assert errors == []
    assert "data/README.md" not in deleted
    assert op.state.files_deleted == 4
    # Its folder goes all the same.
    assert deleted[-1] == "data"
    assert not await fs._exists("data")
