"""Tests of `tree_rm`, the delete that walks and deletes at once."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fsspec_alt_strategies.control import OperationStopped, Stop
from fsspec_alt_strategies.rm import RmState, tree_rm
from fsspec_alt_strategies.testing import SimulatedFileSystem


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet")
    fs.add_file("data/README.md")
    fs.add_file("other/kept")


async def test_deletes_the_tree(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    result = await tree_rm(any_fs, "data")
    assert result.complete
    assert result.files_deleted == 5
    assert not await any_fs._exists("data")
    assert await any_fs._exists("other/kept")


async def test_requests_on_a_hierarchical_storage(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/empty")
    result = await tree_rm(hierarchical_fs, "data")
    assert (result.files_deleted, result.folders_deleted) == (5, 8)
    # A listing per folder, then a deletion per file and per folder.
    assert hierarchical_fs.stats.calls == {"list": 8, "delete": 13}


async def test_requests_on_an_object_store(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    await tree_rm(flat_fs, "data")
    # Object stores have no folders to delete: _rmdir sends nothing.
    assert flat_fs.stats.calls == {"list": 7, "delete": 5}


async def test_a_file(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    result = await tree_rm(any_fs, "data/README.md")
    assert (result.complete, result.files_deleted) == (True, 1)
    assert not await any_fs._exists("data/README.md")


async def test_missing_root_raises(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(FileNotFoundError):
        await tree_rm(any_fs, "missing")


class _Locked(SimulatedFileSystem):
    """Refuses to delete one file, as a storage without permission would."""

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        if self._strip_protocol(path) == "data/2023/01/part-0.parquet":
            raise PermissionError(path)
        await super()._rm_file(path, **kwargs)


async def test_a_failure_keeps_its_folders_and_the_rest_goes() -> None:
    fs = _Locked(hierarchical=True, asynchronous=True)
    _tree(fs)
    result = await tree_rm(fs, "data")
    assert not result.complete
    assert list(result.errors) == ["data/2023/01/part-0.parquet"]
    assert result.files_deleted == 4
    # The folders above the failure are kept; the others are gone.
    assert await fs._exists("data/2023/01")
    assert not await fs._exists("data/2023/02")
    assert not await fs._exists("data/2024")
    with pytest.raises(ExceptionGroup):
        result.raise_for_errors()


async def test_rm_folder_is_used_for_folders(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    removed = []

    async def rm_folder(path: str) -> None:
        removed.append(path)
        await hierarchical_fs._rmdir(path)

    await tree_rm(hierarchical_fs, "data", rm_folder=rm_folder)
    assert len(removed) == 7
    # A folder goes only after everything in it.
    assert removed[-1] == "data"
    assert removed.index("data/2023") > removed.index("data/2023/01")


async def test_deletes_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(40):
        fs.add_file(f"root/file-{i:02}")
    await tree_rm(fs, "root", batch_size=8)
    assert fs.stats.calls == {"list": 1, "delete": 41}
    assert fs.stats.max_concurrency == 8


async def test_callback_sees_progress(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    states: list[RmState] = []
    result = await tree_rm(any_fs, "data", callback=states.append)
    assert states[0].files_found == 1  # The README, in the first listing.
    assert states[-1].files_found == states[-1].files_deleted == 5
    deleted = [s.files_deleted for s in states]
    assert deleted == sorted(deleted)
    assert result.complete


async def test_stop_returns_the_partial_result(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)

    def after_two(state: RmState) -> Stop | None:
        return Stop(reason="enough") if state.files_deleted >= 2 else None

    result = await tree_rm(any_fs, "data", batch_size=1, callback=after_two)
    assert not result.complete
    assert result.stop == Stop(reason="enough")
    assert result.files_deleted == 2
    assert any_fs.stats.calls["delete"] == 2
    assert await any_fs._exists("data")


async def test_stop_waits_for_deletions_sent() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/file-{i}")

    def at_once(state: RmState) -> Stop | None:
        return Stop() if state.files_found else None

    result = await tree_rm(fs, "root", batch_size=10, callback=at_once)
    # Stopped after the listing, before any deletion was sent.
    assert (result.files_deleted, fs.stats.calls["delete"]) == (0, 0)

    def after_first(state: RmState) -> Stop | None:
        return Stop() if state.files_deleted else None

    result = await tree_rm(fs, "root", batch_size=10, callback=after_first)
    # All ten were sent together: each is waited for, and counted.
    assert result.files_deleted == fs.stats.calls["delete"] == 10
    assert fs._in_flight == 0


async def test_stop_can_raise(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(OperationStopped, match="fail fast") as info:
        await tree_rm(
            any_fs,
            "data",
            callback=lambda _: Stop(raise_error=True, reason="fail fast"),
        )
    assert info.value.result.files_deleted == 0


async def test_cancelling_the_task_leaves_nothing_running() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    task = asyncio.ensure_future(tree_rm(fs, "root", batch_size=10))
    await asyncio.sleep(0.07)  # The root is listed; its subfolders are in flight.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fs._in_flight == 0
