"""Tests of `rm` and `irm`, the delete that walks and deletes at once."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import Any

import pytest

from fsspec_alt_strategies.control import OperationStopped, Stop
from fsspec_alt_strategies.rm import RmState, irm, rm
from fsspec_alt_strategies.testing import SimulatedFileSystem

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
    deleted, errors = await rm(any_fs, "data")
    assert errors == []
    assert set(_FILES) <= set(deleted)
    assert not await any_fs._exists("data")
    assert await any_fs._exists("other/kept")


async def test_requests_on_a_hierarchical_storage(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/empty")
    deleted, _ = await rm(hierarchical_fs, "data")
    # The 5 files, then the 8 folders, data included.
    assert len(deleted) == 13
    assert deleted[-1] == "data"
    # A listing per folder, then a deletion per file and per folder.
    assert hierarchical_fs.stats.calls == {"list": 8, "delete": 13}


async def test_requests_on_an_object_store(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    deleted, _ = await rm(flat_fs, "data")
    # Object stores have no folders to delete: _rmdir sends nothing.
    assert len(deleted) == 5 + 7
    assert flat_fs.stats.calls == {"list": 7, "delete": 5}


async def test_a_file(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    assert await rm(any_fs, "data/README.md") == (["data/README.md"], [])
    assert not await any_fs._exists("data/README.md")


async def test_missing_root_raises(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(FileNotFoundError):
        await rm(any_fs, "missing")
    with pytest.raises(FileNotFoundError):
        async for _ in irm(any_fs, "missing"):
            pass


class _Locked(SimulatedFileSystem):
    """Refuses to delete one file, as a storage without permission would."""

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        if self._strip_protocol(path) == "data/2023/01/part-0.parquet":
            raise PermissionError(path)
        await super()._rm_file(path, **kwargs)


async def test_a_failure_keeps_its_folders_and_the_rest_goes() -> None:
    fs = _Locked(hierarchical=True, asynchronous=True)
    _tree(fs)
    deleted, errors = await rm(fs, "data")
    assert [type(e) for e in errors] == [PermissionError]
    assert sum(p in _FILES for p in deleted) == 4
    # The folders above the failure are kept; the others are gone.
    assert await fs._exists("data/2023/01")
    assert not await fs._exists("data/2023/02")
    assert not await fs._exists("data/2024")


async def test_rm_folder_is_used_for_folders(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    removed = []

    async def rm_folder(path: str) -> None:
        removed.append(path)
        await hierarchical_fs._rmdir(path)

    await rm(hierarchical_fs, "data", rm_folder=rm_folder)
    assert len(removed) == 7
    # A folder goes only after everything in it.
    assert removed[-1] == "data"
    assert removed.index("data/2023") > removed.index("data/2023/01")


async def test_deletes_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(40):
        fs.add_file(f"root/file-{i:02}")
    await rm(fs, "root", batch_size=8)
    assert fs.stats.calls == {"list": 1, "delete": 41}
    assert fs.stats.max_concurrency == 8


async def test_callback_sees_progress(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    states: list[RmState] = []
    _, errors = await rm(any_fs, "data", callback=states.append)
    assert states[0].files_found == 1  # The README, in the first listing.
    assert states[-1].files_found == states[-1].files_deleted == 5
    deleted = [s.files_deleted for s in states]
    assert deleted == sorted(deleted)
    assert errors == []


async def test_stop_returns_the_partial_result(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)

    def after_two(state: RmState) -> Stop | None:
        return Stop(reason="enough") if state.files_deleted >= 2 else None

    deleted, _ = await rm(any_fs, "data", batch_size=1, callback=after_two)
    assert len(deleted) == 2
    assert any_fs.stats.calls["delete"] == 2
    assert await any_fs._exists("data")


async def test_stop_waits_for_deletions_sent() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/file-{i}")

    def at_once(state: RmState) -> Stop | None:
        return Stop() if state.files_found else None

    deleted, _ = await rm(fs, "root", batch_size=10, callback=at_once)
    # Stopped after the listing, before any deletion was sent.
    assert (deleted, fs.stats.calls["delete"]) == ([], 0)

    def after_first(state: RmState) -> Stop | None:
        return Stop() if state.files_deleted else None

    deleted, _ = await rm(fs, "root", batch_size=10, callback=after_first)
    # All ten were sent together: each is waited for, and counted.
    assert len(deleted) == fs.stats.calls["delete"] == 10
    assert fs._in_flight == 0


async def test_stop_can_raise(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(OperationStopped, match="fail fast") as info:
        await rm(
            any_fs,
            "data",
            callback=lambda _: Stop(raise_error=True, reason="fail fast"),
        )
    assert info.value.result == ([], [])


async def test_irm_yields_each_path_and_error() -> None:
    fs = _Locked(hierarchical=True, asynchronous=True)
    _tree(fs)
    outcomes = [o async for o in irm(fs, "data")]
    assert [type(o) for o in outcomes if isinstance(o, Exception)] == [PermissionError]
    assert sum(o in _FILES for o in outcomes) == 4


async def test_breaking_out_of_irm_waits_for_deletions_sent() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/file-{i}")
    async with aclosing(irm(fs, "root", batch_size=10)) as deleted:
        async for _ in deleted:
            break  # The first file: the nine others were sent with it.
    assert fs.stats.calls["delete"] == 10
    assert fs._in_flight == 0
    assert await fs._exists("root")


async def test_cancelling_the_task_leaves_nothing_running() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    task = asyncio.ensure_future(rm(fs, "root", batch_size=10))
    await asyncio.sleep(0.07)  # The root is listed; its subfolders are in flight.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fs._in_flight == 0
