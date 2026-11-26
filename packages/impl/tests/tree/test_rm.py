"""Tests of `_tree_rm`, the delete that walks and deletes at once."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies.states import RmState
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.rm import _tree_rm
from fsspec_alt_strategies_impl_testing import (
    SimulatedFileSystem,
    TreeView,
)

_PRIMITIVES = {"ils", "info", "rm_file", "rm_dir", "rm_tree", "cp_file", "mv_file"}
"""The keyword arguments that replace a primitive of the filesystem."""

_SETTINGS = {"profile": "storage_profile", "concurrency_limit": "operation_limit"}
"""The keyword arguments that set the filesystem's profile and limit, by its attribute."""


def _rm(fs: SimulatedFileSystem, path: Any, **kwargs: Any) -> Any:
    """`_tree_rm` on a filesystem's own primitives, which ``kwargs`` may replace."""
    view = TreeView(
        fs,
        **{k: kwargs.pop(k) for k in _PRIMITIVES & kwargs.keys()},
        **{_SETTINGS[k]: kwargs.pop(k) for k in _SETTINGS.keys() & kwargs.keys()},
    )
    return _tree_rm(view, path, **kwargs)


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
    deleted, errors = await _collect_with_errors(_rm(any_fs, "data"))
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
    deleted, _ = await _collect_with_errors(op)
    # The 5 files, then the 8 directories, data included.
    assert len(deleted) == 13
    assert deleted[-1] == "data"
    assert op.state == RmState(
        current="data", files_found=5, files_deleted=5, directories_deleted=8
    )
    # A listing per directory, then a deletion per file and per directory.
    assert hierarchical_fs.stats.calls == {"list": 8, "delete": 13}


async def test_requests_on_an_object_store(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    deleted, _ = await _collect_with_errors(_rm(flat_fs, "data"))
    # Object stores have no directories to delete: _rmdir sends nothing.
    assert len(deleted) == 5 + 7
    assert flat_fs.stats.calls == {"list": 7, "delete": 5}


async def test_implicit_dirs_save_the_dir_deletions(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    directories: list[str] = []

    async def rm_dir(path: str) -> None:
        directories.append(path)  # A request a real object store may send.
        await flat_fs._rmdir(path)

    profile = StorageProfile(implicit_directories=True)
    op = _rm(flat_fs, "data", rm_dir=rm_dir, profile=profile)
    deleted, errors = await _collect_with_errors(op)
    # The root alone, which may have a marker: those below have none listed.
    assert (directories, errors) == (["data"], [])
    # The others, gone with their last file: not yielded.
    assert sorted(deleted) == sorted([*_FILES, "data"])
    assert op.state.directories_deleted == 1
    assert not await flat_fs._exists("data")
    assert await flat_fs._exists("other/kept")


async def test_a_file(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    assert await _collect_with_errors(_rm(any_fs, "data/README.md")) == (
        ["data/README.md"],
        [],
    )
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
    _, errors = await _collect_with_errors(_rm(any_fs, pattern))
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
    _, errors = await _collect_with_errors(_rm(any_fs, paths))
    assert errors == []
    assert set(any_fs._file_data) == before - expanded
    for path in expanded:
        assert not await any_fs._exists(path)


async def test_a_list_matching_nothing_raises(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(_rm(any_fs, ["data/*/13", "missing"]))


async def test_a_pattern_walks_its_matches_at_once() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(8):
        fs.add_file(f"root/{i}/sub/file")
    await _collect_with_errors(
        _rm(fs, "root/*", concurrency_limit=asyncio.Semaphore(8))
    )
    assert fs.stats.max_concurrency == 8
    # The glob's listing of root, then each match and its sub directory.
    assert fs.stats.calls["list"] == 1 + 8 + 8


async def test_a_pattern_matching_nothing_raises(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(_rm(any_fs, "data/*/13"))


async def test_missing_root_raises(any_fs: SimulatedFileSystem) -> None:
    op = _rm(any_fs, "missing")
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(op)
    assert op.done() and not op.cancelled()


class _Locked(SimulatedFileSystem):
    """Refuses to delete one file, as a storage without permission would."""

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        if self._strip_protocol(path) == "data/2023/01/part-0.parquet":
            raise PermissionError(path)
        await super()._rm_file(path, **kwargs)


async def test_a_failure_keeps_its_dirs_and_the_rest_goes() -> None:
    fs = _Locked(hierarchical=True, asynchronous=True)
    _tree(fs)
    deleted, errors = await _collect_with_errors(_rm(fs, "data"))
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == ["Deleting 'data/2023/01/part-0.parquet' failed."]
    assert sum(p in _FILES for p in deleted) == 4
    # The directories above the failure are kept; the others are gone.
    assert await fs._exists("data/2023/01")
    assert not await fs._exists("data/2023/02")
    assert not await fs._exists("data/2024")


class _Forbidding(SimulatedFileSystem):
    """Refuses to list one directory, as a storage without permission would."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


async def test_a_failed_listing_keeps_its_dirs_and_the_rest_goes() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    # With a maxdepth, each directory is listed on its own: the rest goes on.
    op = _rm(fs, "data", maxdepth=3)
    deleted, errors = await _collect_with_errors(op)
    assert [type(e) for e in errors] == [PermissionError]
    assert op.state.errors == 1
    # data, 2024 and its 01 and 02 listed: the refusal sends nothing. Then
    # README.md, 2024's 2 files, and 01, 02 and 2024 deleted.
    assert fs.stats.calls == {"list": 4, "delete": 1 + 2 + 3}
    # Nothing under data/2023 is touched, and data stays above it.
    assert not any(p.startswith("data/2023") for p in deleted)
    assert await fs._exists("data/2023/01/part-0.parquet")
    assert not await fs._exists("data/2024")


async def test_a_failed_listing_in_the_recursive_listing_is_resumed_past() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    op = _rm(fs, "data")
    deleted, errors = await _collect_with_errors(op)
    # The listing breaks at data/2023, and is resumed a directory per request:
    # data/2023 fails again, and the rest is deleted, as with a maxdepth.
    assert [type(e) for e in errors] == [PermissionError]
    assert not any(p.startswith("data/2023") for p in deleted)
    assert await fs._exists("data/2023/01/part-0.parquet")
    assert not await fs._exists("data/2024")
    assert not await fs._exists("data/README.md")


class _BreakingOff(SimulatedFileSystem):
    """Breaks its recursive listing off after ``cut_after`` entries, as a dropped connection does."""

    cut_after = 0

    async def _ils_recursive(self, path: str) -> Any:
        count = 0
        async for entry in super()._ils_recursive(path):
            if count == self.cut_after:
                raise ConnectionResetError(path)
            count += 1
            yield entry


@pytest.mark.parametrize("cut_after", [1, 2, 3, 5, 8])
async def test_a_recursive_listing_broken_off_keeps_what_it_did_not_list(
    any_fs: SimulatedFileSystem, cut_after: int
) -> None:
    fs = _BreakingOff(hierarchical=any_fs.hierarchical, asynchronous=True)
    fs.cut_after = cut_after
    _tree(fs)
    view = TreeView(fs, recursive_listing=True)
    try:
        deleted, errors = await _collect_with_errors(_tree_rm(view, "data"))
    except ConnectionResetError:
        assert await fs._exists("data/README.md")  # Nothing listed, nothing deleted.
        return
    assert await fs._exists("other/kept")
    if not errors:
        assert not await fs._exists(
            "data"
        )  # A listing shorter than the cut ends whole.
    else:
        # A failed listing: its error, and the root kept, with what it did not list.
        assert [type(e) for e in errors] == [ConnectionResetError]
        assert "data" not in deleted
        assert await fs._exists("data")


class _Doubling(SimulatedFileSystem):
    """Lists each directory twice, as a prefix and as its marker object."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        entries = await super()._ls(path, detail=detail, **kwargs)
        return entries + [
            {**e, "name": e["name"] + "/"} for e in entries if e["type"] == "directory"
        ]


async def test_a_dir_listed_twice_is_deleted() -> None:
    fs = _Doubling(hierarchical=True, asynchronous=True)
    _tree(fs)
    deleted, errors = await _collect_with_errors(_rm(fs, "data"))
    assert errors == []
    assert deleted[-1] == "data"
    assert not await fs._exists("data")


async def test_rm_dir_is_used_for_dirs(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    removed = []

    async def rm_dir(path: str) -> None:
        removed.append(path)
        await hierarchical_fs._rmdir(path)

    await _collect_with_errors(_rm(hierarchical_fs, "data", rm_dir=rm_dir))
    assert len(removed) == 7
    # A directory goes only after everything in it.
    assert removed[-1] == "data"
    assert removed.index("data/2023") > removed.index("data/2023/01")


async def test_deletes_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(40):
        fs.add_file(f"root/file-{i:02}")
    await _collect_with_errors(_rm(fs, "root", concurrency_limit=asyncio.Semaphore(8)))
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

    async def ils(path: str, **kwargs: Any) -> Any:
        # Still listing once root/file is deleted, whatever the loop's order of
        # two requests of equal latency: its file's deletion would be in flight.
        if path == "root/sub":
            await asyncio.sleep(0.05)
        for entry in await fs._ls(path, detail=True, **kwargs):
            yield entry

    # With a maxdepth, each directory is listed on its own.
    op = _rm(fs, "root", maxdepth=2, ils=ils, concurrency_limit=asyncio.Semaphore(10))
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
    assert await _collect_with_errors(op) == ([], [])
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
        _collect_with_errors(_rm(fs, "root", concurrency_limit=asyncio.Semaphore(10)))
    )
    await asyncio.sleep(0.07)  # The root is listed; its subdirectories are in flight.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fs._in_flight == 0


async def test_a_file_found_already_gone_counts_as_deleted() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)

    async def rm_file(path: str) -> None:
        await fs._rm_file(path)
        if path == "data/README.md":
            # Deleted by another meanwhile, as the storage tells it.
            raise FileNotFoundError(path)

    op = _rm(fs, "data", rm_file=rm_file)
    deleted, errors = await _collect_with_errors(op)
    assert errors == []
    # Gone is deleted, as rm_files takes it: no error, and yielded.
    assert "data/README.md" in deleted
    assert op.state.files_deleted == 5
    # Its directory goes all the same.
    assert deleted[-1] == "data"
    assert not await fs._exists("data")


# A storage declared to delete a directory whole, by its rm_tree.
_DELETES = StorageProfile(deletes_directories=True)


def _rm_tree(fs: SimulatedFileSystem, seen: list[str] | None = None) -> Any:
    """The storage's recursive delete, one request a directory, recording its paths."""

    async def rm_tree(path: str) -> None:
        if seen is not None:
            seen.append(path)
        await fs._rm(path, recursive=True)

    return rm_tree


async def test_rm_tree_deletes_a_dir_whole(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.reset_stats()
    op = _rm(
        hierarchical_fs, "data", rm_tree=_rm_tree(hierarchical_fs), profile=_DELETES
    )
    deleted, errors = await _collect_with_errors(op)
    assert (deleted, errors) == (["data"], [])
    # One deletion: no listing.
    assert hierarchical_fs.stats.calls == {"delete": 1}
    assert op.state.directories_deleted == 1
    assert not await hierarchical_fs._exists("data")
    assert await hierarchical_fs._exists("other/kept")


async def test_rm_tree_takes_any_root(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    seen: list[str] = []
    hierarchical_fs.reset_stats()
    rm_tree = _rm_tree(hierarchical_fs, seen)
    deleted, errors = await _collect_with_errors(
        _rm(hierarchical_fs, "data/README.md", rm_tree=rm_tree, profile=_DELETES)
    )
    # A file, of unknown kind until deleted: no request asks it first.
    assert (deleted, errors, seen) == (["data/README.md"], [], ["data/README.md"])
    assert hierarchical_fs.stats.calls == {"delete": 1}
    # A pattern's files are deleted as files, its directories whole.
    seen.clear()
    hierarchical_fs.add_file("data/notes.md")
    hierarchical_fs.reset_stats()
    deleted, errors = await _collect_with_errors(
        _rm(hierarchical_fs, "data/*", rm_tree=rm_tree, profile=_DELETES)
    )
    assert errors == []
    assert sorted(seen) == ["data/2023", "data/2024"]
    assert sorted(deleted) == ["data/2023", "data/2024", "data/notes.md"]
    assert hierarchical_fs.stats.calls["delete"] == 3
    assert "info" not in hierarchical_fs.stats.calls


async def test_a_failed_rm_tree_is_the_roots_error(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)

    async def refused(path: str) -> None:
        raise PermissionError(path)

    hierarchical_fs.reset_stats()
    op = _rm(hierarchical_fs, "data", rm_tree=refused, profile=_DELETES)
    deleted, errors = await _collect_with_errors(op)
    # Its cause, not one failure per file: nothing is walked.
    assert deleted == []
    assert [type(e) for e in errors] == [PermissionError]
    assert hierarchical_fs.stats.calls == {}
    assert op.state.errors == 1
    assert await hierarchical_fs._exists("data/README.md")


async def test_a_missing_root_raises_as_without_rm_tree(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    rm_tree = _rm_tree(hierarchical_fs)
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(
            _rm(hierarchical_fs, "missing", rm_tree=rm_tree, profile=_DELETES)
        )


async def test_rm_tree_is_unused_unless_declared(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    seen: list[str] = []
    op = _rm(hierarchical_fs, "data", rm_tree=_rm_tree(hierarchical_fs, seen))
    deleted, errors = await _collect_with_errors(op)
    assert (seen, errors) == ([], [])
    assert set(_FILES) <= set(deleted)


async def test_rm_tree_sends_nothing_once_cancelled(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.reset_stats()
    op = _rm(
        hierarchical_fs, "data", rm_tree=_rm_tree(hierarchical_fs), profile=_DELETES
    )
    op.cancel()
    assert await _collect_with_errors(op) == ([], [])
    assert hierarchical_fs.stats.calls == {}


async def test_rm_tree_starts_no_root_once_cancelled_midway(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    seen: list[str] = []
    roots = ["data/2023/01", "data/2023/02", "data/2024/01", "data/2024/02"]
    op = _rm(
        hierarchical_fs,
        roots,
        rm_tree=_rm_tree(hierarchical_fs, seen),
        profile=_DELETES,
        concurrency_limit=asyncio.Semaphore(1),
    )
    async for _ in op:
        op.cancel()  # After the first batch: no batch more is read.
    # The roots are planned together, and the batch read holds those
    # planned by then: each goes, the rest never.
    assert 1 <= len(seen) <= len(roots)
    assert op.state.directories_deleted == len(seen)


async def _fsspec_deletes(
    fs: SimulatedFileSystem, path: Any, maxdepth: int
) -> set[str]:
    """The files fsspec's ``_rm`` deletes with ``maxdepth``: those it expands to."""
    expanded = await AsyncFileSystem._expand_path(
        fs, path, recursive=True, maxdepth=maxdepth
    )
    fs.reset_stats()
    return {p for p in expanded if p in fs._file_data}


@pytest.mark.parametrize("maxdepth", [1, 2, 3])
@pytest.mark.parametrize(
    "path", ["data", "data/*", "data/20*/*", ["data/2023", "data/README.md"]]
)
async def test_maxdepth_deletes_the_files_fsspecs_rm_does(
    any_fs: SimulatedFileSystem, path: Any, maxdepth: int
) -> None:
    _tree(any_fs)
    expected = set(any_fs._file_data) - await _fsspec_deletes(any_fs, path, maxdepth)
    # A directory at the limit with files in it fails to be deleted, as fsspec's
    # deletion of it does: the files are what to compare.
    await _collect_with_errors(_rm(any_fs, path, maxdepth=maxdepth))
    assert set(any_fs._file_data) == expected


async def test_maxdepth_deletes_the_dirs_at_the_limit_if_empty(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/2024/empty")
    seen: list[str] = []

    async def rm_tree(path: str) -> None:
        seen.append(path)

    op = _rm(hierarchical_fs, "data", maxdepth=2, rm_tree=rm_tree, profile=_DELETES)
    deleted, errors = await _collect_with_errors(op)
    # rm_tree would delete what is deeper too: it is never sent.
    assert seen == []
    assert sorted(deleted) == ["data/2024/empty", "data/README.md"]
    # The four directories at the limit holding files, and data above them, stay.
    assert sorted(e.filename for e in errors) == [
        "data/2023/01",
        "data/2023/02",
        "data/2024/01",
        "data/2024/02",
    ]
    # data, then its two directories, are listed: those at the limit are not.
    assert hierarchical_fs.stats.calls == {"list": 3, "delete": 6}
    assert await hierarchical_fs._exists("data/2023/01/part-0.parquet")
    assert op.state == RmState(
        errors=4,
        current=op.state.current,
        files_found=1,
        files_deleted=1,
        directories_deleted=1,
    )


async def test_maxdepth_deep_enough_deletes_everything(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    deleted, errors = await _collect_with_errors(_rm(any_fs, "data", maxdepth=3))
    assert errors == []
    assert set(_FILES) <= set(deleted)
    assert not await any_fs._exists("data")


async def test_maxdepth_below_1_is_refused(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="maxdepth"):
        _rm(any_fs, "data", maxdepth=0)


def _batching(fs: SimulatedFileSystem, size: int, rm_files: Any = None) -> TreeView:
    """A view deleting files in batches of ``size``, by the filesystem's own, or ``rm_files``."""
    return TreeView(fs, rm_batch_size=size, rm_files=rm_files)


async def test_a_handler_with_a_batch_size_deletes_files_in_batches(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/empty")
    batches: list[list[str]] = []

    async def rm_files(paths: list[str]) -> None:
        batches.append(sorted(paths))
        await hierarchical_fs._rm_files(paths)

    op = _tree_rm(_batching(hierarchical_fs, 2, rm_files), "data")
    deleted, errors = await _collect_with_errors(op)
    assert errors == []
    # The 5 files in batches of at most 2, each file once, then the 8
    # directories. A batch holds files of any directory, as they are read.
    assert sorted(p for b in batches for p in b) == sorted(_FILES)
    assert all(len(b) <= 2 for b in batches) and len(batches) >= 3
    assert set(deleted) == {
        *_FILES,
        "data",
        "data/empty",
        "data/2023",
        "data/2024",
        "data/2023/01",
        "data/2023/02",
        "data/2024/01",
        "data/2024/02",
    }
    assert deleted[-1] == "data"
    assert not await hierarchical_fs._exists("data")
    assert op.state == RmState(
        current="data", files_found=5, files_deleted=5, directories_deleted=8
    )


async def test_a_file_its_batch_refused_keeps_its_directory(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    refused = "data/2023/01/part-0.parquet"

    async def rm_files(paths: list[str]) -> dict[str, Exception]:
        await hierarchical_fs._rm_files([p for p in paths if p != refused])
        return {refused: PermissionError(refused)} if refused in paths else {}

    deleted, errors = await _collect_with_errors(
        _tree_rm(_batching(hierarchical_fs, 10, rm_files), "data")
    )
    assert [type(e) for e in errors] == [PermissionError]
    assert f"Deleting {refused!r} failed." in errors[0].__notes__
    # The file, and the directories above it, stay: the rest goes.
    assert refused not in deleted
    assert {"data/2023/01", "data/2023", "data"}.isdisjoint(deleted)
    assert await hierarchical_fs._exists(refused)
    assert not await hierarchical_fs._exists("data/2024")


async def test_a_batch_failing_whole_keeps_its_files_and_directories(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)

    async def rm_files(paths: list[str]) -> None:
        raise OSError("throttled")

    deleted, errors = await _collect_with_errors(
        _tree_rm(_batching(hierarchical_fs, 10, rm_files), "data")
    )
    # Each batch fails whole, with one error.
    assert errors and {str(e) for e in errors} == {"throttled"}
    assert deleted == []
    assert all([await hierarchical_fs._exists(p) for p in _FILES])


async def test_a_batch_size_below_1_is_refused(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    # Read once the handler is open, as the operation starts.
    with pytest.raises(ValueError, match="rm_batch_size"):
        await _collect_with_errors(_tree_rm(_batching(hierarchical_fs, 0), "data"))


async def test_files_matched_or_listed_are_batched_too(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    batches: list[list[str]] = []

    async def rm_files(paths: list[str]) -> None:
        batches.append(sorted(paths))
        await hierarchical_fs._rm_files(paths)

    files = ["data/README.md", "data/2023/01/part-0.parquet"]
    op = _tree_rm(_batching(hierarchical_fs, 10, rm_files), files)
    deleted, errors = await _collect_with_errors(op)
    assert errors == []
    # One request for both: e.g. one commit of the Hub, rather than one per file.
    assert batches == [sorted(files)]
    assert set(files) <= set(deleted)
    assert op.state.files_found == op.state.files_deleted == 2


async def test_a_single_file_is_deleted_by_a_batch(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    batches: list[list[str]] = []

    async def rm_files(paths: list[str]) -> None:
        batches.append(paths)
        await hierarchical_fs._rm_files(paths)

    path = "data/README.md"
    deleted, errors = await _collect_with_errors(
        _tree_rm(_batching(hierarchical_fs, 10, rm_files), path)
    )
    assert (deleted, errors, batches) == ([path], [], [[path]])
    assert not await hierarchical_fs._exists(path)


async def test_a_directory_given_with_a_slash_goes_whole_by_its_name(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    seen: list[str] = []
    profile = StorageProfile(deletes_directories=True)
    deleted, errors = await _collect_with_errors(
        _rm(
            hierarchical_fs,
            "data/",
            rm_tree=_rm_tree(hierarchical_fs, seen),
            profile=profile,
        )
    )
    # rm_tree is given the directory's path, and it is yielded so: no "data/".
    assert errors == []
    assert seen == ["data"]
    assert deleted == ["data"]
