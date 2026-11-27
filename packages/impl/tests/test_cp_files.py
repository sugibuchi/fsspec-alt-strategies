"""Tests of the batches of files the copies and moves within one storage send to ``cp_files`` and ``mv_files``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.cp_files import (
    _cp_each_file,
    cp_by_request,
    mv_by_request,
)
from fsspec_alt_strategies_impl.common.handler import FileTransfer
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.cp import _tree_cp
from fsspec_alt_strategies_impl.tree.mv import _tree_mv
from fsspec_alt_strategies_impl_testing import (
    FlatView,
    SimulatedFileSystem,
    TreeView,
)

_FILES = [
    "data/a/1",
    "data/a/2",
    "data/a/b/3",
    "data/a/b/4",
    "data/c/5",
    "data/c/d/6",
    "data/7",
]


def _tree(fs: SimulatedFileSystem) -> None:
    for path in _FILES:
        fs.add_file(path, path.encode())


class _Batching:
    """A view sending its batches by one request each, recorded, unless ``refuse`` says otherwise.

    ``refuse`` maps a source path to the error its batch answers for it, and
    ``fail`` fails each batch whole.
    """

    fs: SimulatedFileSystem

    def setup(self, size: int) -> None:
        self.cp_batch_size = self.mv_batch_size = size
        self.batches: list[tuple[str, list[str]]] = []
        self.refuse: dict[str, Exception] = {}
        self.fail: Exception | None = None

    async def request(self, kind: str, transfers: list[Any]) -> dict[str, Exception]:
        self.batches.append((kind, [t.entry["name"] for t in transfers]))
        if self.fail is not None:
            raise self.fail
        refused = {}
        for t in transfers:
            source = t.entry["name"]
            if source in self.refuse:
                refused[source] = self.refuse[source]
            elif kind == "copy":
                await self.fs._cp_file(source, t.path2)
            else:
                await self.fs._mv_file(source, t.path2)
        return refused

    def cp_files(self, transfers: list[Any]) -> Any:
        return cp_by_request(self, transfers, lambda ts: self.request("copy", ts))  # type: ignore[arg-type]

    def mv_files(self, transfers: list[Any]) -> Any:
        return mv_by_request(self, transfers, lambda ts: self.request("move", ts))  # type: ignore[arg-type]


class _TreeBatching(_Batching, TreeView):
    def __init__(self, fs: SimulatedFileSystem, size: int, **kwargs: Any) -> None:
        super().__init__(fs, **kwargs)
        self.setup(size)


class _FlatBatching(_Batching, FlatView):
    def __init__(self, fs: SimulatedFileSystem, size: int, **kwargs: Any) -> None:
        super().__init__(fs, **kwargs)
        self.setup(size)


async def _found(fs: SimulatedFileSystem, path: str) -> list[str]:
    return sorted(await fs._find(path))


@pytest.mark.parametrize("size", [1, 3, 100])
async def test_tree_copy_in_batches(any_fs: SimulatedFileSystem, size: int) -> None:
    _tree(any_fs)
    view = _TreeBatching(any_fs, size)
    copied, errors = await _collect_with_errors(_tree_cp(view, view, "data", "out"))
    assert errors == []
    assert sorted(s for s, _ in copied) == sorted(_FILES)
    assert await _found(any_fs, "out") == sorted(
        f.replace("data", "out", 1) for f in _FILES
    )
    # Each file once, each batch of files of any directory, at most size.
    assert sorted(p for _, paths in view.batches for p in paths) == sorted(_FILES)
    for _, paths in view.batches:
        assert len(paths) <= size
    assert {kind for kind, _ in view.batches} == {"copy"}


async def test_tree_move_in_batches_deletes_each_directory_emptied(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    view = _TreeBatching(any_fs, 3)
    moved, errors = await _collect_with_errors(_tree_mv(view, view, "data", "out"))
    assert errors == []
    assert await _found(any_fs, "out") == sorted(
        f.replace("data", "out", 1) for f in _FILES
    )
    assert not await any_fs._exists("data")
    moves = [p for k, p in view.batches if k == "move"]
    assert sorted(p for paths in moves for p in paths) == sorted(_FILES)
    assert all(len(paths) <= 3 for paths in moves)
    assert ("data", "out") in moved


async def test_a_file_a_batch_refuses_fails_alone() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    view = _TreeBatching(fs, 3)
    view.refuse["data/a/2"] = PermissionError("data/a/2")
    moved, errors = await _collect_with_errors(_tree_mv(view, view, "data", "out"))
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0] is view.refuse["data/a/2"]  # As the storage raised it.
    assert not getattr(errors[0], "__notes__", [])
    # It stays, and so do its directories, a and data: the rest moves.
    assert await fs._exists("data/a/2")
    assert sorted(s for s, _ in moved if s in _FILES) == sorted(
        set(_FILES) - {"data/a/2"}
    )
    assert not await fs._exists("data/c")


async def test_a_copy_refused_onto_an_existing_file_is_replaced_alone() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    # out exists, so data goes into it, as fsspec's copy places it.
    fs.add_file("out/data/a/1", b"old")
    view = _TreeBatching(fs, 100)
    view.refuse["data/a/1"] = FileExistsError("out/data/a/1")
    copied, errors = await _collect_with_errors(_tree_cp(view, view, "data", "out"))
    # Replaced by the source's cp_file, once deleted: as without batches.
    assert errors == []
    assert await fs._cat_file("out/data/a/1") == b"data/a/1"
    assert len(copied) == len(_FILES)


async def test_a_batch_failing_whole_is_one_error() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    view = _TreeBatching(fs, 100)
    view.fail = ConnectionResetError("commit")
    copied, errors = await _collect_with_errors(_tree_cp(view, view, "data", "out"))
    assert copied == []
    # One error per batch, each holding files of any directory.
    assert errors and {type(e) for e in errors} == {ConnectionResetError}


async def test_no_batch_between_two_storages_or_with_the_callers_copy() -> None:
    src = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    dst = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(src)
    source, destination = _TreeBatching(src, 3), _TreeBatching(dst, 3)
    _, errors = await _collect_with_errors(_tree_cp(source, destination, "data", "out"))
    assert errors == []
    assert source.batches == destination.batches == []
    assert await _found(dst, "out") == sorted(
        f.replace("data", "out", 1) for f in _FILES
    )

    async def cp_entry(
        entry: dict[str, Any], path2: str, *, missing: bool = False, parent: Any = None
    ) -> None:
        await src._cp_file(entry["name"], path2)

    _, errors = await _collect_with_errors(
        _tree_cp(source, source, "data", "again", cp_entry=cp_entry)
    )
    assert errors == []
    assert source.batches == []


async def test_the_default_copies_each_file_by_cp_file() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    view = TreeView(fs, operation_limit=asyncio.Semaphore(2))
    entries = [await view.info(view.entry(p)) for p in _FILES[:3]]
    transfers = [
        FileTransfer(e, view.verified(e["name"].replace("data", "x", 1)), True)
        for e in entries
    ]
    copied, errors = await _collect_with_errors(_cp_each_file(view, transfers))
    assert errors == []
    assert sorted(copied) == [(p, p.replace("data", "x", 1)) for p in _FILES[:3]]
    assert fs.stats.max_concurrency <= 2


@pytest.mark.parametrize("renames", [True, False])
async def test_flat_move_in_batches(
    flat_fs: SimulatedFileSystem, renames: bool
) -> None:
    _tree(flat_fs)
    view = _FlatBatching(
        flat_fs,
        3,
        storage_profile=StorageProfile(renames_files=renames),
        rm_batch_size=1000,
    )
    _, errors = await _collect_with_errors(_tree_mv(view, view, "data", "out"))
    assert errors == []
    assert await _found(flat_fs, "out") == sorted(
        f.replace("data", "out", 1) for f in _FILES
    )
    assert not await flat_fs._exists("data/a/1")
    # By renames, or by copies then deletions: either way in batches of 3.
    kind = "move" if renames else "copy"
    assert [(k, len(p)) for k, p in view.batches] == [(kind, 3), (kind, 3), (kind, 1)]


async def test_flat_copy_in_batches(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    view = _FlatBatching(flat_fs, 4)
    copied, errors = await _collect_with_errors(_tree_cp(view, view, "data", "out"))
    assert errors == []
    assert len(copied) == len(_FILES)
    assert [len(p) for _, p in view.batches] == [4, 3]


def test_a_batch_size_below_1_is_refused() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    view = _TreeBatching(fs, 0)
    with pytest.raises(ValueError, match="cp_batch_size must be at least 1"):
        asyncio.run(_collect_with_errors(_tree_cp(view, view, "data", "out")))


async def test_cancel_drops_the_batches_not_sent() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    for i in range(20):
        fs.add_file(f"data/{i:02}/f")
    view = _TreeBatching(fs, 2)
    # Two directories at a time: those not started send nothing once cancelled.
    op = _tree_cp(view, view, "data", "out", max_running=2)
    copied = []
    async for outcome in op:
        copied.append(outcome)
        op.cancel()
    # What was sent is yielded: the batches not sent are dropped, unsent.
    assert op.cancelled()
    assert len(copied) == sum(len(p) for _, p in view.batches) < 20
    assert len(await fs._find("out")) == len(copied)


async def test_a_cancelled_move_by_copies_leaves_each_key_in_one_place(
    flat_fs: SimulatedFileSystem,
) -> None:
    for i in range(20):
        flat_fs.add_file(f"data/{i:02}")
    view = _FlatBatching(
        flat_fs, 2, storage_profile=StorageProfile(renames_files=False)
    )
    op = _tree_mv(view, view, "data", "out")
    async for _ in op:
        op.cancel()
    # The keys copied are deleted all the same: none is in both places.
    for i in range(20):
        here = await flat_fs._exists(f"data/{i:02}")
        there = await flat_fs._exists(f"out/{i:02}")
        assert here != there
