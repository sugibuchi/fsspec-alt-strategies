"""Tests of the ``rm_files`` strategy: each file by ``rm_file``, or a batch by one request."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.common.rm_files import _rm_by_request
from fsspec_alt_strategies_impl.flat.handler import FlatFileOperationHandler
from fsspec_alt_strategies_impl.flat.rm import _flat_rm
from fsspec_alt_strategies_impl.tree.handler import TreeFileOperationHandler
from fsspec_alt_strategies_impl.tree.rm import _tree_rm
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem, TreeView

_FILES = ["d/a", "d/b", "d/c", "d/sub/e"]


def _entries(*paths: str) -> list[dict[str, Any]]:
    return [{"name": p, "type": "file"} for p in paths]


@pytest.mark.parametrize(
    ("view", "default"),
    [
        (TreeView, TreeFileOperationHandler.rm_files),
        (FlatView, FlatFileOperationHandler.rm_files),
    ],
)
async def test_the_default_deletes_each_file_a_missing_one_too(
    view: Any, default: Any
) -> None:
    # Each family's default, as a handler not overriding it has: FlatView does.
    fs = SimulatedFileSystem(asynchronous=True)
    for path in _FILES:
        fs.add_file(path)
    deleted, errors = await _collect_with_errors(
        default(view(fs), _entries("d/a", "missing", "d/b"))
    )
    assert errors == []
    assert sorted(deleted) == ["d/a", "d/b", "missing"]
    assert not await fs._exists("d/a")
    assert fs.stats.calls["delete"] == 3


async def test_the_default_yields_each_failure_noting_its_path() -> None:
    fs = SimulatedFileSystem(asynchronous=True)
    fs.add_file("x")

    async def refusing(path: str) -> None:
        raise PermissionError(path)

    handler = TreeView(fs, rm_file=refusing)
    op = handler.rm_files(_entries("x"))
    deleted, errors = await _collect_with_errors(op)
    assert deleted == []
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == ["Deleting 'x' failed."]
    assert (op.state.files_found, op.state.files_deleted, op.state.errors) == (1, 0, 1)


async def test_the_default_runs_as_many_at_once_as_the_limit_lets_it() -> None:
    fs = SimulatedFileSystem(asynchronous=True)
    running = peak = 0

    async def slow(path: str) -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1

    handler = TreeView(fs, rm_file=slow, operation_limit=asyncio.Semaphore(3))
    await _collect_with_errors(
        handler.rm_files(_entries(*(f"f{i}" for i in range(10))))
    )
    assert peak == 3


async def test_cancelled_it_starts_no_new_deletion() -> None:
    fs = SimulatedFileSystem(asynchronous=True)
    tried: list[str] = []

    async def recording(path: str) -> None:
        tried.append(path)
        await asyncio.sleep(0.01)

    handler = TreeView(fs, rm_file=recording, operation_limit=asyncio.Semaphore(1))
    op = handler.rm_files(_entries(*(f"f{i}" for i in range(5))))
    async with op:
        async for _ in op:
            op.cancel()
    # The first, and at most the one that took its permit as it ended: no more.
    assert len(tried) <= 2
    assert op.cancelled()


async def test_a_request_deletes_a_batch_yielding_each_path_or_its_error() -> None:
    fs = SimulatedFileSystem(asynchronous=True)
    handler = TreeView(fs)
    sent: list[list[str]] = []

    async def request(entries: list[dict[str, Any]]) -> dict[str, Exception]:
        sent.append([e["name"] for e in entries])
        return {"b": PermissionError("b")}

    deleted, errors = await _collect_with_errors(
        _rm_by_request(handler, _entries("a", "b", "c"), request)
    )
    assert sent == [["a", "b", "c"]]
    assert deleted == ["a", "c"]
    assert errors[0].__notes__ == ["Deleting 'b' failed."]


async def test_a_request_failing_whole_ends_the_operation() -> None:
    async def request(entries: list[dict[str, Any]]) -> None:
        raise OSError("throttled")

    handler = TreeView(SimulatedFileSystem(asynchronous=True))
    with pytest.raises(OSError, match="throttled"):
        await _collect_with_errors(_rm_by_request(handler, _entries("a"), request))


@pytest.mark.parametrize("family", ["tree", "flat"])
async def test_a_recursive_rm_at_a_limit_of_1_runs_its_batches_without_deadlock(
    family: str,
) -> None:
    # A batch's job holds no permit: its rm_file takes the only one there is.
    fs = SimulatedFileSystem(hierarchical=family == "tree", asynchronous=True)
    for path in _FILES:
        fs.add_file(path)
    limit = asyncio.Semaphore(1)
    if family == "tree":
        op = _tree_rm(TreeView(fs, operation_limit=limit, rm_batch_size=2), "d")
    else:
        op = _flat_rm(
            FlatView(fs, operation_limit=limit, rm_batch_size=2, rm_files=None), "d"
        )
    deleted, errors = await asyncio.wait_for(_collect_with_errors(op), timeout=5)
    assert errors == []
    assert set(_FILES) <= set(deleted)
    assert not await fs._exists("d/a")
