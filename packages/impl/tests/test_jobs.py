"""Tests of `common.jobs`: the permits of `_Queue`, the steps of `_Jobs`, and `_CopyJobs`'s placing.

The strategies test `_Jobs` through what they do to a storage. These test what
none of them shows: each permit given back, however the operation ends. The
copies of both families test the placing of fsspec's cases: these test the
edges of a pattern that none of them reaches.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies.monitor import State
from fsspec_alt_strategies.states import CpState
from fsspec_alt_strategies_impl.common.jobs import (
    _CopyJobs,
    _Job,
    _Jobs,
    _Queue,
    _Roots,
    _roots_of,
)
from fsspec_alt_strategies_impl.common.operation import Operation, _collect_with_errors

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class _Toy(_Jobs):
    """Jobs of one item each, fed by a source of item lists, e.g. listings.

    A job sleeps ``delay``, and fails for an item in ``failing``.
    """

    def __init__(
        self,
        limit: asyncio.Semaphore,
        op: Operation[Any, Any],
        batches: list[list[str]],
        failing: frozenset[str] = frozenset(),
        delay: float = 0.01,
    ) -> None:
        super().__init__(fs=SimpleNamespace(operation_limit=limit), op=op)
        self.batches = batches
        self.failing = failing
        self.delay = delay
        self.started: list[str] = []
        self.finished: list[str] = []
        self.exhausted_calls = 0
        self.most = 0
        self.at_once = 0

    async def prepare(self) -> _Roots | None:
        return _Roots(["root"])

    def glob(self, pattern: Any) -> Any:
        raise NotImplementedError  # The toy takes no pattern.

    def state(self) -> State:
        return State(errors=self.errors, current=self.current)

    async def source(self, roots: _Roots) -> AsyncGenerator[list[str]]:
        for batch in self.batches:
            await asyncio.sleep(0)
            yield batch

    def fed(self, item: list[str]) -> list[Any]:
        self.waiting.extend(_Job("do", {"name": name}) for name in item)
        return []

    def file_root(self, path: str) -> None:
        self.waiting.append(_Job("do", {"name": path}))

    async def run(self, job: _Job) -> None:
        self.started.append(job.path)
        self.at_once += 1
        self.most = max(self.most, self.at_once)
        try:
            await asyncio.sleep(self.delay)
            if job.path in self.failing:
                raise OSError(job.path)
            self.finished.append(job.path)
        finally:
            self.at_once -= 1

    def done(self, job: _Job) -> list[Any]:
        return [job.path]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        return [error]

    def exhausted(self) -> None:
        self.exhausted_calls += 1


def _toy(limit: asyncio.Semaphore, batches: list[list[str]], **kwargs: Any) -> Any:
    """The operation of a `_Toy`, and the `_Toy` itself."""
    jobs: list[_Toy] = []

    def run(op: Operation[Any, State]) -> AsyncGenerator[Any]:
        jobs.append(_Toy(limit, op, batches, **kwargs))
        return jobs[0].outcomes()

    op = Operation(run, State())
    return op, jobs


_BATCHES = [[f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)]]


async def test_runs_every_job_within_the_limit_and_gives_back_each_permit() -> None:
    limit = asyncio.Semaphore(3)
    op, jobs = _toy(limit, _BATCHES, failing=frozenset({"a2"}))
    items, errors = await _collect_with_errors(op)
    toy = jobs[0]
    assert sorted(items) == sorted({*_BATCHES[0], *_BATCHES[1]} - {"a2"})
    assert [e.args[0] for e in errors] == ["a2"]
    assert op.state.errors == 1
    assert toy.most <= 3
    assert toy.exhausted_calls == 1
    assert limit._value == 3


async def test_an_empty_source_ends_at_once() -> None:
    limit = asyncio.Semaphore(2)
    op, jobs = _toy(limit, [])
    assert await _collect_with_errors(op) == ([], [])
    assert jobs[0].exhausted_calls == 1
    assert limit._value == 2


async def test_cancel_yields_the_jobs_running_and_starts_no_more() -> None:
    limit = asyncio.Semaphore(2)
    op, jobs = _toy(limit, _BATCHES)
    seen = []
    async for item in op:
        seen.append(item)
        if len(seen) == 1:
            op.cancel()
    toy = jobs[0]
    # What started finished, and was yielded: nothing was left unknown.
    assert sorted(seen) == sorted(toy.started) == sorted(toy.finished)
    assert len(toy.started) < 10
    assert op.cancelled()
    assert limit._value == 2


async def test_closing_waits_for_the_jobs_running() -> None:
    limit = asyncio.Semaphore(2)
    op, jobs = _toy(limit, _BATCHES, failing=frozenset({"a1"}))
    async with op:
        async for _ in op:
            break
    toy = jobs[0]
    # The jobs running were waited for, not cancelled, and none started after.
    assert sorted(toy.started) == sorted([*toy.finished, "a1"])
    assert len(toy.started) < 10
    assert op.state.errors == 1
    assert limit._value == 2


async def test_cancelling_its_task_cancels_the_jobs_running() -> None:
    limit = asyncio.Semaphore(2)
    op, jobs = _toy(limit, _BATCHES, delay=10)
    task = asyncio.ensure_future(_collect_with_errors(op))
    await asyncio.sleep(0.005)
    assert jobs[0].started
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert jobs[0].finished == []
    assert limit._value == 2


async def test_a_shared_limit_is_waited_for_and_given_back() -> None:
    # Another holds the only permit: the queue's waiter gets it once given back.
    limit = asyncio.Semaphore(1)
    await limit.acquire()
    op, jobs = _toy(limit, _BATCHES[:1])
    task = asyncio.ensure_future(_collect_with_errors(op))
    await asyncio.sleep(0.02)
    assert jobs[0].started == []
    limit.release()
    items, _ = await task
    assert sorted(items) == _BATCHES[0]
    assert jobs[0].most == 1
    assert limit._value == 1


async def test_closing_a_queue_gives_back_the_permit_its_waiter_got() -> None:
    limit = asyncio.Semaphore(1)
    queue: _Queue[str] = _Queue(limit)
    queue.waiting.extend(["a", "b"])
    gate = asyncio.Event()

    async def run(item: str) -> None:
        await gate.wait()

    await queue.start(run)
    assert len(queue.running) == 1
    pending = queue.pending()
    assert queue.waiter in pending
    gate.set()
    await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
    # The waiter got the permit given back, which nothing took yet.
    assert queue.waiter is not None and queue.waiter.done()
    assert limit.locked()
    await queue.close()
    assert limit._value == 1


def test_roots_of_leave_out_what_a_matching_dir_has() -> None:
    matches = [
        {"name": "a/b", "type": "file"},
        {"name": "a", "type": "directory"},
        {"name": "c/", "type": "directory"},
        {"name": "d", "type": "file"},
    ]
    roots = _roots_of(matches, [], "*")
    assert (list(roots.directories), [f["name"] for f in roots.files]) == (
        ["a", "c"],
        ["d"],
    )


def test_roots_of_nothing_raise_unless_errors_were_found() -> None:
    with pytest.raises(FileNotFoundError):
        _roots_of([], [], "x*")
    error = OSError("x")
    assert _roots_of([], [error], "x*") == _Roots([], [])


# The placing of a copy.


async def _missing(path: str) -> dict[str, Any]:
    raise FileNotFoundError(path)


class _Copy(_CopyJobs):
    """A copy whose glob yields the entries and errors given, and which copies nothing."""

    def __init__(self, path1: Any, path2: str, found: list[Any]) -> None:
        storage = SimpleNamespace(operation_limit=asyncio.Semaphore(1), _info=_missing)
        super().__init__(
            fs=storage,
            dst_fs=storage,
            path1=path1,
            path2=path2,
            same_storage=True,
            op=Operation(lambda op: self.outcomes(), CpState()),
        )
        self.globbed = found

    def glob(self, pattern: Any) -> Operation[Any, Any]:
        async def entries() -> AsyncGenerator[Any]:
            for item in self.globbed:
                yield item

        return Operation(lambda op: entries(), State())

    # The jobs, never run here: only the placing is, and its errors yielded.

    def source(self, roots: _Roots) -> Any:
        return None  # Nothing to list.

    def fed(self, item: Any) -> list[Any]:
        raise NotImplementedError

    def file_root(self, path: str) -> None:
        raise NotImplementedError

    def run(self, job: Any) -> Any:
        raise NotImplementedError

    def done(self, job: Any) -> list[Any]:
        raise NotImplementedError

    def failed(self, job: Any, error: Exception) -> list[Any]:
        raise NotImplementedError


def _file(name: str) -> dict[str, Any]:
    return {"name": name, "type": "file"}


def _dir(name: str) -> dict[str, Any]:
    return {"name": name, "type": "directory"}


async def test_a_pattern_matching_only_errors_yields_them() -> None:
    error = OSError("x")
    copy = _Copy("x*", "dest", [error])
    assert await _collect_with_errors(copy.op) == ([], [error])
    assert copy.op.state.errors == 1


async def test_errors_are_yielded_when_nothing_else_is_to_do() -> None:
    # The first root is in place already, and the second's name is taken.
    copy = _Copy(["a/x", "b/*"], "a", [_file("a/x"), _file("b/x")])
    outcomes, errors = await _collect_with_errors(copy.op)
    assert outcomes == []
    assert [type(e) for e in errors] == [FileExistsError]
    assert copy.op.state.errors == 1


async def test_a_single_match_already_in_place_is_left() -> None:
    copy = _Copy("a/*.txt", "a", [_file("a/x.txt")])
    assert await copy.prepare() is None


async def test_matches_without_a_common_dir_go_under_the_destination() -> None:
    copy = _Copy("*1", "dest", [_dir("x1"), _file("y1")])
    assert await copy.prepare() is not None
    assert copy.prefix == ""
    assert (copy.target("x1/f"), copy.target("y1")) == ("dest/x1/f", "dest/y1")
