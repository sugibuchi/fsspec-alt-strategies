"""Tests of the scheduling of `common.tree`: no node is visited, the tasks are dummies."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies_impl.common.tree import PathTask, TreeTraversal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


class _Node(PathTask):
    """A task on a dummy tree: it emits its path before or after running its children."""

    def __init__(
        self,
        engine: TreeTraversal,
        path: str,
        tree: dict[str, list[str]],
        *,
        after: bool = False,
    ) -> None:
        super().__init__(engine, path)
        self.tree = tree
        self.after = after

    async def run(self) -> Any:
        if not self.after:
            await self._emit_event(self.path)
        children = [
            _Node(self._engine, c, self.tree, after=self.after)
            for c in self.tree.get(self.path, [])
        ]
        if children:
            await self._run_tasks(children)
        if self.after:
            await self._emit_event(self.path)
        return self.path


class _Call(PathTask):
    """A task running a function of itself."""

    def __init__(
        self, engine: TreeTraversal, path: str, body: Callable[[_Call], Any]
    ) -> None:
        super().__init__(engine, path)
        self.body = body

    async def run(self) -> Any:
        return await self.body(self)


async def _all(engine: TreeTraversal, tasks: list[PathTask]) -> list[Any]:
    return [e async for e in engine.events(tasks)]


_TREE = {"r": ["r/b", "r/a"], "r/a": ["r/a/x"]}


async def test_the_leftmost_pending_task_starts_first() -> None:
    engine = TreeTraversal(max_running=1)
    tasks = [_Node(engine, p, {}) for p in ["c", "a", "b"]]
    assert await _all(engine, tasks) == ["a", "b", "c"]


async def test_paths_compare_by_segments() -> None:
    # As strings, "a/b-x" comes before "a/b/c": by segments, after all below "a/b".
    engine = TreeTraversal(max_running=1)
    tasks = [_Node(engine, p, {}) for p in ["a/b-x", "a/b/c", "a/b"]]
    assert await _all(engine, tasks) == ["a/b", "a/b/c", "a/b-x"]


async def test_paths_split_as_given() -> None:
    engine = TreeTraversal(max_running=1, split=lambda path: path.split("."))
    tasks = [_Node(engine, p, {}) for p in ["a.b-x", "a.b.c", "a.b"]]
    assert await _all(engine, tasks) == ["a.b", "a.b.c", "a.b-x"]


@pytest.mark.parametrize(
    ("after", "expected"),
    [
        (False, ["r", "r/a", "r/a/x", "r/b"]),
        (True, ["r/a/x", "r/a", "r/b", "r"]),
    ],
)
async def test_depth_first_top_down_or_bottom_up(
    after: bool, expected: list[str]
) -> None:
    # A parent waiting holds no slot: with one, the tree is still walked.
    engine = TreeTraversal(max_running=1)
    assert await _all(engine, [_Node(engine, "r", _TREE, after=after)]) == expected


async def test_results_in_the_order_given() -> None:
    async def leaf(task: _Call) -> str:
        await asyncio.sleep(0.01 * (3 - int(task.path)))  # The last ends first.
        return task.path

    async def parent(task: _Call) -> None:
        children: list[PathTask] = [_Call(engine, str(i), leaf) for i in range(3)]
        await task._emit_event(await task._run_tasks(children))

    engine = TreeTraversal(max_running=3)
    assert await _all(engine, [_Call(engine, "p", parent)]) == [["0", "1", "2"]]


async def test_the_limits_hold() -> None:
    active = peak = pending = 0

    async def leaf(task: _Call) -> None:
        nonlocal active, peak, pending
        active += 1
        peak = max(peak, active)
        pending = max(pending, len(engine._pending_tasks))
        await asyncio.sleep(0.001)
        active -= 1

    async def parent(task: _Call) -> None:
        await task._run_tasks([_Call(engine, f"p/{i:02}", leaf) for i in range(20)])

    engine = TreeTraversal(max_running=3, max_pending=2)
    await _all(engine, [_Call(engine, "p", parent)])
    assert peak == 3
    assert pending <= 2


async def test_tasks_streamed_are_taken_as_room_frees() -> None:
    made = started = 0
    ahead = 0  # The most made before they start.

    async def leaf(task: _Call) -> None:
        nonlocal started
        started += 1
        await asyncio.sleep(0.001)

    async def children() -> AsyncIterator[PathTask]:
        nonlocal made, ahead
        for i in range(20):
            made += 1
            ahead = max(ahead, made - started)
            yield _Call(engine, f"p/{i:02}", leaf)

    async def parent(task: _Call) -> None:
        await task._emit_event(len(await task._run_tasks(children())))

    engine = TreeTraversal(max_running=2, max_pending=3)
    assert await _all(engine, [_Call(engine, "p", parent)]) == [20]
    # Those pending, those starting and the one made waiting for room.
    assert ahead <= 3 + 2 + 1


async def test_a_stream_breaking_off_lets_the_tasks_taken_end() -> None:
    ended = []

    async def leaf(task: _Call) -> None:
        await asyncio.sleep(0.01)
        ended.append(task.path)

    async def children() -> AsyncIterator[PathTask]:
        for c in "ab":
            yield _Call(engine, f"p/{c}", leaf)
        raise OSError("listing")

    async def parent(task: _Call) -> None:
        await task._run_tasks(children())

    engine = TreeTraversal(max_running=2)
    # The stream's error alone, none of the tasks failing: raised as it is.
    with pytest.raises(OSError, match="listing"):
        await _all(engine, [_Call(engine, "p", parent)])
    assert sorted(ended) == ["p/a", "p/b"]


async def test_failed_children_raise_an_exception_group_after_all_ran() -> None:
    ran = []

    async def leaf(task: _Call) -> None:
        ran.append(task.path)
        if task.path != "b":
            raise ValueError(task.path)

    async def parent(task: _Call) -> None:
        try:
            await task._run_tasks([_Call(engine, p, leaf) for p in "cab"])
        except ExceptionGroup as group:
            await task._emit_event([str(e) for e in group.exceptions])

    engine = TreeTraversal(max_running=1)
    assert await _all(engine, [_Call(engine, "p", parent)]) == [["c", "a"]]
    assert sorted(ran) == ["a", "b", "c"]


async def test_a_child_cancelled_before_it_starts_raises_cancelled() -> None:
    async def leaf(task: _Call) -> None:
        await task._emit_event(task.path)

    async def parent(task: _Call) -> None:
        children: list[PathTask] = [_Call(engine, p, leaf) for p in "ab"]
        children[0].cancel()
        try:
            await task._run_tasks(children)
        except asyncio.CancelledError:
            await task._emit_event("cancelled")

    engine = TreeTraversal(max_running=1)
    assert await _all(engine, [_Call(engine, "p", parent)]) == ["b", "cancelled"]


async def test_an_initial_task_failing_ends_the_traversal() -> None:
    async def fail(task: _Call) -> None:
        raise OSError(task.path)

    engine = TreeTraversal(max_running=1)
    with pytest.raises(OSError, match="x"):
        await _all(engine, [_Call(engine, "x", fail)])


async def test_an_error_emitted_is_yielded() -> None:
    async def emit(task: _Call) -> None:
        await task._emit_event(OSError(task.path))

    engine = TreeTraversal(max_running=1)
    [error] = await _all(engine, [_Call(engine, "x", emit)])
    assert isinstance(error, OSError)


async def test_close_cancels_every_task_and_ends_the_events() -> None:
    cancelled = []

    async def leaf(task: _Call) -> None:
        await task._emit_event(task.path)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(task.path)
            raise

    async def parent(task: _Call) -> None:
        try:
            await task._run_tasks([_Call(engine, f"p/{c}", leaf) for c in "abcd"])
        except asyncio.CancelledError:
            cancelled.append(task.path)
            raise

    engine = TreeTraversal(max_running=2)
    seen = []
    async for event in engine.events([_Call(engine, "p", parent)]):
        seen.append(event)
        engine.close()
    # Both running children emitted before the close: those events are read,
    # and nothing starts after. The running children and their parent are
    # cancelled.
    assert seen == ["p/a", "p/b"]
    assert sorted(cancelled) == ["p", "p/a", "p/b"]


async def test_close_from_elsewhere_wakes_the_reader() -> None:
    async def forever(task: _Call) -> None:
        await asyncio.Event().wait()

    engine = TreeTraversal(max_running=1)
    asyncio.get_running_loop().call_later(0.01, engine.close)
    seen = await asyncio.wait_for(_all(engine, [_Call(engine, "x", forever)]), 1)
    assert seen == []


async def test_close_runs_once() -> None:
    async def forever(task: _Call) -> None:
        await asyncio.Event().wait()

    engine = TreeTraversal(max_running=1)
    events = engine.events([_Call(engine, "x", forever)])
    reading = asyncio.ensure_future(anext(events, None))
    await asyncio.sleep(0)
    engine.close()
    closer = engine._closer
    engine.close()
    assert engine._closer is closer
    assert await reading is None
    engine.close()  # Ended: nothing to close.
    assert engine._closer is closer


async def test_an_event_emitted_waits_for_room() -> None:
    sizes = []

    async def emit(task: _Call) -> None:
        for i in range(6):
            await task._emit_event(i)
            sizes.append(engine._events.qsize())

    engine = TreeTraversal(max_running=1, max_events=2)
    seen = []
    async for event in engine.events([_Call(engine, "x", emit)]):
        seen.append(event)
        await asyncio.sleep(0.001)  # A slow caller.
    assert seen == list(range(6))
    assert max(sizes) <= 2


async def test_no_event_is_emitted_once_closed() -> None:
    raised = []

    async def emit(task: _Call) -> None:
        await task._emit_event(1)
        try:
            await asyncio.Event().wait()
        finally:
            try:
                await task._emit_event(2)
            except RuntimeError as error:
                raised.append(error)

    engine = TreeTraversal(max_running=1)
    seen = []
    async for event in engine.events([_Call(engine, "x", emit)]):
        seen.append(event)
        engine.close()
    assert seen == [1]
    assert len(raised) == 1


async def test_closing_early_cancels_the_tasks_running() -> None:
    cancelled = []

    async def forever(task: _Call) -> None:
        await task._emit_event(task.path)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(task.path)
            raise

    engine = TreeTraversal(max_running=2)
    events = engine.events([_Call(engine, p, forever) for p in "ab"])
    assert await anext(events) == "a"
    await events.aclose()
    assert sorted(cancelled) == ["a", "b"]


async def test_a_failure_unread_is_raised_closing() -> None:
    failed = asyncio.Event()

    async def emit(task: _Call) -> None:
        for i in range(3):
            await task._emit_event(i)
        await asyncio.Event().wait()

    async def fail(task: _Call) -> None:
        failed.set()
        raise OSError(task.path)

    engine = TreeTraversal(max_running=2, max_events=1)
    events = engine.events([_Call(engine, "a", emit), _Call(engine, "b", fail)])
    assert await anext(events) == 0
    await failed.wait()
    for _ in range(3):
        await asyncio.sleep(0)  # Until b's failure is queued, past the limit.
    assert engine._events.qsize() > engine.max_events
    # The caller closes before reading it: closing raises it.
    with pytest.raises(BaseExceptionGroup) as raised:
        await events.aclose()
    assert [str(e) for e in raised.value.exceptions] == ["b"]
    await asyncio.sleep(0)
    assert asyncio.all_tasks() == {asyncio.current_task()}


async def test_an_error_closing_is_raised_after_the_error_ending() -> None:
    async def fail(task: _Call) -> None:
        await asyncio.sleep(0.01)
        raise OSError(task.path)

    async def cleanup_fails(task: _Call) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            raise ValueError(task.path)  # As a failing cleanup.

    engine = TreeTraversal(max_running=2)
    tasks = [_Call(engine, "a", fail), _Call(engine, "b", cleanup_fails)]
    with pytest.raises(BaseExceptionGroup) as raised:
        await _all(engine, tasks)
    # The closing raises its own group, chained to the error ending events.
    [error] = raised.value.exceptions
    assert (type(error), str(error)) == (ValueError, "b")
    ending = raised.value.__context__
    assert (type(ending), str(ending)) == (OSError, "a")


async def test_a_child_error_is_raised_by_its_parent_alone() -> None:
    async def fail(task: _Call) -> None:
        raise OSError(task.path)

    async def parent(task: _Call) -> None:
        await task._run_tasks([_Call(engine, "p/c", fail)])

    engine = TreeTraversal(max_running=1)
    with pytest.raises(BaseExceptionGroup) as raised:
        await _all(engine, [_Call(engine, "p", parent)])
    # The parent's group, as the initial task's error: no closing group.
    [error] = raised.value.exceptions
    assert (type(error), str(error)) == (OSError, "p/c")
    assert raised.value.__context__ is None


async def test_cancel_starts_nothing_more_and_lets_the_running_end() -> None:
    async def leaf(task: _Call) -> str:
        await task._emit_event(task.path)
        await asyncio.sleep(0.01)
        await task._emit_event(task.path + " done")
        return task.path

    async def parent(task: _Call) -> None:
        try:
            await task._run_tasks([_Call(engine, f"p/{c}", leaf) for c in "abcd"])
        except asyncio.CancelledError:
            await task._emit_event("p cancelled")  # Still running: it emits.
            raise

    engine = TreeTraversal(max_running=2)
    seen = []
    async for event in engine.events([_Call(engine, "p", parent)]):
        seen.append(event)
        engine.cancel()
    # The two running end, and emit: the two pending never start.
    assert sorted(seen) == ["p cancelled", "p/a", "p/a done", "p/b", "p/b done"]
    assert seen[-1] == "p cancelled"


async def test_a_child_error_its_cancelled_parent_drops_is_raised_closing() -> None:
    async def cleanup_fails(task: _Call) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            raise ValueError(task.path)  # As a failing cleanup.

    async def parent(task: _Call) -> None:
        await task._run_tasks([_Call(engine, "p/c", cleanup_fails)])

    engine = TreeTraversal(max_running=1)
    events = engine.events([_Call(engine, "p", parent)])
    reading = asyncio.ensure_future(anext(events, None))
    for _ in range(5):
        await asyncio.sleep(0)  # Until p/c runs.
    engine.close()
    with pytest.raises(BaseExceptionGroup) as raised:
        await reading
    # Raised once, by close's _aclose, inside the group of the one ending events.
    [[error]] = [group.exceptions for group in raised.value.exceptions]
    assert (type(error), str(error)) == (ValueError, "p/c")


async def test_a_traversal_runs_once() -> None:
    engine = TreeTraversal(max_running=1)
    await _all(engine, [])
    with pytest.raises(RuntimeError):
        await _all(engine, [])


def test_limits_are_checked() -> None:
    with pytest.raises(AssertionError):
        TreeTraversal(max_running=1, max_pending=0)
