# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A scheduler of tasks on paths, the engine under a traversal of a tree or a key space.

A `TreeTraversal` runs `PathTask`s, each on one path. A task may run others,
e.g. a directory's task its subdirectories', and wait for their results. It
may emit any object as an event, which the traversal's caller reads in the
order emitted. Nothing here lists or visits: the tasks do.

The tasks are the strategy: how a tree is listed and in what order. What is
done on each path is `glob.PathAction`'s: its `map` on a path before what is below
it, its `reduce` on a directory after.

Three limits bound it:

- ``max_running``: the tasks running at once. A task waiting for those it
  runs does not count, so that parents never hold every slot while their
  children wait. Once its children are done, it counts again, so the tasks
  running may briefly exceed it.
- ``max_pending``: the tasks waiting to start. A task adding more waits for
  room, which bounds the memory of a wide directory.
- ``max_events``: the events not yet read. A task emitting more waits, so a
  slow caller slows the traversal.

The task taken next is the pending one with the first path, compared by its
segments as ``split`` gives them. So the tasks under the leftmost subtree go
before its siblings', and the traversal goes quasi depth first: it finishes
subtrees, instead of spreading over siblings. Only quasi, as the tasks
running at once may be siblings, and a task running is never put back.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable, Sequence

__all__: list[str] = []


class PathTask(ABC):
    """A task on one path, run by a `TreeTraversal`: an `asyncio.Task` once it starts.

    A subclass writes `run`. It runs other tasks by `_run_tasks`, and emits
    events by `_emit_event`.

    Before it starts, it may be cancelled: it is then done and cancelled, and
    never starts. Its other methods raise `asyncio.InvalidStateError` until it
    has started.
    """

    def __init__(self, engine: TreeTraversal, path: str) -> None:
        self._engine = engine
        self.path = path
        self._task: asyncio.Task[Any] | None = None
        self._cancelled = False

    @abstractmethod
    async def run(self) -> Any:
        """The task's work: its return value is its result."""

    def done(self) -> bool:
        return self._task.done() if self._task is not None else self._cancelled

    def cancelled(self) -> bool:
        return self._task.cancelled() if self._task is not None else self._cancelled

    def result(self) -> Any:
        if self._task is not None:
            return self._task.result()
        elif self._cancelled:
            raise asyncio.CancelledError()
        else:
            raise asyncio.InvalidStateError(f"Task {self.path!r} has not started.")

    def exception(self) -> BaseException | None:
        if self._task is not None:
            return self._task.exception()
        elif self._cancelled:
            raise asyncio.CancelledError()
        else:
            raise asyncio.InvalidStateError(f"Task {self.path!r} has not started.")

    def cancel(self, msg: Any = None) -> bool:
        """Cancel the task, as `asyncio.Task.cancel` does: one not started never starts."""
        if self._task is not None:
            return self._task.cancel(msg)
        elif self._cancelled:
            return False  # Done already, as a task cancelled is.
        else:
            self._cancelled = True
            self._engine._cancelled_before_start(self)
            return True

    async def _run_tasks(self, tasks: list[PathTask]) -> list[Any]:
        """Run ``tasks`` and wait for all of them: see `TreeTraversal._run_tasks`."""
        return await self._engine._run_tasks(self, tasks)

    async def _emit_event(self, event: Any) -> None:
        """Emit an event to the traversal's caller, waiting while too many are unread."""
        await self._engine._emit_event(event)

    def _set_task(self, task: asyncio.Task[Any]) -> None:
        assert self._task is None and not self._cancelled
        self._task = task


async def _wait(tasks: Sequence[asyncio.Task[Any]]) -> None:
    """Wait for tasks to end, if any: asyncio.wait refuses none."""
    if tasks:
        await asyncio.wait(tasks)


class _End:
    """The event ending the traversal: no task is pending or running."""


class _Failed:
    """The event of a task given to `events` that failed: the traversal raises its error."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


class TreeTraversal:
    """One traversal: the tasks pending, running and waiting, and the events not yet read.

    `events` runs it, once. `close`, or `_aclose`, cancels every task, pending
    or running: a parent waiting for one gets CancelledError. No task emits
    any more event, and `events` ends once none runs. Closing `events` early
    closes it, and waits for the tasks to end.
    """

    def __init__(
        self,
        *,
        max_running: int,
        max_pending: int = 1000,
        max_events: int = 16,
        split: Callable[[str], list[str]] | None = None,
    ) -> None:
        assert max_running >= 1
        assert max_pending >= 1
        assert max_events >= 1
        self.max_running = max_running
        self.max_pending = max_pending
        # A path's segments, e.g. a handler's _split: by "/" by default.
        self.split = split or (lambda path: path.split("/"))
        # One condition over all the state: every change notifies, and each
        # waiter checks its own predicate.
        self._state = asyncio.Condition()
        # The tasks waiting to start, each with its path's segments, its key.
        self._pending_tasks: dict[PathTask, list[str]] = {}
        self._running_tasks: set[PathTask] = set()
        self._waiting_tasks: set[PathTask] = set()
        self._events: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_events)
        # The tasks given to events whose error is not raised yet: their
        # failure ends the traversal. One is dropped once its error is raised,
        # by events or by an _aclose, so that none raises it again.
        self._initial: set[PathTask] = set()
        # The traversal's own tasks, finishing a task or cancelling: closing
        # cancels those left, e.g. one emitting to a queue no longer read.
        self._own_tasks: set[asyncio.Task[None]] = set()
        # The task starting the others, while events runs.
        self._scheduler: asyncio.Task[None] | None = None
        self._started = False
        self._closed = False

    async def events(self, tasks: Iterable[PathTask]) -> AsyncGenerator[Any]:
        """Run ``tasks``, and those they run, yielding every event they emit, in order.

        It raises the error of a task in ``tasks`` that fails, which ends the
        traversal. An error a task emits is yielded as any event. It raises
        RuntimeError if called twice.

        As it ends, it closes the traversal by `_aclose`, which raises the
        errors of the tasks closing: the error ending it, if any, is then
        their ExceptionGroup's context.
        """
        if self._started:
            raise RuntimeError("A traversal runs once.")
        self._started = True
        for task in tasks:
            if task in self._pending_tasks:
                raise ValueError(f"Task {task.path!r} is added twice.")
            # Given up front, they are in memory already: no room is waited for.
            self._initial.add(task)
            self._pending_tasks[task] = self.split(task.path)

        self._scheduler = asyncio.create_task(self._schedule())
        try:
            while True:
                if self._closed and self._events.empty():
                    return  # No event is emitted once closed: all are read.
                event = await self._events.get()
                if isinstance(event, _End):
                    return
                elif isinstance(event, _Failed):
                    raise event.error
                else:
                    yield event
        finally:
            await self._aclose()

    def close(self) -> None:
        """Cancel every task, pending or running, and every event emitted after: `_aclose` in a task.

        Closed at once, it starts no task and emits no event more, though the
        tasks are cancelled once `_aclose` runs.
        """
        self._closed = True
        self._spawn(self._aclose())

    async def _aclose(self) -> None:
        """Cancel every task, pending or running, and wait for them to end: a parent waiting for one gets CancelledError.

        No event is emitted after, so `events` yields those emitted before,
        then ends. The scheduler and the traversal's own tasks are cancelled:
        none is needed once closed. It raises the errors of the tasks given to
        `events`, the scheduler and its own tasks, but their cancellation, in
        an ExceptionGroup: another task's error is its parent's.
        """
        self._closed = True
        async with self._state:
            for task in self._pending_tasks:
                task._cancelled = True  # Never starts: its parent sees it done.
            self._pending_tasks.clear()
            for task in self._running_tasks:
                task.cancel()
            self._state.notify_all()
        if self._events.empty():
            # Wakes events, if it waits for an event: none would come.
            self._events.put_nowait(_End())

        # asyncio.wait, never gather: an _aclose cancelled meanwhile cancels
        # none of them, and leaves their errors to the other _aclose.
        running = [t for t in self._running_tasks if t._task is not None]
        await _wait([t._task for t in running if t._task is not None])
        errors: list[BaseException] = []
        for task in running:
            # Read always, so that asyncio never warns of an exception not
            # retrieved. A task's error is its parent's, raised by its
            # _run_tasks: only those given to events, which have none, are
            # raised here, once, as they are dropped from _initial.
            error = None if task.cancelled() else task.exception()
            if task in self._initial:
                self._initial.discard(task)
                if error is not None:
                    errors.append(error)

        ends: list[asyncio.Task[Any]] = []
        if self._scheduler is not None:
            self._scheduler.cancel()
            ends.append(self._scheduler)
        # Those ending above have spawned their _done: none is needed now.
        ends += [t for t in self._own_tasks if t is not asyncio.current_task()]
        for end in ends:
            end.cancel()
        await _wait(ends)
        errors += [
            e for t in ends if not t.cancelled() and (e := t.exception()) is not None
        ]
        if errors:
            raise BaseExceptionGroup("The traversal failed to close.", errors)

    async def _schedule(self) -> None:
        """Start the leftmost pending task while a slot is free, until nothing is pending or running."""
        while True:
            async with self._state:
                await self._state.wait_for(self._can_go_on)
                if not self._running_tasks and (
                    self._closed or not self._pending_tasks
                ):
                    break
                # Depth first among those pending: segments compare as a walk
                # visits them, "a/b/c" between "a/b" and "a/b-x".
                task = min(self._pending_tasks, key=self._pending_tasks.__getitem__)
                del self._pending_tasks[task]
                if not task.cancelled():
                    running = asyncio.create_task(task.run())
                    running.add_done_callback(partial(self._on_done, task))
                    task._set_task(running)
                    self._running_tasks.add(task)
                self._state.notify_all()  # Room for a task more.
        await self._events.put(_End())

    def _can_go_on(self) -> bool:
        """Whether the scheduler can start a task, or end."""
        active = len(self._running_tasks) - len(self._waiting_tasks)
        can_start = (
            not self._closed and self._pending_tasks and active < self.max_running
        )
        can_end = not self._running_tasks and (self._closed or not self._pending_tasks)
        return bool(can_start or can_end)

    def _on_done(self, task: PathTask, _: object) -> None:
        # A done callback is synchronous: the lock is taken in a task of its own.
        self._spawn(self._done(task))

    def _spawn(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._own_tasks.add(task)
        task.add_done_callback(self._own_tasks.discard)

    def _cancelled_before_start(self, task: PathTask) -> None:
        self._on_done(task, None)

    async def _done(self, task: PathTask) -> None:
        """A task is done, ended or cancelled before it started: forget it, and wake every waiter."""
        # Read always, so that asyncio never warns of an exception not retrieved.
        error = None if task.cancelled() else task.exception()
        if error is not None and task in self._initial and not self._closed:
            # Emitted while it still runs, so that the end never comes before it.
            await self._events.put(_Failed(error))
            self._initial.discard(task)  # Raised by events: not again.
        async with self._state:
            self._running_tasks.discard(task)
            self._pending_tasks.pop(task, None)
            self._state.notify_all()

    async def _add_tasks(self, tasks: list[PathTask]) -> None:
        """Add tasks to start, each waiting for room: once closed, cancel them instead."""
        async with self._state:
            for task in tasks:
                if task._task is not None or task in self._pending_tasks:
                    raise ValueError(f"Task {task.path!r} is added twice.")
                if task.done():
                    continue  # Cancelled already: nothing to run.
                await self._state.wait_for(
                    lambda: self._closed or len(self._pending_tasks) < self.max_pending
                )
                if self._closed:
                    task._cancelled = True  # Never runs: its parent sees it done.
                else:
                    self._pending_tasks[task] = self.split(task.path)
                self._state.notify_all()

    async def _run_tasks(self, caller: PathTask, tasks: list[PathTask]) -> list[Any]:
        """Run ``tasks`` for ``caller``, which waits: their results, in the order given.

        ``caller`` counts no more against ``max_running`` while it waits. Once
        all are done, it raises CancelledError if any was cancelled, else an
        ExceptionGroup of those that failed, in the order given. None is
        cancelled for another's failure.
        """
        async with self._state:
            if caller not in self._running_tasks or caller in self._waiting_tasks:
                raise ValueError(
                    f"Task {caller.path!r} is not running, or waits already."
                )
            self._waiting_tasks.add(caller)
            self._state.notify_all()
        try:
            await self._add_tasks(tasks)
            async with self._state:
                await self._state.wait_for(lambda: all(t.done() for t in tasks))
        finally:
            async with self._state:
                self._waiting_tasks.discard(caller)
                self._state.notify_all()

        if any(t.cancelled() for t in tasks):
            raise asyncio.CancelledError()
        errors = [e for t in tasks if (e := t.exception()) is not None]
        if errors:
            # An ExceptionGroup, unless an error is no Exception, e.g. a KeyboardInterrupt.
            raise BaseExceptionGroup(
                f"{len(errors)} of {len(tasks)} tasks failed", errors
            )
        return [t.result() for t in tasks]

    async def _emit_event(self, event: Any) -> None:
        if self._closed:
            raise RuntimeError("The traversal is closed: no event is emitted.")
        await self._events.put(event)
