# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A scheduler of tasks on paths, the engine under a traversal of a tree or a key space.

A `TreeTraversal` runs `PathTask`s, each on one path. A task may run others,
e.g. a directory's task its subdirectories', and wait for their results. It
may emit any object as an event, which the traversal's caller reads in the
order emitted. Nothing here lists or visits: the tasks do.

The tasks are the strategy: how a tree is listed and in what order. What is
done on each path is a `PathAction`'s: its `map` on a path before what is below
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
    from collections.abc import AsyncGenerator, Callable, Coroutine, Iterable

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


class PathAction:
    """What an operation does on each path a traversal visits: the strategy is the `PathTask`'s.

    A task calls `map` on a path before traversing what is below it, and
    `reduce` on a directory after. An item's value is what `reduce` returned
    for a directory, or `map` for a file. Each defaults to None, so a subclass
    writes only those it needs.
    """

    def __init__(self, emit_event: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        # The traversal's, which a strategy passes: one action serves every task.
        self._emit_event = emit_event

    async def map(self, info: dict[str, Any]) -> Any:
        """Act on a path, before its subtree is traversed: a value for `reduce`, or None.

        ``info`` is the path's entry, as a listing or ``info`` gave it.
        """
        return None

    async def reduce(self, info: dict[str, Any], value: Any, values: list[Any]) -> Any:
        """Act on a directory, after its subtree is traversed: its value, or None.

        ``value`` is what `map` returned for it, and ``values`` the values of
        the items in it.
        """
        return None

    async def emit_event(self, event: Any) -> None:
        """Emit an event to the traversal's caller, waiting while too many are unread."""
        await self._emit_event(event)


class _End:
    """The event ending the traversal: no task is pending or running."""


class _Failed:
    """The event of a task given to `events` that failed: the traversal raises its error."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


class TreeTraversal:
    """One traversal: the tasks pending, running and waiting, and the events not yet read.

    `events` runs it, once. `cancel` starts nothing more: the tasks running
    finish, and their events are read. Closing `events` early cancels the tasks
    running at once, and waits for them to end.
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
        # The tasks given to events: their failure ends the traversal.
        self._initial: set[PathTask] = set()
        self._started = False
        self._stopping = False
        self._closing = False

    async def events(self, tasks: Iterable[PathTask]) -> AsyncGenerator[Any]:
        """Run ``tasks``, and those they run, yielding every event they emit, in order.

        It raises the error of a task in ``tasks`` that fails, which ends the
        traversal. An error a task emits is yielded as any event. It raises
        RuntimeError if called twice.
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

        scheduler = asyncio.create_task(self._schedule())
        try:
            while True:
                event = await self._events.get()
                if isinstance(event, _End):
                    return
                elif isinstance(event, _Failed):
                    raise event.error
                else:
                    yield event
        finally:
            await self._close(scheduler)

    def cancel(self) -> None:
        """Start nothing more: the tasks running finish, and their events are yielded."""
        if not self._stopping:
            self._stopping = True
            asyncio.get_running_loop().create_task(self._cancel_pending())

    async def _cancel_pending(self) -> None:
        async with self._state:
            self._drop_pending()
            self._state.notify_all()

    def _drop_pending(self) -> None:
        """Cancel every task not started: their parents see them done."""
        for task in self._pending_tasks:
            task._cancelled = (
                True  # Each would notify on its own: the caller does once.
            )
        self._pending_tasks.clear()

    async def _schedule(self) -> None:
        """Start the leftmost pending task while a slot is free, until nothing is pending or running."""
        while True:
            async with self._state:
                await self._state.wait_for(self._can_go_on)
                if not self._running_tasks and (
                    self._stopping or not self._pending_tasks
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
        await self._emit_event(_End())

    def _can_go_on(self) -> bool:
        """Whether the scheduler can start a task, or end."""
        active = len(self._running_tasks) - len(self._waiting_tasks)
        can_start = (
            not self._stopping and self._pending_tasks and active < self.max_running
        )
        can_end = not self._running_tasks and (
            self._stopping or not self._pending_tasks
        )
        return bool(can_start or can_end)

    def _on_done(self, task: PathTask, _: object) -> None:
        # A done callback is synchronous: the lock is taken in a task of its own.
        asyncio.get_running_loop().create_task(self._done(task))

    def _cancelled_before_start(self, task: PathTask) -> None:
        self._on_done(task, None)

    async def _done(self, task: PathTask) -> None:
        """A task is done, ended or cancelled before it started: forget it, and wake every waiter."""
        # Read always, so that asyncio never warns of an exception not retrieved.
        error = None if task.cancelled() else task.exception()
        if error is not None and task in self._initial and not self._closing:
            # Emitted while it still runs, so that the end never comes before it.
            await self._emit_event(_Failed(error))
        async with self._state:
            self._running_tasks.discard(task)
            self._pending_tasks.pop(task, None)
            self._state.notify_all()

    async def _add_tasks(self, tasks: list[PathTask]) -> None:
        """Add tasks to start, each waiting for room: once stopping, cancel them instead."""
        async with self._state:
            for task in tasks:
                if task._task is not None or task in self._pending_tasks:
                    raise ValueError(f"Task {task.path!r} is added twice.")
                if task.done():
                    continue  # Cancelled already: nothing to run.
                await self._state.wait_for(
                    lambda: (
                        self._stopping or len(self._pending_tasks) < self.max_pending
                    )
                )
                if self._stopping:
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
        await self._events.put(event)

    async def _close(self, scheduler: asyncio.Task[None]) -> None:
        """Cancel the scheduler and every task running, and wait for them to end."""
        self._stopping = True
        self._closing = True
        self._drop_pending()
        running = [scheduler, *(t._task for t in self._running_tasks if t._task)]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
