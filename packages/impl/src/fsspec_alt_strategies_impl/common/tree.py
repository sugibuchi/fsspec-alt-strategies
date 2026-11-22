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
  slow caller slows the traversal. The traversal's own, its end and a
  failure, never wait: so finishing a task never blocks.

The task taken next is the pending one with the first path, compared by its
segments as ``split`` gives them. So the tasks under the leftmost subtree go
before its siblings', and the traversal goes quasi depth first: it finishes
subtrees, instead of spreading over siblings. Only quasi, as the tasks
running at once may be siblings, and a task running is never put back.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable as AsyncIterableABC
from functools import partial
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterable,
        AsyncIterator,
        Callable,
        Iterable,
        Sequence,
    )

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
        # The group _run_tasks runs it in, its caller's: None for one given to
        # events.
        self._group: PathTaskGroup | None = None

    def _tell_group(self) -> None:
        """Tell its group it is done, ended or cancelled before it started."""
        if self._group is not None:
            self._group.on_task_done()

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

    async def _run_tasks(
        self, tasks: Iterable[PathTask] | AsyncIterable[PathTask]
    ) -> list[Any]:
        """Run ``tasks`` and wait for all of them: see `TreeTraversal._run_tasks`."""
        return await self._engine._run_tasks(self, tasks)

    async def _emit_event(self, event: Any) -> None:
        """Emit an event to the traversal's caller, waiting while too many are unread."""
        await self._engine._emit_event(event)

    def _set_task(self, task: asyncio.Task[Any]) -> None:
        assert self._task is None and not self._cancelled
        self._task = task


class PathTaskGroup:
    """The tasks one `_run_tasks` runs for its caller, and how many of them are done.

    Its status is checked as a task is done, by `on_task_done`, so that the
    caller waiting reads `done` in O(1) at each wake.
    """

    def __init__(self, caller: PathTask) -> None:
        self.caller = caller
        self.tasks: list[PathTask] = []
        # Read only once all are taken, and checked again then: while a
        # stream of them runs on, it may be true for those taken so far.
        self._done = False
        # The tasks before it are done. It only moves on, each task passed
        # once, and each check stops at the first not done: O(n) in all, in
        # whatever order the tasks end.
        self._first_not_done = 0

    def add(self, task: PathTask) -> None:
        task._group = self
        self.tasks.append(task)

    def on_task_done(self) -> None:
        """Check the status of the tasks, once one is done or all are taken."""
        while (
            self._first_not_done < len(self.tasks)
            and self.tasks[self._first_not_done].done()
        ):
            self._first_not_done += 1
        self._done = self._first_not_done == len(self.tasks)

    def done(self) -> bool:
        """Whether every task is taken and done."""
        return self._done


async def _aiter(
    tasks: Iterable[PathTask] | AsyncIterable[PathTask],
) -> AsyncIterator[PathTask]:
    """Tasks given either way, taken one at a time."""
    if isinstance(tasks, AsyncIterableABC):
        async for task in tasks:
            yield task
    else:
        for task in tasks:
            yield task


class _Failures(BaseExceptionGroup):
    """The errors the traversal raises together, of the tasks a `_run_tasks` ran, or closing.

    Flat: one of them in the errors is unpacked into its own, so that an
    error below is one level down whatever its depth. A group a task raises
    itself is kept whole. An ExceptionGroup, `_ExceptionFailures`, unless an
    error is no Exception, e.g. a KeyboardInterrupt.
    """

    def __new__(cls, message: str, errors: Sequence[BaseException]) -> Self:
        flat: list[BaseException] = []
        for error in errors:
            if isinstance(error, _Failures):
                flat += error.exceptions
            else:
                flat.append(error)
        made: type[Any] = cls
        if cls is _Failures and all(isinstance(e, Exception) for e in flat):
            made = _ExceptionFailures
        return super().__new__(made, message, flat)

    def __init__(self, message: str, errors: Sequence[BaseException]) -> None:
        # The flat errors are __new__'s: BaseExceptionGroup keeps them.
        pass


class _ExceptionFailures(ExceptionGroup, _Failures):
    """`_Failures` of Exceptions alone: caught as any Exception."""


class _End:
    """The event ending the traversal: no task is pending or running."""


class _Failed:
    """The event of a task given to `events` that failed: the traversal raises its error."""

    def __init__(self, task: PathTask, error: BaseException) -> None:
        self.task = task
        self.error = error


class TreeTraversal:
    """One traversal: the tasks pending, running and waiting, and the events not yet read.

    `events` runs it, once. Its stopping mirrors `asyncio.Task`'s:

    - `cancel` starts nothing more: the tasks pending are cancelled, those
      running run on and emit their events, and `events` ends once none runs;
    - `close`, or `_aclose`, cancels every task, pending or running, and no
      task emits any more event. Closing `events` early closes it, and waits
      for the tasks to end.

    Either way, a parent waiting for a task cancelled gets CancelledError.
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
        self.max_events = max_events
        # A path's segments: a strategy passes its handler's segments. By "/"
        # by default.
        self.split = split or (lambda path: path.split("/"))
        # One condition over all the state: every change notifies, and each
        # waiter checks its own predicate.
        self._state = asyncio.Condition()
        # The tasks waiting to start.
        self._pending_tasks: set[PathTask] = set()
        # The same, in a heap by their path's segments, the leftmost first: the
        # order added breaks ties. One dropped from _pending_tasks, cancelled
        # before it starts, stays until popped, then is skipped.
        self._heap: list[tuple[list[str], int, PathTask]] = []
        self._added = itertools.count()
        self._running_tasks: set[PathTask] = set()
        self._waiting_tasks: set[PathTask] = set()
        # Unbounded: _emit_event waits for room itself, the traversal's own
        # events never.
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        # The tasks given to events whose error is not raised yet: their
        # failure ends the traversal. One is dropped once its error is raised,
        # by events or by an _aclose, so that none raises it again.
        self._initial: set[PathTask] = set()
        # The tasks whose parent was cancelled, their error raised by an
        # _aclose: the other, close's or events', raises it not again.
        self._orphans_raised: set[PathTask] = set()
        # The _aclose that close runs in a task: the one ending events waits
        # for it, and raises its errors.
        self._closer: asyncio.Task[None] | None = None
        # The task cancel runs to drop the tasks pending: _aclose waits for it.
        self._stopper: asyncio.Task[None] | None = None
        # The task starting the others, while events runs.
        self._scheduler: asyncio.Task[None] | None = None
        self._started = False
        # No task starts once stopping, and none emits once closed: closed is
        # stopping too.
        self._stopping = False
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
            self._pend(task)

        self._scheduler = asyncio.create_task(self._schedule())
        try:
            while True:
                if self._closed and self._events.empty():
                    return  # No event is emitted once closed: all are read.
                event = await self._events.get()
                async with self._state:
                    self._state.notify_all()  # Room for an event more.
                if isinstance(event, _End):
                    return
                elif isinstance(event, _Failed):
                    self._initial.discard(event.task)  # Raised: not again.
                    raise event.error
                else:
                    yield event
        finally:
            await self._aclose()

    def cancel(self) -> None:
        """Start no task more: those pending are cancelled, those running run on and emit their events.

        `events` yields them, and ends once none runs. Once stopping, or
        closed, it does nothing.
        """
        if self._stopping:
            return
        self._stopping = True
        self._stopper = asyncio.get_running_loop().create_task(self._drop_pending())

    async def _drop_pending(self) -> None:
        """Cancel the tasks pending, which never start, and wake their parents and the scheduler."""
        async with self._state:
            for task in self._pending_tasks:
                task._cancelled = True  # Never starts: its parent sees it done.
                task._tell_group()
            self._pending_tasks.clear()
            self._heap.clear()
            self._state.notify_all()

    def close(self) -> None:
        """Cancel every task, pending or running, and every event emitted after: `_aclose` in a task.

        Closed at once, it starts no task and emits no event more, though the
        tasks are cancelled once `_aclose` runs. Once closed, by it or by
        `events` ending, it does nothing.
        """
        if self._closed:
            return
        self._stopping = self._closed = True
        self._closer = asyncio.get_running_loop().create_task(self._aclose())

    async def _cancel_all(self) -> None:
        """Close: cancel every task, pending or running, and wake every waiter, `events` too."""
        self._stopping = self._closed = True
        await self._drop_pending()
        async with self._state:
            for task in self._running_tasks:
                task.cancel()
            self._state.notify_all()
        if self._events.empty():
            # Wakes events, if it waits for an event: none would come.
            self._events.put_nowait(_End())

    def _unread_failures(self) -> list[BaseException]:
        """The errors of the tasks given to events that `events` did not raise: each once.

        Those have no parent. Each failed, now or before with its failure
        unread, and is dropped from _initial as its error is taken.
        """
        errors: list[BaseException] = []
        for task in list(self._initial):
            if task.done():
                self._initial.discard(task)
                if not task.cancelled() and (error := task.exception()) is not None:
                    errors.append(error)
        return errors

    async def _aclose(self) -> None:
        """Cancel every task, pending or running, and wait for them to end: a parent waiting for one gets CancelledError.

        No event is emitted after, so `events` yields those emitted before,
        then ends. The scheduler is cancelled: none is needed once closed. The
        _aclose that `close` runs, and the task of `cancel`, are waited for.
        It raises, in an ExceptionGroup, the errors but cancellation of:

        - the tasks given to `events`, unread;
        - the tasks whose parent was cancelled, which raised none of them;
        - the scheduler, that _aclose and `cancel`'s task.

        Another task's error is its parent's.
        """
        await self._cancel_all()
        # asyncio.wait, never gather: an _aclose cancelled meanwhile cancels
        # none of them, and leaves their errors to the other _aclose.
        running = list(self._running_tasks)
        if running:  # asyncio.wait refuses none.
            await asyncio.wait([t._task for t in running if t._task is not None])
        errors = self._unread_failures()
        for task in running:
            # Read always, so that asyncio never warns of an exception not
            # retrieved.
            if task.cancelled() or (error := task.exception()) is None:
                continue
            parent = task._group.caller if task._group is not None else None
            if (
                parent is not None
                and parent.done()
                and parent.cancelled()
                and task not in self._orphans_raised
            ):
                # A parent cancelled raises CancelledError alone: its error
                # would be lost, e.g. a cleanup failing as it is cancelled.
                self._orphans_raised.add(task)
                errors.append(error)

        ends: list[asyncio.Task[Any]] = []
        if self._stopper is not None:
            ends.append(self._stopper)
        if self._scheduler is not None:
            self._scheduler.cancel()
            ends.append(self._scheduler)
        if self._closer is not None and self._closer is not asyncio.current_task():
            ends.append(self._closer)  # Waited for, not cancelled: its errors too.
        if ends:
            await asyncio.wait(ends)
        errors += [
            e for t in ends if not t.cancelled() and (e := t.exception()) is not None
        ]
        if errors:
            raise _Failures("The traversal failed to close.", errors)

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
                task = self._pop_leftmost()
                if not task.cancelled():
                    running = asyncio.create_task(task.run())
                    running.add_done_callback(partial(self._on_done, task))
                    task._set_task(running)
                    self._running_tasks.add(task)
                self._state.notify_all()  # Room for a task more.
        self._events.put_nowait(_End())

    def _pend(self, task: PathTask) -> None:
        self._pending_tasks.add(task)
        heapq.heappush(self._heap, (self.split(task.path), next(self._added), task))

    def _pop_leftmost(self) -> PathTask:
        """The pending task with the first path, dropped: in O(log n), skipping those no more pending."""
        while True:
            _, _, task = heapq.heappop(self._heap)
            if task in self._pending_tasks:
                self._pending_tasks.discard(task)
                return task

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
        # A done callback is synchronous: the lock is taken in a task of its
        # own. Kept by none: the lock it waits for holds it, until it ends.
        asyncio.get_running_loop().create_task(self._done(task))

    def _cancelled_before_start(self, task: PathTask) -> None:
        self._on_done(task, None)

    async def _done(self, task: PathTask) -> None:
        """A task is done, ended or cancelled before it started: forget it, and wake every waiter."""
        # Read always, so that asyncio never warns of an exception not retrieved.
        error = None if task.cancelled() else task.exception()
        if error is not None and task in self._initial and not self._closed:
            # Emitted while it still runs, so that the end never comes before it.
            self._events.put_nowait(_Failed(task, error))
        async with self._state:
            self._running_tasks.discard(task)
            self._pending_tasks.discard(task)  # Its heap entry is skipped.
            task._tell_group()
            self._state.notify_all()

    async def _add_task(self, group: PathTaskGroup, task: PathTask) -> None:
        """Add a task of a group to start, waiting for room: once stopping, cancel it instead."""
        async with self._state:
            if task._task is not None or task in self._pending_tasks:
                raise ValueError(f"Task {task.path!r} is added twice.")
            group.add(task)
            if task.done():
                task._tell_group()  # Cancelled already: nothing to run.
                return
            await self._state.wait_for(
                lambda: self._stopping or len(self._pending_tasks) < self.max_pending
            )
            if self._stopping:
                task._cancelled = True  # Never runs: its parent sees it done.
                task._tell_group()
            else:
                self._pend(task)
            self._state.notify_all()

    async def _run_tasks(
        self, caller: PathTask, tasks: Iterable[PathTask] | AsyncIterable[PathTask]
    ) -> list[Any]:
        """Run ``tasks`` for ``caller``, which waits: their results, in the order given.

        ``tasks`` may be an async iterator, e.g. made from a listing as it
        streams: each is taken once there is room for it among those pending,
        so that a wide directory's are never all in memory before they start.

        ``caller`` counts no more against ``max_running`` while it waits. Once
        all are done, it raises CancelledError if any was cancelled, else an
        ExceptionGroup of those that failed, in the order given. None is
        cancelled for another's failure. An error of ``tasks`` itself, e.g. of
        the listing, lets those taken end: it is raised alone where none
        failed, else first in their group.
        """
        async with self._state:
            if caller not in self._running_tasks or caller in self._waiting_tasks:
                raise ValueError(
                    f"Task {caller.path!r} is not running, or waits already."
                )
            self._waiting_tasks.add(caller)
            self._state.notify_all()
        group = PathTaskGroup(caller)
        added = group.tasks
        failed: Exception | None = None
        try:
            try:
                async for task in _aiter(tasks):
                    await self._add_task(group, task)
            except Exception as error:  # Never CancelledError: it propagates.
                failed = error
            async with self._state:
                # All taken: checked again, as done may hold for those
                # taken so far only, and all may be done already.
                group.on_task_done()
                await self._state.wait_for(group.done)
        finally:
            async with self._state:
                self._waiting_tasks.discard(caller)
                self._state.notify_all()

        if any(t.cancelled() for t in added):
            raise asyncio.CancelledError()
        errors: list[BaseException] = [
            e for t in added if (e := t.exception()) is not None
        ]
        if failed is not None:
            if not errors:
                raise failed
            errors.insert(0, failed)
        if errors:
            raise _Failures(f"{len(errors)} of {len(added)} tasks failed", errors)
        return [t.result() for t in added]

    async def _emit_event(self, event: Any) -> None:
        if self._closed:
            raise RuntimeError("The traversal is closed: no event is emitted.")
        async with self._state:
            await self._state.wait_for(
                lambda: self._closed or self._events.qsize() < self.max_events
            )
            if self._closed:
                raise RuntimeError("The traversal is closed: no event is emitted.")
            self._events.put_nowait(event)
