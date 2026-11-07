# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Operations: what the strategies return, to iterate, watch and stop.

A strategy returns an `Operation`. It does its work as it is iterated, and
yields what it did, path by path, mixed with the errors of the paths that
failed. The caller handles each, and can stop the operation as an asyncio task
is stopped: `Operation.cancel` asks it to stop, and `Operation.done` and
`Operation.cancelled` tell how it ended.

`collect` runs an operation to its end, for callers that want everything at
once.

`TreeFileSystem` and `FlatFileSystem` answer fsspec's methods, e.g. ``copy``,
by running an operation to its end, so their callers see only the answer. A
`Monitor` sees the operation itself, as it runs, within a `monitoring` block:

>>> class Progress(Monitor):
...     async def yielded(self, call, op, item):
...         print(call.method, op.state)
>>> with monitoring(Progress()):
...     fs.copy("data", "backup", recursive=True)

The block sets a context variable, so it holds for every call made within it,
by any code, including the sync methods, which fsspec runs on its own loop in
a copy of the caller's context. Tasks started within the block inherit it as
well. fsspec's own ``callback=`` is taken by ``copy``, ``mv`` and ``rm`` too,
as a monitor counting what was done.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from typing import Self

    from fsspec.asyn import AsyncFileSystem

__all__ = [
    "Call",
    "CallCancelled",
    "Monitor",
    "Operation",
    "State",
    "collect",
    "monitoring",
]


@dataclass(frozen=True, kw_only=True)
class State:
    """The progress of an operation, as `Operation.state` gives it.

    Each strategy's state adds its own counts, e.g. the files deleted.

    Attributes
    ----------
    errors : int
        The errors yielded so far.
    current : str, optional
        The path last worked on.
    """

    errors: int = 0
    current: str | None = None


T = TypeVar("T")
S = TypeVar("S", bound=State)


class Operation(Generic[T, S]):
    """A strategy's work, done as it is iterated, and stopped as a task is.

    Iterating it yields what it did, e.g. each path deleted, and the error of
    each path that failed, as an exception object. An error ending the whole
    operation, such as a missing root, is raised instead.

    It does its work only while iterated, a few requests ahead of the caller.
    So a slow loop body slows it, and an operation never iterated does nothing.

    Its methods follow `asyncio.Task`'s:

    - `cancel` asks it to stop. It starts no new request, but goes on yielding
      the outcomes of those already sent, then ends.
    - `cancelling` tells whether a stop was asked, and `cancelled` whether it
      ended so.
    - `done` tells whether it ended, whatever the reason.

    Leaving the loop early stops it at once, without yielding what is in
    flight. Use it as an async context manager, so that this happens on
    leaving the block: requests that only read are then cancelled, and those
    that change the storage, e.g. deletions, are waited for.

    Attributes
    ----------
    state : State
        Its progress so far. Each strategy's docstring names its state class.

    Examples
    --------
    >>> rm = tree_rm(
    ...     "data/2024",
    ...     ls=fs._ls,
    ...     info=fs._info,
    ...     rm_file=fs._rm_file,
    ...     rm_folder=fs._rmdir,
    ...     concurrency_limit=asyncio.Semaphore(32),
    ... )
    >>> async with rm as op:
    ...     async for path in op:
    ...         if isinstance(path, Exception):
    ...             op.cancel()  # Stop, but see what was in flight.
    ...             print("failed:", path)
    >>> op.cancelled(), op.state.files_deleted
    (True, 12)
    """

    def __init__(
        self,
        run: Callable[[Operation[T, S]], AsyncGenerator[T | Exception]],
        state: S,
    ) -> None:
        self.state = state
        self._run = run
        self._steps: AsyncGenerator[T | Exception] | None = None
        self._cancel_requests = 0
        self._done = False

    def __repr__(self) -> str:
        status = (
            "cancelled"
            if self.cancelled()
            else "done"
            if self._done
            else "cancelling"
            if self._cancel_requests
            else "pending"
        )
        return f"<{type(self).__name__} {status} {self.state}>"

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T | Exception:
        if self._done:
            raise StopAsyncIteration
        if self._steps is None:
            if self._cancel_requests:
                # As a task cancelled before it runs: nothing is sent.
                self._done = True
                raise StopAsyncIteration
            self._steps = self._run(self)
        try:
            return await anext(self._steps)
        except BaseException:
            # The end, an error ending the operation, or this task cancelled:
            # each finishes the steps, which are done.
            self._done = True
            raise

    def cancel(self) -> bool:
        """Ask the operation to stop.

        It starts no new request. Iterating it goes on with the outcomes of the
        requests already sent, then ends.

        Returns
        -------
        bool
            False if it is already done, as `asyncio.Task.cancel` returns.
        """
        if self._done:
            return False
        self._cancel_requests += 1
        return True

    def cancelling(self) -> int:
        """The number of stops asked, as `asyncio.Task.cancelling` counts them."""
        return self._cancel_requests

    def cancelled(self) -> bool:
        """Whether it ended after a stop was asked, or was closed before its end."""
        return self._done and self._cancel_requests > 0

    def done(self) -> bool:
        """Whether it ended: run to its end, stopped, closed, or failed."""
        return self._done

    async def aclose(self) -> None:
        """Stop at once, without yielding what is in flight.

        Requests that only read are cancelled. Those that change the storage,
        e.g. deletions, are waited for, as their outcome would otherwise be
        unknown.
        """
        if self._done:
            return
        self._cancel_requests += 1
        try:
            if self._steps is not None:
                await self._steps.aclose()
        finally:
            self._done = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


async def collect(operation: Operation[T, State]) -> tuple[list[T], list[Exception]]:
    """Run an operation to its end, and return all it yielded.

    Parameters
    ----------
    operation : Operation
        The operation, not iterated yet.

    Returns
    -------
    items : list
        What it yielded but errors, in the order yielded.
    errors : list of Exception
        The errors it yielded, in the order yielded.

    Examples
    --------
    >>> limit = asyncio.Semaphore(32)
    >>> find = tree_find("data", ls=fs._ls, info=fs._info, concurrency_limit=limit)
    >>> entries, errors = await collect(find)
    >>> if errors:
    ...     raise ExceptionGroup("find failed", errors)
    """
    items: list[T] = []
    errors: list[Exception] = []
    async with operation:
        async for item in operation:
            if isinstance(item, Exception):
                errors.append(item)
            else:
                items.append(item)
    return items, errors


@dataclass(frozen=True)
class Call:
    """The fsspec method an operation answers, as a `Monitor` is told it.

    Attributes
    ----------
    fs : AsyncFileSystem
        The filesystem called, e.g. to tell the storage of a call.
    method : str
        Its name, without the underscore of its async version, e.g. "copy".
    path : str or list of str
        The path it was given, as given, e.g. with the protocol.
    path2 : str, optional
        The second path, for ``copy`` and ``mv``.
    """

    fs: AsyncFileSystem
    method: str
    path: str | list[str]
    path2: str | list[str] | None = None


class CallCancelled(Exception):
    """The call ended early, as a `Monitor` cancelled its operation.

    fsspec's methods either finish or raise, so the call raises this. It is no
    `asyncio.CancelledError`: the task running the call was not cancelled.

    Attributes
    ----------
    call : Call
        The call cancelled.
    state : State
        The progress of its operation when it ended.
    """

    def __init__(self, call: Call, state: State) -> None:
        super().__init__(f"{call.method} of {call.path!r} was cancelled by a monitor")
        self.call = call
        self.state = state


class Monitor:
    """What watches the operations of the calls within a `monitoring` block.

    A subclass overrides the methods it needs: each does nothing here. Each is
    awaited where the operation runs, before it goes on: a slow monitor slows
    the call, as a slow loop body slows an operation. A monitor raising ends
    the call with its error.

    To stop the call, a monitor calls ``op.cancel()``: the operation yields
    what is in flight, then the call raises `CallCancelled`.

    The methods of the sync API's calls run on fsspec's own loop, in its
    thread: a monitor shared with other threads must allow for it.

    See Also
    --------
    monitoring : Where monitors watch.

    Examples
    --------
    >>> class StopOnError(Monitor):
    ...     async def yielded(self, call, op, item):
    ...         if isinstance(item, Exception):
    ...             op.cancel()
    """

    async def started(self, call: Call, op: Operation[Any, Any]) -> None:
        """Called before the operation sends its first request."""

    async def yielded(self, call: Call, op: Operation[Any, Any], item: Any) -> None:
        """Called with each item the operation yields, an error of a path too.

        ``op.state`` holds its progress, the item counted.
        """

    async def ended(
        self, call: Call, op: Operation[Any, Any], error: BaseException | None
    ) -> None:
        """Called once the operation ended, however it did.

        ``error`` is what ends the call, e.g. a missing root or `CallCancelled`,
        or None if it ran to its end. The errors of single paths it yielded
        were given to ``yielded`` already.
        """


_monitors: contextvars.ContextVar[tuple[Monitor, ...]] = contextvars.ContextVar(
    "fsspec_alt_strategies_monitors", default=()
)


class monitoring:
    """Have a monitor watch the operations of the calls made within the block.

    It works as ``with`` and as ``async with``, and blocks nest: the calls
    within an inner one are watched by the monitors of both, the outer first.

    Parameters
    ----------
    monitor : Monitor
        The monitor.

    See Also
    --------
    Monitor : What it is told.

    Examples
    --------
    >>> async with monitoring(StopOnError()):
    ...     await fs._rm("data/2024", recursive=True)
    """

    def __init__(self, monitor: Monitor) -> None:
        self.monitor = monitor
        self._token: contextvars.Token[tuple[Monitor, ...]] | None = None

    def __enter__(self) -> Self:
        self._token = _monitors.set((*_monitors.get(), self.monitor))
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._token is not None:
            _monitors.reset(self._token)
            self._token = None

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(self, *exc_info: object) -> None:
        self.__exit__(*exc_info)
