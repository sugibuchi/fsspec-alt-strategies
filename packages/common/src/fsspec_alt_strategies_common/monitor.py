# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What a caller sees of a strategy's work: its state, its control, and monitors.

A strategy returns an operation, which does its work as it is iterated. Its
iteration belongs to the strategies' package, copied into a project. What any
caller sees of it, whatever copy it comes from, is here:

- `State`, its progress so far;
- `OperationHandler`, to stop it as an asyncio task is stopped, and tell how
  it ended;
- `Monitored`, the marker of a filesystem whose calls monitors watch.

`TreeFileSystem` and `FlatFileSystem` answer fsspec's methods, e.g. ``copy``,
by running an operation to its end, so their callers see only the answer. A
`Monitor` sees the operation itself, as it runs, within a `monitoring` block.
It is told the `Call` the operation answers, and a call it stops raises
`CallCancelled`:

>>> class Progress(Monitor):
...     async def yielded(self, call, item, state, op):
...         print(call.method, state)
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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Self

    from fsspec.spec import AbstractFileSystem

__all__ = [
    "Call",
    "CallCancelled",
    "Monitor",
    "Monitored",
    "OperationHandler",
    "State",
    "current_monitors",
    "monitoring",
]


@dataclass(frozen=True, kw_only=True)
class State:
    """The progress of an operation.

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


class Monitored(ABC):  # noqa: B024, a marker: it declares no method.
    """A filesystem whose fsspec methods tell the monitors of their operations.

    A marker: the filesystem classes of the strategies, and their wrappers,
    inherit it. A filesystem without it is wrapped to be monitored.
    """


class OperationHandler(ABC):
    """The control of an operation: stopping it, and telling how it ended.

    Its methods follow `asyncio.Task`'s:

    - `cancel` asks it to stop. It starts no new request, but goes on with the
      outcomes of those already sent, then ends.
    - `cancelling` tells whether a stop was asked, and `cancelled` whether it
      ended so.
    - `done` tells whether it ended, whatever the reason.

    The operations of the strategies implement it. A `Monitor` is given one,
    to stop the operation it watches.
    """

    @abstractmethod
    def cancel(self) -> bool:
        """Ask the operation to stop.

        It starts no new request. It goes on with the outcomes of the requests
        already sent, then ends.

        Returns
        -------
        bool
            False if it is already done, as `asyncio.Task.cancel` returns.
        """

    @abstractmethod
    def cancelling(self) -> int:
        """The number of stops asked, as `asyncio.Task.cancelling` counts them."""

    @abstractmethod
    def cancelled(self) -> bool:
        """Whether it ended after a stop was asked, or was closed before its end."""

    @abstractmethod
    def done(self) -> bool:
        """Whether it ended: run to its end, stopped, closed, or failed."""


@dataclass(frozen=True)
class Call:
    """The fsspec method an operation answers, as a `Monitor` is told it.

    Attributes
    ----------
    fs : AbstractFileSystem
        The filesystem called, e.g. to tell the storage of a call. A call to a
        wrapper names the filesystem it wraps, which its callers know.
    method : str
        Its name, without the underscore of its async version, e.g. "copy".
    path : str or list of str
        The path it was given, as given, e.g. with the protocol.
    path2 : str, optional
        The second path, for ``copy`` and ``mv``.
    """

    fs: AbstractFileSystem
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
    given the call, then what happened, e.g. the item yielded, then the
    operation's state at that moment, then its handler, to act on it. Each is
    awaited where the operation runs, before it goes on: a slow monitor slows
    the call, as a slow loop body slows an operation. A monitor raising ends
    the call with its error. If one's `ended` raises, the monitors after it
    are not told that the call ended.

    To stop the call, a monitor calls ``op.cancel()``: the operation goes on
    with what is in flight, then the call raises `CallCancelled`.

    The methods of the sync API's calls run on fsspec's own loop, in its
    thread: a monitor shared with other threads must allow for it.

    See Also
    --------
    monitoring : Where monitors watch.

    Examples
    --------
    >>> class StopOnError(Monitor):
    ...     async def yielded(self, call, item, state, op):
    ...         if isinstance(item, Exception):
    ...             op.cancel()
    """

    async def started(self, call: Call, state: State, op: OperationHandler) -> None:
        """Called before the operation sends its first request."""

    async def yielded(
        self, call: Call, item: Any, state: State, op: OperationHandler
    ) -> None:
        """Called with each item the operation yields, an error of a path too.

        ``state`` is its progress, the item counted.
        """

    async def ended(
        self,
        call: Call,
        error: BaseException | None,
        state: State,
        op: OperationHandler,
    ) -> None:
        """Called once the operation ended, however it did.

        ``error`` is what ends the call, e.g. a missing root or `CallCancelled`,
        or None if it ran to its end. The errors of single paths it yielded
        were given to ``yielded`` already.
        """


_monitors: contextvars.ContextVar[tuple[Monitor, ...]] = contextvars.ContextVar(
    "fsspec_alt_strategies_monitors", default=()
)


def current_monitors() -> tuple[Monitor, ...]:
    """The monitors of the `monitoring` blocks around the caller, the outer first.

    The strategies' filesystems tell them of each operation they run. It is
    how a copy of the strategies, of any version, reads them: the context
    variable holding them stays this module's own.

    Returns
    -------
    tuple of Monitor
        The monitors, none outside any block.
    """
    return _monitors.get()


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
