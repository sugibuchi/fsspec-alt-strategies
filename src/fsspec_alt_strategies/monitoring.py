# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Watching the operations behind fsspec's methods, without changing their API.

`TreeFileSystem` and `FlatFileSystem` answer fsspec's methods, e.g. ``copy``,
by running an `Operation` to its end. Its caller sees only the answer. A
`Monitor` sees the operation itself, as it runs, within a `monitoring` block:

>>> class Progress(Monitor):
...     async def yielded(self, call, op, item):
...         print(call.method, op.state)
>>> with monitoring(Progress()):
...     fs.copy("data", "backup", recursive=True)

The block sets a context variable, so it holds for every call made within it,
by any code, including the sync methods, which fsspec runs on its own loop in
a copy of the caller's context. Tasks started within the block inherit it as
well.

fsspec's own ``callback=``, an `fsspec.callbacks.Callback`, is taken by
``copy``, ``mv`` and ``rm`` too, as a monitor counting what was done.

This module depends on `control`.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from typing import Self

    from .control import Operation, State

__all__ = ["Call", "CallCancelled", "Monitor", "monitoring"]


@dataclass(frozen=True)
class Call:
    """The fsspec method an operation answers, as a `Monitor` is told it.

    Attributes
    ----------
    method : str
        Its name, without the underscore of its async version, e.g. "copy".
    path : str or list of str
        The path it was given, as given, e.g. with the protocol.
    path2 : str, optional
        The second path, for ``copy`` and ``mv``.
    """

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


class _Counting(Monitor):
    """fsspec's ``callback=``, told of each path done, as fsspec's ``put`` tells it.

    The total is not known ahead, as the operation lists while it works, so the
    callback's size stays as its caller set it.
    """

    def __init__(self, callback: Any) -> None:
        self.callback = callback

    async def yielded(self, call: Call, op: Operation[Any, Any], item: Any) -> None:
        if not isinstance(item, Exception):
            self.callback.relative_update(1)


async def _watched(
    operation: Operation[Any, Any], call: Call, callback: Any = None
) -> AsyncGenerator[Any]:
    """Run an operation for a call, yielding its items, as its monitors watch it.

    The monitors are those of the `monitoring` blocks around the call, and
    ``callback``, fsspec's, if given.
    """
    monitors = _monitors.get()
    if callback is not None:
        monitors = (*monitors, _Counting(callback))
    error: BaseException | None = None
    try:
        async with operation:
            for monitor in monitors:
                await monitor.started(call, operation)
            async for item in operation:
                for monitor in monitors:
                    await monitor.yielded(call, operation, item)
                yield item
        if operation.cancelled():
            raise CallCancelled(call, operation.state)
    except GeneratorExit:
        # The caller stopped early, e.g. a walk's: no error of the call.
        raise
    except BaseException as e:
        error = e
        raise
    finally:
        for monitor in monitors:
            await monitor.ended(call, operation, error)


async def _collected(
    operation: Operation[Any, Any], call: Call, callback: Any = None
) -> tuple[list[Any], list[Exception]]:
    """`collect` of an operation, as its monitors watch it."""
    items: list[Any] = []
    errors: list[Exception] = []
    async for item in _watched(operation, call, callback):
        if isinstance(item, Exception):
            errors.append(item)
        else:
            items.append(item)
    return items, errors
