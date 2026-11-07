# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""How an operation is run to its end, and watched by monitors meanwhile.

From what its callers use to what that builds on:

- `_collect` runs an operation to its end, returning all it yielded.
- `_collected` does so for an fsspec call, as the call's monitors watch it.
- `_watched` runs an operation for an fsspec call, yielding its items, telling
  the monitors of the `monitoring` blocks around it, and fsspec's ``callback=``.
- `_Counting` is the monitor fsspec's ``callback=`` is told through.

This module depends on nothing else in the package.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

# _monitors is control's, beside the monitoring blocks setting it.
from fsspec_alt_strategies_common.control import CallCancelled, Monitor, _monitors

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fsspec_alt_strategies_common.control import Call, Operation


async def _collect(
    items: Operation[Any, Any] | AsyncGenerator[Any],
) -> tuple[list[Any], list[Exception]]:
    """Run an operation to its end, and return what it yielded but errors, and its errors.

    Both lists are in the order yielded, which is the order the storage
    answered: callers sort, as fsspec's ``_find`` and ``_glob`` do.
    """
    found: list[Any] = []
    errors: list[Exception] = []
    async with aclosing(items):
        async for item in items:
            if isinstance(item, Exception):
                errors.append(item)
            else:
                found.append(item)
    return found, errors


async def _collected(
    operation: Operation[Any, Any], call: Call, callback: Any = None
) -> tuple[list[Any], list[Exception]]:
    """`_collect` of an operation, as its monitors watch it."""
    return await _collect(_watched(operation, call, callback))


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
