# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Observing and stopping strategies: callbacks and `Stop`.

Every strategy reports its progress the same way. Its ``callback`` is called
with a state after each step, and may return a `Stop` to cancel it. The strategy
then returns what it did so far, or raises `OperationStopped` if the `Stop` asks
for it.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

__all__ = ["Callback", "OperationStopped", "State", "Stop"]


@dataclass(frozen=True, kw_only=True)
class Stop:
    """What a callback returns to stop a strategy, with how to stop.

    A strategy stopped starts no new requests, and waits for those already sent.
    New options get defaults, so that callbacks keep working.

    Parameters
    ----------
    cleanup : bool
        Whether to undo what the operation did, where it can, e.g. delete the
        files a copy wrote. Each strategy's docstring says what it undoes. Work
        that cannot be undone, such as a deletion, is kept.
    raise_error : bool
        Whether the strategy raises `OperationStopped` once stopped. Otherwise,
        it returns its partial result.
    reason : str, optional
        Why the operation was stopped, for logs and messages.

    Examples
    --------
    >>> def on_progress(state):
    ...     if state.errors:
    ...         return Stop(raise_error=True, reason="fail fast")
    """

    cleanup: bool = False
    raise_error: bool = False
    reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class State:
    """The state of a strategy's operation, as its callback gets it.

    Each strategy's state adds its own counts, e.g. the files deleted.

    Attributes
    ----------
    errors : int
        The paths that have failed so far.
    current : str, optional
        The path last worked on.
    """

    errors: int = 0
    current: str | None = None


S = TypeVar("S", bound=State)
R = TypeVar("R")

Callback = Callable[[S], Stop | Awaitable[Stop | None] | None]
"""A strategy's callback: a plain or async function of its state.

It returns None to go on, or a `Stop`. It runs on the event loop, so it must
return quickly.
"""


class OperationStopped(Exception):
    """Raised by a strategy that a callback stopped with ``Stop(raise_error=True)``.

    When paths had failed, it is raised from their `ExceptionGroup`.

    Attributes
    ----------
    result : tuple
        What the strategy did before it stopped, as it would have returned it.
    stop : Stop
        The `Stop` the callback returned.
    """

    def __init__(self, result: Any, stop: Stop) -> None:
        reason = stop.reason
        super().__init__(f"Stopped: {reason}" if reason else "Stopped")
        self.result = result
        self.stop = stop


async def _report(callback: Callback[S] | None, state: S) -> Stop | None:
    """Call a strategy's callback, and check what it returned."""
    if callback is None:
        return None
    answer = callback(state)
    if inspect.isawaitable(answer):
        answer = await answer
    if answer is None or isinstance(answer, Stop):
        return answer
    # A callback returning True or False by mistake must not pass silently.
    raise TypeError(f"A callback must return None or a Stop, not {answer!r}")


def _finish(result: R, stop: Stop | None, errors: list[Exception]) -> R:
    """Return a strategy's result, or raise it, as its stop asks."""
    if stop is not None and stop.raise_error:
        cause = None
        if errors:
            cause = ExceptionGroup(f"{len(errors)} path(s) failed", errors)
        raise OperationStopped(result, stop) from cause
    return result
