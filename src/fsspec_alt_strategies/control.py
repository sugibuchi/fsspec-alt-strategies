# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Observing and stopping strategies: callbacks, `Stop`, and results.

Every strategy reports its progress the same way. Its ``callback`` is called
with a state after each step, and may return a `Stop` to cancel it. The strategy
then returns a `Result`, or raises `OperationStopped` if the `Stop` asks for it.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

__all__ = ["Callback", "OperationStopped", "Result", "State", "Stop"]


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
        Why the operation was stopped, kept in the result for logs and messages.

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

Callback = Callable[[S], Stop | Awaitable[Stop | None] | None]
"""A strategy's callback: a plain or async function of its state.

It returns None to go on, or a `Stop`. It runs on the event loop, so it must
return quickly.
"""


@dataclass(kw_only=True)
class Result:
    """What a strategy did, even when stopped or when some paths failed.

    Each strategy's result adds its own fields, e.g. the files deleted.

    Attributes
    ----------
    stop : Stop, optional
        The `Stop` a callback returned, if any.
    errors : dict of str to Exception
        Each path that failed, with its error.
    """

    stop: Stop | None = None
    errors: dict[str, Exception] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether the operation ran to its end without failures."""
        return self.stop is None and not self.errors

    def raise_for_errors(self) -> None:
        """Raise the paths' errors, if any, as one exception.

        Raises
        ------
        ExceptionGroup
            Of the errors in `errors`.
        """
        if self.errors:
            raise ExceptionGroup(
                f"{len(self.errors)} path(s) failed", list(self.errors.values())
            )


class OperationStopped(Exception):
    """Raised by a strategy that a callback stopped with ``Stop(raise_error=True)``.

    When paths had failed, it is raised from their `ExceptionGroup`.

    Attributes
    ----------
    result : Result
        What the strategy did before it stopped.
    """

    def __init__(self, result: Result) -> None:
        reason = result.stop.reason if result.stop else None
        super().__init__(f"Stopped: {reason}" if reason else "Stopped")
        self.result = result


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


def _finish(result: Result) -> Result:
    """Return a strategy's result, or raise it, as its stop asks."""
    if result.stop is not None and result.stop.raise_error:
        cause = None
        if result.errors:
            cause = ExceptionGroup(
                f"{len(result.errors)} path(s) failed", list(result.errors.values())
            )
        raise OperationStopped(result) from cause
    return result
