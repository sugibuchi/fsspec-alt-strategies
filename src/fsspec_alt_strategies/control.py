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

This module depends on nothing else in the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from typing import Self

__all__ = ["Operation", "State", "collect"]


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
    >>> async with tree_rm(fs, "data/2024") as op:
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
    >>> entries, errors = await collect(tree_find(fs, "data"))
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
