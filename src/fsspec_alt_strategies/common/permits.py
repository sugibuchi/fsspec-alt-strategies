# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""An operation's share of a concurrency limit, taken as its requests start.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable


class _Permits:
    """An operation's share of a concurrency limit, taken as its requests start.

    A request takes a permit to start, and gives it back once done. When none
    is free, `waiter` is a task getting the next one: awaited along with the
    operation's own requests, it picks up the permits that others sharing the
    limit give back.
    """

    def __init__(self, limit: asyncio.Semaphore) -> None:
        self.limit = limit
        self.waiting: asyncio.Task[bool] | None = None

    async def take(self) -> bool:
        """Take a permit if one is free now, or got by the waiter."""
        if self.waiting is not None:
            if not self.waiting.done():
                return False
            self.waiting = None  # Its permit is now this operation's.
            return True
        if self.limit.locked():
            return False
        await self.limit.acquire()  # A permit is free: this does not wait.
        return True

    def waiter(self) -> asyncio.Task[bool]:
        """A task done once the next permit is this operation's."""
        if self.waiting is None:
            self.waiting = asyncio.ensure_future(self.limit.acquire())
        return self.waiting

    def start(self, request: Awaitable[Any]) -> asyncio.Task[Any]:
        """Run a request with a permit taken, giving it back once done."""
        task = asyncio.ensure_future(request)
        task.add_done_callback(lambda _: self.limit.release())
        return task

    async def close(self) -> None:
        """Stop waiting for a permit, and give back one got but not used."""
        task, self.waiting = self.waiting, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if not task.cancelled() and task.exception() is None:
            self.limit.release()
