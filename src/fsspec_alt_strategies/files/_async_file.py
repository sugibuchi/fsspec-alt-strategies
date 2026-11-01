# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The machinery under the file objects: a loop, a concurrency limit, requests.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
import io
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsspec.asyn import sync

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Self

_MiB = 1024 * 1024


class _AsyncFile(io.RawIOBase):
    """A file whose requests run on an asyncio event loop, under a concurrency limit.

    Each sync method, such as ``read``, runs its async counterpart, such as
    ``_read``, on the loop and waits for it, as fsspec's filesystems do. So the
    sync methods cannot be called on the loop itself: code there awaits the
    async ones instead, through `files.fsspec_async.AsyncFileAdapter`.

    The async counterparts are ``_read``, ``_write``, ``_flush`` and
    ``aclose``. A file not supporting one keeps the default here, which raises
    `io.UnsupportedOperation`. ``seek`` and ``tell`` have none: as in fsspec's
    async files, they must send no request on the loop.

    ``flush`` is abstract, taking ``force`` as fsspec's files do: each class
    decides what flushing sends, and whether the file shows what was written
    before closing.

    A request holds the concurrency limit while it runs. Shared by files, the
    limit bounds their requests together, and so the memory these hold.

    Its parameters are the subclasses' first ones: ``path``, ``loop`` and
    ``concurrency_limit``.
    """

    default_concurrency_limit: int = 10
    """The requests at a time of a file given no concurrency limit."""

    mode: str
    """The mode, e.g. "rb", as fsspec's files have it."""

    def __init__(
        self,
        path: str,
        loop: asyncio.AbstractEventLoop,
        *,
        concurrency_limit: asyncio.Semaphore | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.loop = loop
        if concurrency_limit is None:
            concurrency_limit = asyncio.Semaphore(self.default_concurrency_limit)
        self._requests = _Requests(concurrency_limit)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.path!r}>"

    def _sync(self, func: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        return sync(self.loop, func, *args)

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError(f"I/O operation on closed file: {self.path}")

    def _unsupported(self, operation: str) -> io.UnsupportedOperation:
        return io.UnsupportedOperation(f"{operation} on {self.path}")

    async def _read(self, size: int = -1) -> bytes:
        """Read as ``read`` does."""
        raise self._unsupported("read")

    async def _write(self, data: Any) -> int:
        """Write as ``write`` does."""
        raise self._unsupported("write")

    @abstractmethod
    def flush(self, force: bool = False) -> None:
        """Send what the writes so far allow.

        ``force`` marks the last flush, before closing, as in fsspec's files.
        Whether the file then shows what was written, or only once closed, is
        the class's choice.
        """

    async def _flush(self, force: bool = False) -> None:
        """Flush as ``flush`` does: a class whose flush sends requests overrides it."""
        self.flush(force)

    async def _close(self) -> None:
        """Finish the file's requests: what a subclass adds to closing."""
        await self._requests.cancel_all()

    async def aclose(self) -> None:
        """Close the file, finishing or cancelling its requests."""
        if self.closed:
            return
        try:
            await self._close()
        finally:
            super().close()

    def close(self) -> None:
        if self.closed:
            return
        if getattr(self, "loop", None) is None:
            # The constructor failed before the loop was set: no request was sent.
            super().close()
            return
        self._sync(self.aclose)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


@dataclass(eq=False)
class _Range:
    """A request for the byte range [start, end) of a file."""

    start: int
    end: int
    task: asyncio.Task[Any]

    def succeeded(self) -> bool:
        return (
            self.task.done()
            and not self.task.cancelled()
            and self.task.exception() is None
        )


class _Requests:
    """A file's requests, in the order sent, each holding the limit while it runs."""

    def __init__(self, concurrency_limit: asyncio.Semaphore) -> None:
        self.concurrency_limit = concurrency_limit
        self.queue: deque[_Range] = deque()
        # The first request that failed, so that writes stop at once.
        self.error: BaseException | None = None

    async def send(
        self, start: int, end: int, func: Callable[..., Awaitable[Any]], *args: Any
    ) -> None:
        """Send a request once a slot is free: the caller waits for it."""
        await self.concurrency_limit.acquire()
        try:
            task = asyncio.ensure_future(func(*args))
        except BaseException:
            self.concurrency_limit.release()
            raise
        task.add_done_callback(self._done)
        self.queue.append(_Range(start, end, task))

    def _done(self, task: asyncio.Task[Any]) -> None:
        self.concurrency_limit.release()
        if not task.cancelled() and self.error is None:
            self.error = task.exception()

    async def cancel(self, ranges: list[_Range]) -> None:
        for r in ranges:
            r.task.cancel()
        await asyncio.gather(*(r.task for r in ranges), return_exceptions=True)

    async def cancel_all(self) -> None:
        ranges = list(self.queue)
        self.queue.clear()
        await self.cancel(ranges)
