# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A handler's clients: a sync client's calls in threads, and an async client on its loop.

`SyncClientSupport` serves a handler over a sync client, e.g. an SDK's or a
sync fsspec filesystem. It runs each blocking call in a thread of the
handler's own, as many as its requests at a time: the loop's default
executor, shared with all else, is then never taken up by one handler, nor
shared with another.

`AsyncClientSupport` serves a handler over an async client, e.g. an aiohttp
session or an aiobotocore client. Such a client belongs to the event loop it
was made on, and is closed on it. The mixin makes the clients on the running
loop when first needed, notes that loop, and makes them again on another. It
closes them on their own loop: awaited where that loop runs this, or else as
a task sent to it.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import weakref
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

__all__ = ["AsyncClientSupport", "SyncClientSupport"]

_T = TypeVar("_T")
_C = TypeVar("_C")


class SyncClientSupport:
    """A handler mixin running a sync client's blocking calls in the handler's own threads.

    ``_run_in_thread`` runs a call in one of the threads, made on first use.
    There are as many as ``_thread_count`` says: by default the handler's
    ``operation_concurrency``, if a number. A semaphore shared with others
    leaves the count to the executor's default.

    ``aclose()`` and ``close()`` shut the threads down, without waiting,
    cancelling the calls not started, then close as the next class does. So
    the handler's ``aclose()``, and the sync ``close()`` of a filesystem on
    it, shut them down, as does collecting the handler. A call after makes
    them again.

    Examples
    --------
    >>> class MyHandler(SyncClientSupport, TreeHandler):
    ...     async def info(self, entry, **kwargs):
    ...         return await self._run_in_thread(self.client.stat, entry["name"])
    """

    _executor: ThreadPoolExecutor | None = None

    def _thread_count(self) -> int | None:
        """How many threads to make: None leaves it to the executor's default."""
        concurrency = getattr(self, "operation_concurrency", None)
        return concurrency if isinstance(concurrency, int) else None

    async def _run_in_thread(
        self, func: Callable[..., _T], /, *args: Any, **kwargs: Any
    ) -> _T:
        """Run a blocking call in one of the handler's own threads, and await it.

        The call's context goes with it, e.g. the monitors of the operation it
        is part of.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._thread_pool(),
            contextvars.copy_context().run,
            functools.partial(func, *args, **kwargs),
        )

    def _thread_pool(self) -> ThreadPoolExecutor:
        """The handler's own threads, made on first use."""
        executor = self._executor
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=self._thread_count(),
                thread_name_prefix=type(self).__name__,
            )
            # Collecting the handler shuts the threads down, without the
            # finalizer keeping it alive.
            weakref.finalize(self, executor.shutdown, wait=False, cancel_futures=True)
            self._executor = executor
        return executor

    def _shutdown_threads(self) -> None:
        """Shut the threads down, without waiting: the calls not started are cancelled."""
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    async def aclose(self) -> None:
        """Shut the threads down, then close as the next class's `aclose` does, e.g. `StorageHandler`'s."""
        self._shutdown_threads()
        aclose = getattr(super(), "aclose", None)
        if aclose is not None:
            await aclose()

    def close(self) -> None:
        """Shut the threads down, then close as the next class's `close` does, e.g. `StorageHandler`'s."""
        self._shutdown_threads()
        close = getattr(super(), "close", None)
        if close is not None:
            close()


class AsyncClientSupport(Generic[_C]):
    """A handler mixin making an async client's clients on the running loop, and closing them on theirs.

    A subclass implements `create_clients`, which makes them, e.g. a session,
    and `close_clients`, which closes them. Its requests get them by
    `ensure_clients`:

    - the first call makes them on the running loop, and notes that loop.
      Calls at once on that loop wait for the one making them;
    - a call on another loop, e.g. after the first one ended, makes them again
      on it, and closes the old ones on theirs, as `aclose` does.

    `aclose` closes them, then closes as the next class does. Where their loop
    runs `aclose`, it awaits `close_clients`. Where it runs in another thread,
    e.g. fsspec's own loop, it sends it there as a task, and awaits it. Where
    their loop has ended, they are dropped: nothing can close them now. The
    sync `close` sends it as a task to their loop, without waiting, as it
    cannot await. Either way, a call after makes them again.

    A failed close is dropped, its error never logged: a storage's message
    may carry credentials.

    Examples
    --------
    >>> class MyHandler(AsyncClientSupport[aiohttp.ClientSession], TreeHandler):
    ...     async def create_clients(self):
    ...         return aiohttp.ClientSession()
    ...
    ...     async def close_clients(self, clients):
    ...         await clients.close()
    ...
    ...     async def info(self, entry, **kwargs):
    ...         session = await self.ensure_clients()
    ...         ...
    """

    _clients: _C | None = None
    _clients_loop: asyncio.AbstractEventLoop | None = None
    _clients_lock: asyncio.Lock | None = None
    _lock_loop: asyncio.AbstractEventLoop | None = None

    @abstractmethod
    async def create_clients(self) -> _C:
        """Make the clients on the running loop, e.g. a session: `ensure_clients` keeps them."""

    @abstractmethod
    async def close_clients(self, clients: _C) -> None:
        """Close clients `create_clients` made, on the loop they were made on."""

    async def ensure_clients(self) -> _C:
        """The clients of the running loop: made by `create_clients` the first time, and on another loop."""
        loop = asyncio.get_running_loop()
        clients = self._clients
        if clients is not None and self._clients_loop is loop:
            return clients
        async with self._lock(loop):
            clients = self._clients
            if clients is not None and self._clients_loop is loop:
                return clients  # Made meanwhile, by a call that held the lock.
            await self._drop_clients()
            clients = await self.create_clients()
            self._clients, self._clients_loop = clients, loop
            return clients

    def _lock(self, loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
        """A lock of ``loop``'s, for the calls making the clients on it: a lock serves one loop only."""
        lock = self._clients_lock
        if lock is None or self._lock_loop is not loop:
            lock = self._clients_lock = asyncio.Lock()
            self._lock_loop = loop
        return lock

    async def _drop_clients(self) -> None:
        """Forget the clients, and close them on their loop: awaited, there or from here."""
        clients, loop = self._clients, self._clients_loop
        self._clients = self._clients_loop = None
        if clients is None or loop is None:
            return
        if loop is asyncio.get_running_loop():
            await _quietly(self.close_clients(clients))
            return
        sent = _send(loop, self.close_clients(clients))
        if sent is not None:
            await asyncio.wrap_future(sent)

    async def aclose(self) -> None:
        """Close the clients on their loop, then close as the next class's `aclose` does, e.g. `StorageHandler`'s."""
        await self._drop_clients()
        aclose = getattr(super(), "aclose", None)
        if aclose is not None:
            await aclose()

    def close(self) -> None:
        """Send the clients' closing to their loop, without waiting, then close as the next class's `close` does."""
        clients, loop = self._clients, self._clients_loop
        self._clients = self._clients_loop = None
        if clients is not None and loop is not None:
            _send(loop, self.close_clients(clients))
        close = getattr(super(), "close", None)
        if close is not None:
            close()


async def _quietly(closing: Coroutine[Any, Any, None]) -> None:
    """Await a closing, dropping its failure: a storage's message may carry credentials."""
    # A failed close leaves nothing to do; CancelledError is no Exception, so it goes on.
    with contextlib.suppress(Exception):
        await closing


def _send(loop: asyncio.AbstractEventLoop, closing: Coroutine[Any, Any, None]) -> Any:
    """Run a closing on ``loop`` as a task: a future of it from another thread, else None.

    A loop that has ended runs nothing: the closing is dropped, unstarted.
    """
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if loop.is_closed() or (running is not loop and not loop.is_running()):
        closing.close()
        return None
    if running is loop:
        task = loop.create_task(_quietly(closing))
        _tasks.add(task)  # Kept until done: a loop holds its tasks weakly.
        task.add_done_callback(_tasks.discard)
        return None
    return asyncio.run_coroutine_threadsafe(_quietly(closing), loop)


_tasks: set[asyncio.Task[None]] = set()
"""The closings sent as tasks to the running loop, kept until done."""
