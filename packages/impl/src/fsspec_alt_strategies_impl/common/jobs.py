# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What the strategies changing a storage share: how each file goes, and requests waiting for permits.

`_transfer_of` and `_moves_of` say how a copy or a move sends each file. `_Queue`
holds items waiting for a permit of a concurrency limit, e.g. the ``info`` of
a glob's literal paths. `_check_batch_size` refuses a handler's batch size
below 1.

This module depends on `handler` and `transfer`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from .transfer import _across

if TYPE_CHECKING:
    from collections.abc import (
        Awaitable,
        Callable,
    )
    from typing import Protocol

    from .handler import StorageHandler

    class _Transfer(Protocol):
        """How one file goes from its source entry to a destination path.

        It is the source's ``cp_file`` or ``mv_file`` within one storage.
        Between two, a copy's is `transfer._across`, and a move's that copy,
        then the source's ``rm_file``, unless the caller gives its own. The entry is the
        source's, as its listing or ``info`` gave it, its ``"name"`` the source
        path. ``missing`` says nothing is at the destination, and ``parent``
        is the destination's parent's entry, as the primitives take them: one
        ignoring them is right all the same.
        """

        def __call__(
            self,
            entry: dict[str, Any],
            path2: str,
            *,
            missing: bool = False,
            parent: dict[str, Any] | None = None,
        ) -> Awaitable[None]: ...


_T = TypeVar("_T")


def _transfer_of(
    src: StorageHandler,
    dst: StorageHandler,
    cp_entry: _Transfer | None,
    same_storage: bool | None,
) -> tuple[_Transfer, bool]:
    """How a copy sends each file, and whether it stays in one storage.

    ``same_storage`` is whether ``dst`` is ``src``, unless given. Unless
    ``cp_entry`` is given, a file goes by the source's own ``cp_file`` within
    one storage, and by `transfer._across` between two, its data through the
    caller.
    """
    if same_storage is None:
        same_storage = src is dst
    if cp_entry is None:
        # A primitive takes a StoragePath, which the strategies pass: a _Transfer.
        cp_entry = cast("_Transfer", src.cp_file) if same_storage else _across(src, dst)
    return cp_entry, same_storage


def _moves_of(
    src: StorageHandler,
    dst: StorageHandler,
    cp_entry: _Transfer | None,
    mv_entry: _Transfer | None,
    same_storage: bool | None,
) -> tuple[_Transfer, _Transfer, bool]:
    """How a move sends each file, by copy and by move, and whether it stays in one storage.

    ``cp_entry`` is as `_transfer_of` makes it. ``mv_entry``, unless given, is
    the source's ``mv_file`` within one storage. Between two, it is
    ``cp_entry``, then the source's ``rm_file``: a file whose copy failed is
    kept. So a caller's copy, e.g. a cloud's copy from a URL, moves too.
    """
    cp_entry, same_storage = _transfer_of(src, dst, cp_entry, same_storage)
    if mv_entry is None:
        if same_storage:
            mv_entry = cast("_Transfer", src.mv_file)
        else:
            copy = cp_entry

            async def move(
                entry: dict[str, Any],
                path2: str,
                *,
                missing: bool = False,
                parent: dict[str, Any] | None = None,
            ) -> None:
                await copy(entry, path2, missing=missing, parent=parent)
                await src.rm_file(entry)

            mv_entry = move

    return cp_entry, mv_entry, same_storage


class _Queue(Generic[_T]):
    """Items waiting for a permit of a concurrency limit, and the requests running.

    Each item's request starts once a permit is free, and gives it back once
    done. When none is free, a waiter task gets the next one: awaited, in
    `pending`, along with the owner's requests, it picks up the permits that
    others sharing the limit give back. The owner takes each finished
    request's item back with `pop`.

    A request's permit is given back once, by whichever comes first: `pop`,
    or the task's done callback. The callback alone would be late: it runs a
    turn of the loop after the task ends, and `asyncio.wait` reports a task
    done before that. An operation ending on it would then end holding a
    permit. The callback still gives back that of a task nobody pops, e.g.
    one cancelled before it ran.
    """

    def __init__(self, limit: asyncio.Semaphore) -> None:
        self.limit = limit
        self.waiting: deque[_T] = deque()
        self.running: dict[asyncio.Task[Any], _T] = {}
        self.waiter: asyncio.Task[bool] | None = None
        # The requests whose permit is not given back yet.
        self.holding: set[asyncio.Task[Any]] = set()

    async def start(self, run: Callable[[_T], Awaitable[Any]]) -> None:
        """Start the request of each item waiting, while permits are free."""
        while self.waiting and await self._take():
            item = self.waiting.popleft()
            task = asyncio.ensure_future(run(item))
            self.holding.add(task)
            task.add_done_callback(self._give_back)
            self.running[task] = item

    def _give_back(self, task: asyncio.Task[Any]) -> None:
        """Give back a request's permit, unless it was already."""
        if task in self.holding:
            self.holding.remove(task)
            self.limit.release()

    async def _take(self) -> bool:
        """Take a permit if one is free now, or got by the waiter."""
        if self.waiter is not None:
            if not self.waiter.done():
                return False
            self.waiter = None  # Its permit is now this queue's.
            return True

        if self.limit.locked():
            return False
        await self.limit.acquire()  # A permit is free: this does not wait.
        return True

    def pending(self) -> list[asyncio.Task[Any]]:
        """The requests running, and the waiter for a permit if items wait."""
        if self.waiting and self.waiter is None:
            self.waiter = asyncio.ensure_future(self.limit.acquire())
        pending = list(self.running)
        if self.waiting and self.waiter is not None:
            pending.append(self.waiter)
        return pending

    def pop(self, task: asyncio.Task[Any]) -> _T | None:
        """The item of a request done, its permit given back, or None for the waiter, whose permit `start` takes."""
        self._give_back(task)
        return self.running.pop(task, None)

    async def close(self) -> None:
        """Cancel the requests running, stop waiting, and give back a permit got."""
        tasks = list(self.running)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            self._give_back(task)
        self.running.clear()

        waiter, self.waiter = self.waiter, None
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            if not waiter.cancelled() and waiter.exception() is None:
                self.limit.release()


def _check_batch_size(batch_size: int, name: str) -> None:
    """Refuse a handler's batch size below 1, e.g. its ``rm_batch_size``, as no batch could be sent."""
    if batch_size < 1:
        raise ValueError(f"{name} must be at least 1: {batch_size}")
