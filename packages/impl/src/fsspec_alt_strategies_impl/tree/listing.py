# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive listing made of one listing per directory, several at a time.

A `TreeFileOperationHandler`'s recursive ``ils`` lists all below a directory,
depth first in preorder, which covers name order, where the profile's
``recursive_ls_depth_first_preorder`` declares it, else parents first. A storage with a recursive listing of its
own implements it so. One without has `_tree_ils`, its default, which lists a
directory per request by ``ils``. One whose own may be refused, as only
asking tells, falls back on it.

It yields the entries depth first, as a walk visits them: each directory's
entries in the order its listing gives them, and all below a subdirectory right
after its entry. So it lists the directories in that order too, but starts the
next ones ahead, as their entries arrive, so that their listings overlap. The
handler's ``operation_limit`` bounds how many run at once, and `_HELD` how
much they read ahead.

This module depends on `handler` and `paths` of `common`.
"""

from __future__ import annotations

import asyncio
import heapq
from collections import deque
from typing import TYPE_CHECKING, Any

from ..common.paths import _trim

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from ..common.handler import Entry, StorageHandler
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


_BATCH = 1000
"""The most entries of a directory read ahead of those yielded: a page's worth."""

_HELD = 32 * _BATCH
"""The entries read and not yielded, past which no listing ahead starts."""


def _tree_ils(
    handler: TreeFileOperationHandler, entry: Entry, **ils_kwargs: Any
) -> AsyncGenerator[dict[str, Any]]:
    """List all below a directory depth first, by one ``ils`` per directory.

    It yields what a recursive ``ils`` yields: each entry below ``entry``, at
    any depth, depth first. A directory's entries come in the order its
    ``ils`` gives them, each subdirectory's followed by all below it. A file
    lists itself, as ``ils`` lists it. A directory listed twice, e.g. as a
    prefix and as its marker object, is yielded twice, but listed once.

    It lists the directories that come next while it yields, the nearest
    first, as many at once as the handler's ``operation_limit`` lets it.
    Each such listing holds a permit while it reads a batch of entries, a
    page's worth: what it read waits for its turn, holding no permit. No
    listing ahead starts while `_HELD` entries wait so, which bounds the
    memory to that, plus a batch per listing in flight. Only the listings in
    flight count, not those waiting: a tree of small directories is listed as
    fast as the limit allows, whatever its size. The directory being yielded
    is listed under the permit its caller holds, as for any ``ils``. So it
    needs no permit more to go on.

    A directory below ``entry`` that fails to list raises its error, as the
    storage raised it, and ends the listing: a recursive listing cannot go on
    past a failure.
    A missing ``entry`` raises FileNotFoundError, when the iteration starts, as
    its own ``ils`` raises it.

    ``entry`` is the directory's entry, as its parent's listing gave it, or as
    the handler's ``entry`` makes it from a path: it is listed by it whole, so
    that a storage addressing items by id asks nothing again. ``ils_kwargs`` go
    to each ``ils``.
    """
    return _entries(handler, entry, ils_kwargs)


async def _entries(
    handler: TreeFileOperationHandler,
    entry: Entry,
    ils_kwargs: dict[str, Any],
) -> AsyncGenerator[dict[str, Any]]:
    """The entries `_tree_ils` yields, as its docstring says."""
    root = _trim(entry["name"])
    ahead = _Ahead(handler, ils_kwargs)
    # The directories being yielded, the root first: each is yielding one of the next.
    stack = [ahead.take(root, entry)]

    try:
        while stack:
            listing = stack[-1]
            item = await listing.next()
            if item is None:
                stack.pop()
                continue

            entry, subdirectory = item
            if entry["name"].rstrip("/") == listing.directory:
                # A file lists itself, as the root may. A directory may list its
                # own marker: it is no entry below it.
                if len(stack) == 1 and entry["type"] != "directory":
                    yield entry
                continue

            yield entry
            if subdirectory is not None:
                stack.append(ahead.take(subdirectory))
    finally:
        await ahead.close(stack)


class _Ahead:
    """The directories found but not yielded yet, some of them listed ahead."""

    def __init__(self, handler: StorageHandler, ils_kwargs: dict[str, Any]) -> None:
        self.handler = handler
        self.ils_kwargs = ils_kwargs
        self.found: dict[str, _DirListing] = {}
        # The directories found and not listed yet, the nearest first: depth
        # first, a path comes before another as its parts do.
        self.queue: list[tuple[list[str], str]] = []
        self.held = 0  # The entries read and not yielded yet.
        self.closed = False

    def add(self, directory: str, entry: Entry) -> None:
        """Note a subdirectory read in a listing, with its entry, to list it ahead."""
        self.found[directory] = _DirListing(self, directory, entry)
        heapq.heappush(self.queue, (directory.split("/"), directory))
        self.start()

    def start(self) -> None:
        """Start listing the nearest directories found, while few entries wait.

        Each listing waits for a permit of ``operation_limit``, so the limit
        bounds those in flight. Those waiting get theirs in the order they
        started, the nearest first.
        """
        while not self.closed and self.queue and self.held < _HELD:
            _, directory = heapq.heappop(self.queue)
            listing = self.found.get(directory)
            if listing is not None:  # Else taken already, to be yielded.
                listing.task = asyncio.ensure_future(listing.read_ahead())

    def yielded(self) -> None:
        """Note an entry yielded, which may let a listing ahead start."""
        self.held -= 1
        if self.held == _HELD - 1:
            self.start()

    def take(self, directory: str, entry: Entry | None = None) -> _DirListing:
        """The listing of the directory to yield next, listed ahead or not."""
        listing = self.found.pop(directory, None)
        if listing is None:
            return _DirListing(self, directory, entry)
        else:
            return listing

    async def close(self, stack: list[_DirListing]) -> None:
        """Stop the listings ahead, and close those of every directory left."""
        self.closed = True  # A listing cancelled below starts no other.
        listings = [*stack, *self.found.values()]
        tasks = [lst.task for lst in listings if lst.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for listing in listings:
            await listing.close()


class _DirListing:
    """A directory's ``ils``, read a batch at a time, ahead of its turn or in it."""

    def __init__(
        self, ahead: _Ahead, directory: str, entry: Entry | None = None
    ) -> None:
        self.ahead = ahead
        self.directory = directory
        # Its parent's listing gave it, e.g. with the storage's id: the root's is its path alone.
        self.entry = entry if entry is not None else ahead.handler.entry(directory)
        # Each entry read, with the path of a subdirectory it is the first entry of.
        self.entries: deque[tuple[dict[str, Any], str | None]] = deque()
        self.subdirectories: set[str] = set()
        self.iterator: AsyncIterator[dict[str, Any]] | None = None
        self.done = False
        self.error: Exception | None = None
        self.task: asyncio.Task[None] | None = None
        self.reading = False  # Whether the task ahead holds a permit, reading.

    async def next(self) -> tuple[dict[str, Any], str | None] | None:
        """The next entry, read under the caller's permit if need be, or None at the end."""
        while not self.entries:
            if self.task is not None:
                task, self.task = self.task, None
                if self.reading:
                    await task  # It reads a batch: wait for it.
                else:
                    # Still waiting for a permit: read under the caller's instead.
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            elif self.error is not None:
                raise self.error
            elif self.done:
                return None
            else:
                await self.read()
        self.ahead.yielded()
        return self.entries.popleft()

    async def read_ahead(self) -> None:
        """Read a batch ahead of the directory's turn, under a permit of its own."""
        async with self.ahead.handler.operation_limit:
            self.reading = True
            try:
                await self.read()
            finally:
                self.reading = False

    async def read(self) -> None:
        """Read a batch of entries, noting the subdirectories in it to list ahead."""
        if self.iterator is None:
            self.iterator = self.ahead.handler.ils(self.entry, **self.ahead.ils_kwargs)

        try:
            for _ in range(_BATCH):
                entry = await anext(self.iterator)
                self.ahead.held += 1
                self.entries.append((entry, self.subdirectory(entry)))
        except StopAsyncIteration:
            self.done = True
        except Exception as e:
            self.error = e

        if self.done or self.error is not None:
            await self.close()

    def subdirectory(self, entry: dict[str, Any]) -> str | None:
        """The entry's path, if a subdirectory not read before: it is listed once."""
        name = entry["name"].rstrip("/")
        if (
            entry["type"] != "directory"
            or name == self.directory
            or name in self.subdirectories
        ):
            return None
        self.subdirectories.add(name)
        self.ahead.add(name, {**entry, "name": name})
        return name

    async def close(self) -> None:
        iterator, self.iterator = self.iterator, None
        if iterator is not None and (aclose := getattr(iterator, "aclose", None)):
            await aclose()
