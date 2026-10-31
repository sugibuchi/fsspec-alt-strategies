# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The engine under the tree strategies that change a tree, such as tree_rm.

They walk the tree as tree_walk does, and start their jobs, e.g. deleting a
file, as each folder's listing arrives. `_Jobs` runs both at once, up to
``batch_size`` jobs at a time, and yields what each step did. A subclass says
what a listing queues, how a job runs, and what its outcome is.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, NamedTuple

from fsspec.asyn import _get_batch_size

from .walk import _listings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable

    from fsspec.asyn import AsyncFileSystem

    from ..control import Operation
    from .walk import _Listing


class _Job(NamedTuple):
    """A request to make: its kind, e.g. "copy", and the paths it is about."""

    kind: str
    source: str
    destination: str = ""


class _Jobs(ABC):
    """The jobs of one operation, run while its tree is walked."""

    def __init__(
        self, fs: AsyncFileSystem, batch_size: int | None, op: Operation[Any, Any]
    ) -> None:
        self.fs = fs
        self.batch_size = batch_size or fs.batch_size or _get_batch_size(nofiles=True)
        self.op = op
        self.errors = 0
        self.current: str | None = None
        # The paths in each folder listed that are not done yet, for strategies
        # finishing a folder after all in it, as a deletion does. A folder with
        # a failure in it never reaches zero, and nor do the folders above it.
        self.remaining: dict[str, int] = {}
        self.parents: dict[str, str] = {}
        self.waiting: deque[_Job] = deque()
        self.running: dict[asyncio.Task[Any], _Job] = {}
        # The walk's next listing, awaited along with the jobs.
        self.next: asyncio.Task[_Listing] | None = None

    @abstractmethod
    async def prepare(self) -> str | None:
        """The folder or file to walk, or None if there is nothing to do."""

    @abstractmethod
    def state(self) -> Any:
        """The operation's state now."""

    @abstractmethod
    def file_root(self, path: str) -> None:
        """Queue the jobs for ``path``, the operation's root, which is a file."""

    @abstractmethod
    def add(self, listing: _Listing) -> None:
        """Queue the jobs for a folder listed without error."""

    @abstractmethod
    def run(self, job: _Job) -> Awaitable[Any]:
        """The coroutine doing a job."""

    @abstractmethod
    def done(self, job: _Job) -> list[Any]:
        """Count a job done, queue what it allows, and return what to yield."""

    @abstractmethod
    def failed(self, job: _Job, error: Exception) -> list[Any]:
        """Handle a job's error, and return what to yield: the error, noted."""

    async def outcomes(self) -> AsyncGenerator[Any]:
        """Yield each outcome, as the operation does, keeping its state up to date."""
        path = await self.prepare()
        if path is None:
            return
        try:
            async with aclosing(self.steps(path)) as steps:
                async for outcomes in steps:
                    self.op.state = self.state()
                    for outcome in outcomes:
                        yield outcome
        finally:
            # Closing the steps waits for the jobs running, and counts them.
            self.op.state = self.state()

    async def steps(self, path: str) -> AsyncGenerator[list[Any]]:
        """Run the jobs, yielding what each step did: a listing, or a job done.

        Closing this stops the operation: it starts nothing new, and waits for
        the jobs running, as their outcome would otherwise be unknown.
        """
        listings = _listings(
            self.fs, path, batch_size=self.batch_size, stopping=self.op.cancelling
        )
        try:
            yield await self.begin(path, listings)
            while True:
                self.start()
                pending = [*self.running, *([self.next] if self.next else [])]
                if not pending:
                    break
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if (outcome := self.take(task, listings)) is not None:
                        yield outcome
        except GeneratorExit:
            self.waiting.clear()
            while self.running:
                done, _ = await asyncio.wait(self.running)
                for task in done:
                    self.finish(task)
            raise
        finally:
            # Listings are only reads, so cancelling them loses nothing. The
            # jobs are cancelled only when this task itself is.
            tasks = [*self.running, *([self.next] if self.next else [])]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await listings.aclose()

    async def begin(self, path: str, listings: AsyncGenerator[_Listing]) -> list[Any]:
        """List ``path``, or queue its own jobs if a file: the first step."""
        try:
            first = await anext(listings)
        except NotADirectoryError:
            self.file_root(self.fs._strip_protocol(path))
            return []
        self.next = asyncio.ensure_future(anext(listings))
        return self.listed(first)

    def take(
        self, task: asyncio.Task[Any], listings: AsyncGenerator[_Listing]
    ) -> list[Any] | None:
        """Handle a finished task, returning its step, or None if the walk ended."""
        if task is not self.next:
            return self.finish(task)
        self.next = None
        try:
            listing = task.result()
        except StopAsyncIteration:
            return None
        self.next = asyncio.ensure_future(anext(listings))
        return self.listed(listing)

    def listed(self, listing: _Listing) -> list[Any]:
        self.current = listing.folder
        if listing.error is not None:
            self.errors += 1
            return [listing.error]
        self.add(listing)
        return []

    def start(self) -> None:
        if self.op.cancelling():
            self.waiting.clear()
        while self.waiting and len(self.running) < self.batch_size:
            job = self.waiting.popleft()
            self.running[asyncio.ensure_future(self.run(job))] = job

    def finish(self, task: asyncio.Task[Any]) -> list[Any]:
        job = self.running.pop(task)
        self.current = job.source or job.destination
        try:
            task.result()
        except Exception as e:
            outcome = self.failed(job, e)
            self.errors += sum(isinstance(o, Exception) for o in outcome)
            return outcome
        return self.done(job)

    def track(self, listing: _Listing) -> None:
        """Count the paths in a listed folder, for `child_done`."""
        self.remaining[listing.folder] = len(listing.entries)
        for entry in listing.entries:
            self.parents[entry["name"].rstrip("/")] = listing.folder

    def child_done(self, path: str) -> str | None:
        """Count a path done, and return its folder if all in it is now done."""
        parent = self.parents.get(path)
        if parent is None:
            return None
        self.remaining[parent] -= 1
        return parent if self.remaining[parent] == 0 else None
