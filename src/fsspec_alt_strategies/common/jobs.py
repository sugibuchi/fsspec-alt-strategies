# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The engine under the strategies that change a storage, such as tree_rm.

They find what to change, e.g. by walking a tree or listing a prefix, and start
their jobs, e.g. deleting a file, as each item found arrives. `_Jobs` runs both
at once, each request holding a permit of one concurrency limit, and yields
what each step did. A family says where the items come from, in its
``source``. A strategy says what an item queues, how a job runs, and what its
outcome is.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Sequence

    from ..control import Operation


class _Job(NamedTuple):
    """A request to make: its kind, e.g. "copy", and the paths it is about.

    ``paths`` are those of a request about several, e.g. a batch deletion.
    """

    kind: str
    source: str
    destination: str = ""
    paths: tuple[str, ...] = ()


class _Roots(NamedTuple):
    """Where an operation starts: what it walks, and what it handles alone.

    ``folders`` are walked, or listed, at once. A single path of unknown kind is
    among them, as its listing tells. ``files`` are known files, e.g. glob
    matches. ``errors`` are those found in finding the roots, yielded first.
    """

    folders: Sequence[str]
    files: Sequence[str] = ()
    errors: Sequence[Exception] = ()


def _roots_of(
    matches: list[dict[str, Any]],
    errors: list[Exception],
    pattern: str | Sequence[str],
) -> _Roots:
    """The roots of a glob's matches, as fsspec's ``_expand_path`` finds them.

    A match inside a folder that matches is left out: walking the folder finds
    it. Matching nothing raises FileNotFoundError, as fsspec does.
    """
    if not matches and not errors:
        raise FileNotFoundError(pattern)
    folders: dict[str, None] = {}
    files: list[str] = []
    # Sorted, a folder comes before all in it.
    for entry in sorted(matches, key=lambda e: e["name"]):
        name = entry["name"].rstrip("/")
        parts = name.split("/")
        if any("/".join(parts[:i]) in folders for i in range(1, len(parts))):
            continue
        if entry["type"] == "directory":
            folders[name] = None
        else:
            files.append(name)
    return _Roots(list(folders), files, errors)


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


class _Jobs(ABC):
    """The jobs of one operation, run while what it changes is found."""

    def __init__(
        self,
        concurrency_limit: asyncio.Semaphore,
        op: Operation[Any, Any],
    ) -> None:
        # The source's requests, the jobs, and a pattern's glob share it.
        self.concurrency_limit = concurrency_limit
        self.permits = _Permits(self.concurrency_limit)
        self.op = op
        self.errors = 0
        self.current: str | None = None
        self.waiting: deque[_Job] = deque()
        self.running: dict[asyncio.Task[Any], _Job] = {}
        # The source's next item, awaited along with the jobs.
        self.next: asyncio.Task[Any] | None = None

    @abstractmethod
    async def prepare(self) -> _Roots | None:
        """What to walk, or None if there is nothing to do."""

    @abstractmethod
    def state(self) -> Any:
        """The operation's state now."""

    @abstractmethod
    def source(self, roots: _Roots) -> AsyncGenerator[Any] | None:
        """The items found in the root folders, if any: e.g. their listings."""

    @abstractmethod
    def fed(self, item: Any) -> list[Any]:
        """Queue the jobs for an item of the source, and return what to yield."""

    @abstractmethod
    def file_root(self, path: str) -> None:
        """Queue the jobs for ``path``, a root of the operation, which is a file."""

    @abstractmethod
    def run(self, job: _Job) -> Awaitable[Any]:
        """The coroutine doing a job."""

    @abstractmethod
    def done(self, job: _Job) -> list[Any]:
        """Count a job done, queue what it allows, and return what to yield."""

    @abstractmethod
    def failed(self, job: _Job, error: Exception) -> list[Any]:
        """Handle a job's error, and return what to yield: the error, noted."""

    # A hook, not abstract: most strategies queue nothing at the source's end.
    def exhausted(self) -> None:  # noqa: B027
        """Queue what waits for the source's end, e.g. a last batch."""

    async def outcomes(self) -> AsyncGenerator[Any]:
        """Yield each outcome, as the operation does, keeping its state up to date."""
        roots = await self.prepare()
        if roots is None:
            return
        try:
            if roots.errors:
                self.errors += len(roots.errors)
                self.op.state = self.state()
                for error in roots.errors:
                    yield error
            async with aclosing(self.steps(roots)) as steps:
                async for outcomes in steps:
                    self.op.state = self.state()
                    for outcome in outcomes:
                        yield outcome
        finally:
            # Closing the steps waits for the jobs running, and counts them.
            self.op.state = self.state()

    async def steps(self, roots: _Roots) -> AsyncGenerator[list[Any]]:
        """Run the jobs, yielding what each step did: an item, or a job done.

        Closing this stops the operation: it starts nothing new, and waits for
        the jobs running, as their outcome would otherwise be unknown.
        """
        source = self.source(roots)
        try:
            yield await self.begin(roots, source)
            while True:
                await self.start()
                if not (pending := self.pending()):
                    break
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if (outcome := self.take(task, source)) is not None:
                        yield outcome
        except GeneratorExit:
            self.waiting.clear()
            while self.running:
                done, _ = await asyncio.wait(self.running)
                for task in done:
                    self.finish(task)
            raise
        finally:
            # The source only reads, so cancelling it loses nothing. The jobs
            # are cancelled only when this task itself is.
            tasks = [*self.running, *([self.next] if self.next else [])]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.permits.close()
            if source is not None:
                await source.aclose()

    def pending(self) -> list[asyncio.Task[Any]]:
        """What to wait for: the jobs, the next item, and a permit if needed."""
        pending = [*self.running, *([self.next] if self.next else [])]
        if self.waiting:
            pending.append(self.permits.waiter())
        return pending

    async def begin(
        self, roots: _Roots, source: AsyncGenerator[Any] | None
    ) -> list[Any]:
        """Queue the files' jobs, and take the source's first item: the first step."""
        for path in roots.files:
            self.file_root(path)
        if source is None:
            self.exhausted()
            return []
        try:
            first = await anext(source)
        except StopAsyncIteration:
            self.exhausted()
            return []
        self.next = asyncio.ensure_future(anext(source))
        return self.fed(first)

    def take(
        self, task: asyncio.Task[Any], source: AsyncGenerator[Any] | None
    ) -> list[Any] | None:
        """Handle a finished task, returning its step, or None if it has none."""
        if task in self.running:
            return self.finish(task)
        if task is not self.next or source is None:
            return None  # The waiter's permit, which start takes.
        self.next = None
        try:
            item = task.result()
        except StopAsyncIteration:
            self.exhausted()
            return None
        self.next = asyncio.ensure_future(anext(source))
        return self.fed(item)

    def stopping(self) -> None:
        """Drop the jobs waiting, as the operation is cancelled.

        A subclass may keep those finishing what was done, e.g. the deletions
        of files a move has copied.
        """
        self.waiting.clear()

    async def start(self) -> None:
        if self.op.cancelling():
            self.stopping()
        while self.waiting and await self.permits.take():
            job = self.waiting.popleft()
            self.running[self.permits.start(self.run(job))] = job

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
