# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The engine under the strategies that change a storage, such as _tree_rm.

They find what to change, e.g. by walking a tree or listing a prefix, and start
their jobs, e.g. deleting a file, as each item found arrives. `_Jobs` runs both
at once, each request holding a permit of one concurrency limit, and yields
what each step did. A family says where the items come from, in its
``source``. A strategy says what an item queues, how a job runs, and what its
outcome is.

`_CopyJobs` is the base of every copy's and move's jobs, whatever the
family: it places each path as fsspec's ``copy`` does. A family's copy joins
it with the family's jobs, e.g. ``_CopyOrMove(_CopyJobs, _TreeJobs)``: the
family finds what to copy, `_CopyJobs` where it goes. The two define no
method in common, and their constructors take keywords and pass the rest on,
so either order works.

This module depends on `filesystem` and `paths`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from contextlib import aclosing
from glob import has_magic
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

from fsspec.utils import common_prefix

from fsspec_alt_strategies.states import CpState

from .filesystem import _profile_of
from .paths import _trim

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
    from typing import TypeAlias

    from .operation import Operation

    _Transfer: TypeAlias = Callable[[str, str], Awaitable[None]]
    """How one file goes from a source path to a destination path.

    It is the source's ``_cp_file`` or ``_mv_file`` within one storage, or a
    router's copy or move of a file between two.
    """

_T = TypeVar("_T")


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
    matches.
    """

    folders: Sequence[str]
    files: Sequence[str] = ()


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

    return _Roots(list(folders), files)


class _Queue(Generic[_T]):
    """Items waiting for a permit of a concurrency limit, and the requests running.

    Each item's request starts once a permit is free, and gives it back once
    done. When none is free, a waiter task gets the next one: awaited, in
    `pending`, along with the owner's requests, it picks up the permits that
    others sharing the limit give back. The owner takes each finished
    request's item back with `pop`.
    """

    def __init__(self, limit: asyncio.Semaphore) -> None:
        self.limit = limit
        self.waiting: deque[_T] = deque()
        self.running: dict[asyncio.Task[Any], _T] = {}
        self.waiter: asyncio.Task[bool] | None = None

    async def start(self, run: Callable[[_T], Awaitable[Any]]) -> None:
        """Start the request of each item waiting, while permits are free."""
        while self.waiting and await self._take():
            item = self.waiting.popleft()
            task = asyncio.ensure_future(run(item))
            task.add_done_callback(lambda _: self.limit.release())
            self.running[task] = item

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
        return [*self.running, *([self.waiter] if self.waiting else [])]

    def pop(self, task: asyncio.Task[Any]) -> _T | None:
        """The item of a request done, or None for the waiter, whose permit `start` takes."""
        return self.running.pop(task, None)

    async def close(self) -> None:
        """Cancel the requests running, stop waiting, and give back a permit got."""
        tasks = list(self.running)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.running.clear()

        waiter, self.waiter = self.waiter, None
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            if not waiter.cancelled() and waiter.exception() is None:
                self.limit.release()


class _Jobs(ABC):
    """The jobs of one operation on ``fs``, run while what it changes is found.

    Each request holds a permit of the filesystem's ``operation_limit``. The
    constructor takes keywords only, and passes on what it does not take, so
    that a subclass may join a family's jobs with another base, such as a
    copy's.
    """

    def __init__(self, *, fs: Any, op: Operation[Any, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fs = fs

        # The source's requests, the jobs, and a pattern's glob share it.
        self.concurrency_limit: asyncio.Semaphore = fs.operation_limit
        self.jobs: _Queue[_Job] = _Queue(self.concurrency_limit)

        # The queue's own, which the subclasses queue their jobs in.
        self.waiting = self.jobs.waiting
        self.running = self.jobs.running

        self.op = op
        self.errors = 0
        self.current: str | None = None

        # The errors found in finding the roots, e.g. by a glob: yielded first.
        self.found: list[Exception] = []

        # The source's next item, awaited along with the jobs.
        self.next: asyncio.Task[Any] | None = None

    @abstractmethod
    async def prepare(self) -> _Roots | None:
        """What to walk, or None if there is nothing to do."""

    @abstractmethod
    def state(self) -> Any:
        """The operation's state now."""

    @abstractmethod
    async def glob_roots(self, pattern: str | Sequence[str]) -> _Roots:
        """The roots matching glob patterns, found with the family's glob.

        Its errors go to ``found``, yielded first.
        """

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
            if self.found:
                self.errors += len(self.found)
                self.op.state = self.state()
                for error in self.found:
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
            if self.next is not None:
                self.next.cancel()
                await asyncio.gather(self.next, return_exceptions=True)

            await self.jobs.close()
            if source is not None:
                await source.aclose()

    def pending(self) -> list[asyncio.Task[Any]]:
        """What to wait for: the jobs, the next item, and a permit if needed."""
        return [*self.jobs.pending(), *([self.next] if self.next else [])]

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
        """Start the jobs waiting, while permits are free, but none once cancelled."""
        if self.op.cancelling():
            self.stopping()
        await self.jobs.start(self.run)

    def finish(self, task: asyncio.Task[Any]) -> list[Any]:
        """Handle a job's task done, counting its errors, and return what to yield."""
        job = self.running.pop(task)
        self.current = job.source or job.destination

        try:
            task.result()
        except Exception as e:
            outcome = self.failed(job, e)
            self.errors += sum(isinstance(o, Exception) for o in outcome)
            return outcome

        return self.done(job)


class _CopyJobs(_Jobs):
    """The jobs of a copy or a move: where each path goes, onto ``dst_fs``.

    A path under ``prefix`` goes to the same place under ``base``. For a list,
    a path under a root of ``places`` goes there instead. `prepare` sets them,
    before `target` is used.

    The destination is described by ``dst_fs``, the filesystem copied
    to, and its profile, unknown if it has none. Within one storage, it is the
    source, and ``same_storage`` is True: a path copied onto itself, or under
    itself, is refused. Between two, a path may go to the same path, or under
    it, as that is another storage's.
    """

    verb = "copy"
    doing = "Copying"

    def __init__(
        self,
        *,
        path1: str | Sequence[str],
        path2: str,
        dst_fs: Any,
        same_storage: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.path1 = path1
        self.path2 = path2
        self.same_storage = same_storage

        # The profile is the destination's, as what a copy saves depends on
        # where it writes. Within one storage, it is the source's too.
        self.dst_fs = dst_fs
        self.profile = _profile_of(dst_fs)

        # Unknown, folders are made first. Implicit folders need none.
        self.implicit_folders = bool(self.profile.implicit_folders)
        self.parents_required = (
            self.profile.parents_required is not False and not self.implicit_folders
        )

        self.prefix = ""
        self.base = ""

        # For a list: where each root goes, by the root.
        self.places: dict[str, str] = {}

        # The destinations found missing, which need no asking again, e.g.
        # by a move renaming a root onto one.
        self.missing: set[str] = set()

        self.files_found = 0
        self.files_done = 0

    def state(self) -> Any:
        return CpState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_copied=self.files_done,
        )

    async def prepare(self) -> _Roots | None:
        """Set where the paths go, as fsspec's ``copy`` places them."""
        destination = _trim(self.path2)

        if not isinstance(self.path1, str):
            roots = await self._place(destination)
            sources = [*roots.folders, *roots.files]
        elif has_magic(self.path1):
            roots = await self.glob_roots(self.path1)
            sources = [*roots.folders, *roots.files]
            if not sources:
                return roots  # Only errors, in found, to yield.
            self.prefix = common_prefix(sources)
            if sources == list(roots.files) and len(sources) == 1:
                # A single file goes into the destination.
                self.prefix = self.prefix.rpartition("/")[0]
            self.base = destination
        else:
            root = _trim(self.path1)
            if root == destination and self.same_storage:
                return None
            roots = _Roots([root])
            sources = [root]
            self.prefix = root
            self.base = await self._base(root, destination)

        for source in sources if self.same_storage else ():
            target = self.target(source)
            if target == source and len(sources) == 1:
                return None
            if not source or target == source or target.startswith(source + "/"):
                raise ValueError(
                    f"Cannot {self.verb} {source!r} into itself: {target!r}"
                )

        return roots

    async def _place(self, destination: str) -> _Roots:
        """Place each root of a list into ``destination``, under its name.

        Two roots of the same name would go to the same place. As GNU's cp
        does, the first, in path order, goes, and each other is an error. That
        error is the root's own: what is in it is not copied either.
        """
        roots = await self.glob_roots(self.path1)
        folders = set(roots.folders)
        placed: dict[str, str] = {}

        for source in sorted([*roots.folders, *roots.files]):
            name = source.rpartition("/")[2]
            target = f"{destination}/{name}" if destination else name
            if (other := placed.get(target)) is not None:
                error = FileExistsError(target)
                error.add_note(
                    f"{self.doing} {source!r} to {target!r} skipped: {other!r} goes"
                    " there too."
                )
                self.found.append(error)
                continue

            placed[target] = source
            self.places[source] = target

        return _Roots(
            [s for s in self.places if s in folders],
            [s for s in self.places if s not in folders],
        )

    async def _base(self, root: str, destination: str) -> str:
        """Where ``root``, a single path, goes: into ``destination``, or onto it."""
        if self.path2.endswith("/"):
            into = True
        else:
            try:
                entry = await self.dst_fs._info(destination)
                into = entry["type"] == "directory"
            except FileNotFoundError:
                self.missing.add(destination)
                into = False

        if not into or self.path1.endswith("/"):
            return destination

        name = root.rsplit("/", 1)[-1]
        return f"{destination}/{name}" if destination else name

    def target(self, path: str) -> str:
        """Where a path goes."""
        if self.places:
            # The root it is under, found from the path up.
            root = path
            while root not in self.places and "/" in root:
                root = root.rpartition("/")[0]
            return self.places[root] + path[len(root) :]

        if self.prefix:
            return self.base + path[len(self.prefix) :]
        return f"{self.base}/{path}" if self.base else path

    async def copy_file(
        self, cp_file: _Transfer, source: str, destination: str
    ) -> None:
        """Copy a file by ``cp_file``, replacing a file there the destination keeps.

        A destination refusing to overwrite, as fsspec's copy does not, costs
        an ``_info``, an ``_rm_file`` and a copy more. A folder there is no file
        to replace: ``_rm_file`` takes files only. The job holds its permit
        already. An error raised notes its paths, as it is yielded among paths.
        """
        try:
            try:
                await cp_file(source, destination)
            except FileExistsError:
                entry = await self.dst_fs._info(destination)
                if entry["type"] == "directory":
                    raise

                await self.dst_fs._rm_file(destination)
                await cp_file(source, destination)
        except Exception as e:
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise
