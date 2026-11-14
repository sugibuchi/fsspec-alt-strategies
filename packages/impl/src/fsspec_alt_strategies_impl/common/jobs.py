# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The engine under the strategies that change a storage, such as ``_tree_rm``.

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

This module depends on `handler`, `operation`, `paths` and `transfer`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from contextlib import aclosing
from glob import has_magic
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, cast

from fsspec.utils import common_prefix

from fsspec_alt_strategies.profile import UNKNOWN
from fsspec_alt_strategies.states import CpState

from .handler import FileTransfer, StoragePath
from .operation import Operation, _collect_with_errors
from .paths import _trim
from .transfer import _across

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Awaitable,
        Callable,
        Iterable,
        Sequence,
    )
    from typing import Protocol

    from .handler import Entry, FileOperationHandler, StorageHandler

    class _Transfer(Protocol):
        """How one file goes from its source entry to a destination path.

        It is the source's ``cp_file`` or ``mv_file`` within one storage, and
        `transfer._across`, or its caller's own, between two. The entry is the
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


class _Job(NamedTuple):
    """A request to make: its kind, e.g. "copy", its source, and where it goes.

    ``source`` is the source's entry, as its listing gave it, so that a copy or
    a move hands it on: at least its ``"name"``, all a path known otherwise
    has. ``paths`` are those of a request about several, e.g. a batch deletion.
    """

    kind: str
    source: dict[str, Any]
    destination: StoragePath = StoragePath("")
    paths: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        """The source's path."""
        return self.source["name"]  # type: ignore[no-any-return]


class _Roots(NamedTuple):
    """Where an operation starts: what it walks, and what it handles alone.

    Each root is its entry, so that a copy or move hands it on without asking
    it again. ``directories`` are walked, or listed, at once. A single path of
    unknown kind is among them, its entry its ``"name"`` alone, as its listing
    tells what it is. ``files`` are known files, e.g. glob matches.
    """

    directories: Sequence[dict[str, Any]]
    files: Sequence[dict[str, Any]] = ()

    @property
    def paths(self) -> list[str]:
        """The paths of all the roots, the directories first."""
        return [e["name"] for e in (*self.directories, *self.files)]


def _roots_of(
    matches: list[dict[str, Any]],
    errors: list[Exception],
    pattern: str | Sequence[str],
) -> _Roots:
    """The roots of a glob's matches, as fsspec's ``_expand_path`` finds them.

    A match inside a directory that matches is left out: walking the directory
    finds it. Matching nothing raises FileNotFoundError, as fsspec does.
    """
    if not matches and not errors:
        raise FileNotFoundError(pattern)

    directories: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []

    # Sorted, a directory comes before all in it.
    for entry in sorted(matches, key=lambda e: e["name"]):
        name = entry["name"].rstrip("/")
        parts = name.split("/")
        if any("/".join(parts[:i]) in directories for i in range(1, len(parts))):
            continue
        if entry["type"] == "directory":
            directories[name] = {**entry, "name": name}
        else:
            files.append({**entry, "name": name})

    return _Roots(list(directories.values()), files)


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
        pending = list(self.running)
        if self.waiting and self.waiter is not None:
            pending.append(self.waiter)
        return pending

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


def _check_batch_size(batch_size: int, name: str = "rm_batch_size") -> None:
    """Refuse a handler's batch size below 1, e.g. its ``rm_batch_size``, as no batch could be sent."""
    if batch_size < 1:
        raise ValueError(f"{name} must be at least 1: {batch_size}")


def _keys_of(paths: tuple[str, ...]) -> str:
    """A batch's paths, for its error's note: all of a few, or the first and last."""
    if len(paths) <= 3:
        return ", ".join(map(repr, paths))
    else:
        return f"{len(paths)} keys, from {paths[0]!r} to {paths[-1]!r}"


class _Jobs(ABC):
    """The jobs of one operation on ``handler``, run while what it changes is found.

    Each request holds a permit of the filesystem's ``operation_limit``. The
    constructor takes keywords only, and passes on what it does not take, so
    that a subclass may join a family's jobs with another base, such as a
    copy's.

    A batch deletion, queued by `send_batch`, runs the handler's ``rm_files``
    operation, by `delete`, and its outcome is told by `deleted`: the paths it
    deleted, and the errors of those it did not. A batch holds no permit of
    the limit: its operation's requests take theirs, which a permit held
    around them would wait for.
    """

    def __init__(
        self,
        *,
        handler: FileOperationHandler,
        op: Operation[Any, Any],
        maxdepth: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.handler: FileOperationHandler = handler
        self.maxdepth = maxdepth

        # The options of each listing the jobs make, a pattern's glob too.
        self.ils_kwargs: dict[str, Any] = ils_kwargs or {}

        # Each batch's outcome, by its number: the paths deleted, and the errors.
        self._deleted: dict[int, tuple[list[str], list[Exception]]] = {}

        # The batches to send, e.g. deletions, and those sent, which hold no permit.
        self.deletions: deque[_Job] = deque()
        self.deleting: dict[asyncio.Task[Any], _Job] = {}

        # The entries of each batch to delete, by its number, as the listing
        # gave them: a file is deleted by its entry, never by its name verified
        # again, nor by another file's of the same name.
        self.batched: dict[int, list[Entry]] = {}
        self.batches_sent = 0

        # The directory roots a pattern matched: fsspec walks them one level less
        # deep than a path named, as its glob counted that level already.
        self.matched: set[str] = set()

        # The source's requests, the jobs, and a pattern's glob share it.
        self.concurrency_limit: asyncio.Semaphore = self.handler.operation_limit
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

    def send_batch(self, entries: list[Entry]) -> None:
        """Queue a batch of files, by their entries as the listing gave them, to be deleted by the handler's ``rm_files``.

        The batch is numbered, and keeps its entries, never their names
        alone: two files listed under one name, e.g. on Google Drive, are two
        entries, and may be in two batches of the same paths.
        """
        self.batches_sent += 1
        self.batched[self.batches_sent] = entries
        paths = tuple(e["name"] for e in entries)
        source = {"name": paths[-1], "batch": self.batches_sent}
        self.deletions.append(_Job("delete", source, paths=paths))

    async def delete(self, job: _Job) -> None:
        """Delete a batch by the handler's ``rm_files`` operation, by its entries, keeping its outcome for `deleted`.

        The operation is iterated to its end, as the batch is one job. An
        error it raises fails the batch whole.
        """
        number = job.source["batch"]
        entries = self.batched.pop(number)
        deleted: list[str] = []
        errors: list[Exception] = []
        async with self.handler.rm_files(entries) as operation:
            async for outcome in operation:
                if isinstance(outcome, Exception):
                    errors.append(outcome)
                else:
                    deleted.append(outcome)
        self._deleted[number] = (deleted, errors)

    def deleted(self, job: _Job) -> tuple[list[str], list[Exception]]:
        """A batch's paths deleted, and the errors of those it did not delete, counted.

        A path neither deleted nor failed, e.g. dropped by a cancelled
        operation, is in neither.
        """
        deleted, errors = self._deleted.pop(job.source["batch"], ([], []))
        self.errors += len(errors)
        return deleted, errors

    @classmethod
    def operation(cls, **kwargs: Any) -> Operation[Any, Any]:
        """An operation running new jobs of this class, made at once with ``kwargs``.

        Made at once, the jobs give the operation its first state, e.g. whether
        a move renames directories, as they read it from the profile once.
        """
        jobs: Any = None
        op: Operation[Any, Any] = Operation(
            lambda _: jobs.outcomes(),
            None,  # type: ignore[arg-type]  # The jobs' own, set below.
        )
        jobs = cls(op=op, **kwargs)
        op.state = jobs.state()
        return op

    # A hook, not abstract: only the jobs reading a profile await it.
    async def setup(self) -> None:  # noqa: B027
        """Read what the jobs need of the storage before preparing, e.g. its profile."""

    @abstractmethod
    async def prepare(self) -> _Roots | None:
        """What to walk, or None if there is nothing to do."""

    @abstractmethod
    def state(self) -> Any:
        """The operation's state now."""

    @abstractmethod
    def glob(self, pattern: StoragePath | Sequence[StoragePath]) -> Operation[Any, Any]:
        """The family's glob of ``pattern`` on ``handler``, within ``maxdepth``."""

    async def glob_roots(self, pattern: StoragePath | Sequence[StoragePath]) -> _Roots:
        """The roots matching glob patterns, noting those a pattern matched.

        The glob's errors go to ``found``, yielded first.
        """
        matches, errors = await _collect_with_errors(self.glob(pattern))
        roots = _roots_of(matches, errors, pattern)
        self.found.extend(errors)

        if self.maxdepth is not None:
            named = [pattern] if isinstance(pattern, str) else pattern
            literal = {_trim(p) for p in named if not has_magic(p)}
            self.matched = {
                d["name"] for d in roots.directories if d["name"] not in literal
            }
        return roots

    @abstractmethod
    def source(self, roots: _Roots) -> AsyncGenerator[Any] | None:
        """The items found in the root directories, if any: e.g. their listings."""

    @abstractmethod
    def fed(self, item: Any) -> list[Any]:
        """Queue the jobs for an item of the source, and return what to yield."""

    @abstractmethod
    def file_root(self, entry: dict[str, Any]) -> None:
        """Queue the jobs for a root of the operation which is a file, by its entry."""

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
        await self.setup()
        self.op.state = self.state()
        roots = await self.prepare()

        try:
            # Yielded even when there is nothing to do: e.g. a list copied onto
            # itself but for a root whose name another one took.
            if self.found:
                self.errors += len(self.found)
                self.op.state = self.state()
                for error in self.found:
                    yield error

            if roots is None:
                return

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
            self.deletions.clear()
            while self.running or self.deleting:
                done, _ = await asyncio.wait([*self.running, *self.deleting])
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
            await self._cancel_batches()
            if source is not None:
                await source.aclose()

    async def _cancel_batches(self) -> None:
        """Cancel the batches deleting: left only by an error, as the jobs are cancelled then."""
        for task in self.deleting:
            task.cancel()
        await asyncio.gather(*self.deleting, return_exceptions=True)
        self.deleting.clear()

    def pending(self) -> list[asyncio.Task[Any]]:
        """What to wait for: the jobs, the next item, and a permit if needed."""
        return [
            *self.jobs.pending(),
            *self.deleting,
            *([self.next] if self.next else []),
        ]

    async def begin(
        self, roots: _Roots, source: AsyncGenerator[Any] | None
    ) -> list[Any]:
        """Queue the files' jobs, and take the source's first item: the first step."""
        for entry in roots.files:
            self.file_root(entry)

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
        if task in self.running or task in self.deleting:
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
        """Drop the jobs and batches waiting, as the operation is cancelled.

        A subclass may keep those finishing what was done, e.g. the deletions
        of files a move has copied.
        """
        self.waiting.clear()
        self.deletions.clear()

    async def start(self) -> None:
        """Start the jobs waiting, while permits are free, and the batches at once, but none once cancelled."""
        if self.op.cancelling():
            self.stopping()
        await self.jobs.start(self.run)
        while self.deletions:
            job = self.deletions.popleft()
            self.deleting[asyncio.ensure_future(self.run_batch(job))] = job

    def run_batch(self, job: _Job) -> Awaitable[Any]:
        """The coroutine sending a batch, e.g. a deletion by ``rm_files``: by `run` unless a subclass runs its own."""
        return self.run(job)

    def finish(self, task: asyncio.Task[Any]) -> list[Any]:
        """Handle a job's task done, counting its errors, and return what to yield."""
        job = self.running.pop(task, None) or self.deleting.pop(task)
        self.current = job.path or job.destination

        try:
            task.result()
        except Exception as e:
            outcome = self.failed(job, e)
            self.errors += sum(isinstance(o, Exception) for o in outcome)
            return outcome

        return self.done(job)


class _CopyJobs(_Jobs):
    """The jobs of a copy or a move: where each path goes, onto ``dst``.

    A path under ``prefix`` goes to the same place under ``base``. For a list,
    a path under a root of ``places`` goes there instead. `prepare` sets them,
    before `target` is used.

    ``dst`` is the handler copied to, described by its profile, unknown
    if it has none. Within one storage, it is the source, and ``same_storage``
    is True: a path copied onto itself, or under itself, is refused. Between two, a path may go to the same path, or under
    it, as that is another storage's. The paths are compared folded on a storage
    whose names are not case-sensitive, as its profile says.

    ``path1_entry`` is the entry of ``path1``, a single path, if the caller
    found it already, e.g. to tell a file. A file's is then a root known to be
    a file, not listed to find what it is.

    With ``batching``, the files go to the handler's ``cp_files``, or
    ``mv_files`` for a subclass whose ``transfers`` is "move", in batches of
    its ``cp_batch_size`` or ``mv_batch_size``. A family's jobs hand a file's
    job on by `release`, once its destination's directory is ready. A batch
    is sent once full, or once the walk has ended with no directory left to
    make, as no file can join it then. It holds no permit, as a deletion's
    batch: its operation's requests take theirs. Each file's outcome then
    goes to the family's `done` or `failed`, as a job of its own would. A
    copy refused by a destination not overwriting is retried on its own, as
    `copy_file` replaces it.
    """

    verb = "copy"
    doing = "Copying"
    transfers = "copy"
    """Which of the handler's strategies a batch of files goes to: "copy" for ``cp_files``, "move" for ``mv_files``."""

    def __init__(
        self,
        *,
        path1: StoragePath | Sequence[StoragePath],
        path2: StoragePath,
        dst: StorageHandler,
        same_storage: bool,
        path1_entry: dict[str, Any] | None = None,
        batching: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # The files ready, and the batches sent, by their number: two files
        # listed under one name, e.g. on Google Drive, may make two batches of
        # the same paths. The size is the handler's, once open.
        self.batching = batching
        self.transfer_batch_size = 1
        self.ready: list[_Job] = []
        self.sent: dict[int, list[_Job]] = {}
        self._transferred: dict[
            int, tuple[dict[str, Exception | None], list[Exception]]
        ] = {}
        self.walked = False

        # The jobs waiting for their destination directory, by that directory.
        # A directory listed in several parts is made once, as `made` notes it.
        self.after_mkdir: dict[str, list[_Job]] = {}

        self.path1 = path1
        self.path2 = path2
        self.same_storage = same_storage
        self.path1_entry = path1_entry

        # The profile is the destination's, as what a copy saves depends on
        # where it writes. Within one storage, it is the source's too. Unknown
        # until setup awaits it.
        self.dst: StorageHandler = dst
        self.profile = UNKNOWN
        self.implicit_directories = False
        self.parents_required = True
        self.case_insensitive = False

        # Set by prepare: the destination's path, verified, where the prefix goes.
        self.prefix = ""
        self.base = StoragePath("")

        # For a list: where each root goes, by the root.
        self.places: dict[str, StoragePath] = {}

        # The destinations found missing, and those found existing, which need
        # no asking again, e.g. by a move renaming a root onto one.
        self.missing: set[str] = set()
        self.existing: set[str] = set()

        # The destination directories made, tried, or known to exist: their
        # copies need not wait for a mkdir.
        self.made: set[str] = set()

        # The entries of the destination directories found or made, by path:
        # each is the parent hint of what is made inside it.
        self.directories: dict[str, Entry] = {}

        self.files_found = 0
        self.files_done = 0

    async def setup(self) -> None:
        # The destination's profile, loaded once as its handler opens. A plain
        # fsspec filesystem a router copies to has an unknown one.
        await self.dst.open()
        self.profile = self.dst.profile
        # Unknown, directories are made first. Implicit directories need none.
        # Within one storage, a file goes by the storage's own copy or move,
        # whose parents the profile tells apart from a write's.
        self.implicit_directories = bool(self.profile.implicit_directories)
        declared = (
            self.profile.copy_parents_required
            if self.same_storage
            else self.profile.parents_required
        )
        self.parents_required = declared is not False and not self.implicit_directories
        # Only within one storage are a source and its destination compared.
        self.case_insensitive = (
            self.same_storage and self.profile.case_sensitive is False
        )
        if self.batching:
            # Within one storage, the source's handler is the destination's.
            if self.transfers == "copy":
                self.transfer_batch_size = self.handler.cp_batch_size
                _check_batch_size(self.transfer_batch_size, "cp_batch_size")
            else:
                self.transfer_batch_size = self.handler.mv_batch_size
                _check_batch_size(self.transfer_batch_size, "mv_batch_size")

    def folded(self, path: str) -> str:
        """A path as the storage compares it: folded unless its names are case-sensitive."""
        return path.lower() if self.case_insensitive else path

    def state(self) -> Any:
        return CpState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_copied=self.files_done,
        )

    async def prepare(self) -> _Roots | None:
        """Set where the paths go, as fsspec's ``copy`` places them."""
        # The caller's destination, as the handler verifies it: ValueError if refused.
        destination = self.dst.verified(_trim(self.path2))

        if not isinstance(self.path1, str):
            roots = await self._place(destination)
            sources = roots.paths
        elif has_magic(self.path1):
            roots = await self.glob_roots(self.path1)
            sources = roots.paths
            if not sources:
                return roots  # Only errors, in found, to yield.
            self.prefix = common_prefix(sources)
            if not roots.directories and len(sources) == 1:
                # A single file goes into the destination.
                self.prefix = self.prefix.rpartition("/")[0]
            self.base = destination
        else:
            root = _trim(self.path1)
            if self.same_storage and self.folded(root) == self.folded(destination):
                # Onto itself: left as it is, unless a move to another case.
                recased = root != destination and self.verb == "move"
                if not recased or self.path1.endswith("/") or self.path2.endswith("/"):
                    return None
                return self.recased(root, destination)
            roots = await self._single(root, destination)
            sources = [root]

        for source in sources if self.same_storage else ():
            target, folded = self.folded(self.target(source)), self.folded(source)
            if target == folded and len(sources) == 1:
                return None
            if not source or target == folded or target.startswith(folded + "/"):
                raise ValueError(
                    f"Cannot {self.verb} {source!r} into itself: {self.target(source)!r}"
                )

        return roots

    def recased(self, root: str, destination: StoragePath) -> _Roots:
        """The root of a single path moved onto itself in another case, to be renamed.

        On a storage whose names are not case-sensitive, only a rename of the
        root whole changes its case: a move doing so places it, as a root
        going to a destination found missing. Here, it raises ValueError.
        """
        raise ValueError(
            f"Cannot {self.verb} {root!r} to {destination!r}, itself in another"
            " case: the storage renames nothing whole"
        )

    async def _place(self, destination: StoragePath) -> _Roots:
        """Place each root of a list into ``destination``, under its name.

        Two roots of the same name would go to the same place. As GNU's cp
        does, the first, in path order, goes, and each other is an error. That
        error is the root's own: what is in it is not copied either.
        """
        roots = await self.glob_roots(self.path1)
        directories = {d["name"]: d for d in roots.directories}
        files = {f["name"]: f for f in roots.files}
        placed: dict[StoragePath, str] = {}

        for source in sorted(roots.paths):
            name = source.rpartition("/")[2]
            target = self.dst.join(destination, name)
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
            [directories[s] for s in self.places if s in directories],
            [files[s] for s in self.places if s in files],
        )

    async def _single(self, root: str, destination: StoragePath) -> _Roots:
        """Place a single path, and return it as a root: a file if its entry says so."""
        self.prefix = root
        self.base = await self._base(root, destination)
        if self.same_storage:
            # The root's parent exists: e.g. a file renamed in its directory needs no mkdir.
            self.made.add(root.rpartition("/")[0])

        entry = self.path1_entry
        if entry is not None and entry["type"] != "directory":
            return _Roots([], [{**entry, "name": root}])
        else:
            return _Roots([{"name": root}])

    async def _base(self, root: str, destination: StoragePath) -> StoragePath:
        """Where ``root``, a single path, goes: into ``destination``, or onto it.

        What its ``_info`` finds is kept: a destination missing or existing,
        and the directory it shows to exist, which needs no mkdir.
        """
        if self.path2.endswith("/"):
            into = True
        else:
            try:
                entry = await self.dst.info(self.dst.entry(destination))
            except FileNotFoundError:
                self.missing.add(destination)
                into = False
            else:
                self.existing.add(destination)
                into = entry["type"] == "directory"
                self.made.add(destination if into else destination.rpartition("/")[0])
                if into:
                    self.directories[destination] = entry

        if not into or (isinstance(self.path1, str) and self.path1.endswith("/")):
            return destination

        name = root.rsplit("/", 1)[-1]
        return self.dst.join(destination, name)

    def target(self, path: str) -> StoragePath:
        """Where a path goes, each name below its root checked by the destination's ``join``.

        A name the destination refuses, e.g. "..", raises ValueError: a path
        listed on the source never leaves the destination. A flat storage's
        directory marker, a key ending with "/", keeps its "/".
        """
        if self.places:
            # The root it is under, found from the path up.
            root = path
            while root not in self.places and "/" in root:
                root = root.rpartition("/")[0]
            target = self.dst.join(self.places[root], path[len(root) :])
        elif self.prefix:
            target = self.dst.join(self.base, path[len(self.prefix) :])
        else:
            target = self.dst.join(self.base, path)

        if path.endswith("/") and not target.endswith("/"):
            # A flat storage's marker is a key: its directory's path and a "/".
            return StoragePath(f"{target}/")
        else:
            return target

    def known_missing(self, target: str) -> bool:
        """Whether nothing is at a destination, as the operation knows: it, or a directory above it, was found missing.

        All below a destination found missing is missing too, unless another
        client puts something there meanwhile: the primitives making it are
        told so, by ``missing=True``, and a storage whose creation never
        conflicts asks nothing first.
        """
        path = target.rstrip("/")
        while path:
            if path in self.missing:
                return True
            path = path.rpartition("/")[0]
        return False

    def refuse_name(self, path: str, error: ValueError) -> ValueError:
        """A path whose name the destination refuses, as its ``target`` raised: counted, noted, to yield.

        It is the path's own error. The operation goes on with the others.
        """
        self.errors += 1
        error.add_note(
            f"{self.doing} {path!r} refused: the destination takes no such name."
        )
        return error

    async def copy_file(
        self, cp_entry: _Transfer, entry: dict[str, Any], destination: StoragePath
    ) -> None:
        """Copy a file by ``cp_entry``, replacing a file there the destination keeps.

        A destination refusing to overwrite, as fsspec's copy does not, costs
        an ``_info``, an ``_rm_file`` and a copy more. A directory there is no file
        to replace: ``_rm_file`` takes files only. The job holds its permit
        already. An error raised notes its paths, as it is yielded among paths.
        """
        source = entry["name"]
        try:
            try:
                await cp_entry(
                    entry,
                    destination,
                    missing=self.known_missing(destination),
                    parent=self.parent_of(destination),
                )
            except FileExistsError as refused:
                await self.replace_file(cp_entry, entry, destination, refused)
        except Exception as e:
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise

    async def replace_file(
        self,
        cp_entry: _Transfer,
        entry: dict[str, Any],
        destination: StoragePath,
        refused: FileExistsError,
    ) -> None:
        """Copy a file onto one its destination refused to overwrite: deleted first, unless a directory.

        A directory there raises ``refused`` again: ``rm_file`` takes files only.
        """
        found = await self.dst.info(self.dst.entry(destination))
        if found["type"] == "directory":
            raise refused

        await self.dst.rm_file({**found, "name": destination})
        await cp_entry(entry, destination, parent=self.parent_of(destination))

    def parent_of(self, path: StoragePath) -> Entry | None:
        """The entry of a destination's parent, as found or made, as the primitives take it: None if unknown."""
        try:
            return self.directories.get(self.dst.parent(path))
        except ValueError:
            return None  # A path at the root: no parent of the destination's.

    async def make_directory(self, path: StoragePath) -> None:
        """Make a destination directory, keeping the entry ``makedirs`` returns as the parent hint of what goes inside."""
        made = await self.dst.makedirs(
            path,
            exist_ok=True,
            missing=self.known_missing(path),
            parent=self.parent_of(path),
        )
        if made is not None:
            self.directories[path] = made

    # The batches of files, sent to the handler's cp_files or mv_files.

    def release(self, jobs: Iterable[_Job]) -> None:
        """Queue jobs whose destination's directory is ready: a file's into the batch, when batching."""
        for job in jobs:
            if not self.batching or job.kind != "copy":
                self.waiting.append(job)
                continue

            self.ready.append(job)
            if len(self.ready) >= self.transfer_batch_size:
                self.send_ready()

    def send_ready(self) -> None:
        """Queue the batch of the files ready, to be sent without a permit."""
        jobs, self.ready = self.ready, []
        paths = tuple(j.path for j in jobs)
        number = self.batches_sent = self.batches_sent + 1
        self.sent[number] = jobs
        batch = _Job("transfers", {"name": paths[-1], "batch": number}, paths=paths)
        self.deletions.append(batch)

    def exhausted(self) -> None:
        self.walked = True
        super().exhausted()

    async def start(self) -> None:
        # The last batch, once no file can join it: the walk ended, and no
        # directory is left to make, whose files it would release.
        if self.ready and self.walked and not self.op.cancelling():
            making = (*self.waiting, *self.running.values())
            if not any(j.kind == "mkdir" for j in making):
                self.send_ready()
        await super().start()

    def stopping(self) -> None:
        super().stopping()
        self.drop_transfers()

    def drop_transfers(self) -> None:
        """Drop the files ready and the batches not sent, as the operation is cancelled."""
        self.ready.clear()
        for job in [j for j in self.deletions if j.kind == "transfers"]:
            self.deletions.remove(job)
            for dropped in self.sent.pop(job.source["batch"]):
                self.dropped(dropped)

    def dropped(self, job: _Job) -> None:
        """Note a file's job dropped unsent, e.g. by a cancelled batch: neither done nor failed."""

    def transfers_running(self) -> int:
        """The files in the batches being sent."""
        return sum(
            len(self.sent[j.source["batch"]])
            for j in self.deleting.values()
            if j.kind == "transfers"
        )

    def run_batch(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "transfers":
            return self.transfer_batch(job.source["batch"])
        else:
            return super().run_batch(job)

    async def transfer_batch(self, number: int) -> None:
        """Send a batch of files by the handler's ``cp_files`` or ``mv_files``, keeping each one's outcome.

        A copy refused by a destination not overwriting is replaced on its own,
        under a permit, as `copy_file` replaces it. An error with no ``source``
        is kept apart, yielded as it is.
        """
        jobs = self.sent[number]
        transfers = [
            FileTransfer(
                j.source,
                j.destination,
                self.known_missing(j.destination),
                self.parent_of(j.destination),
            )
            for j in jobs
        ]
        operation: Operation[tuple[str, str], Any]
        if self.transfers == "copy":
            operation = self.handler.cp_files(transfers)
        else:
            operation = self.handler.mv_files(transfers)

        outcomes: dict[str, Exception | None] = {}
        apart: list[Exception] = []
        async with operation as files:
            async for outcome in files:
                if not isinstance(outcome, Exception):
                    outcomes[outcome[0]] = None
                elif (
                    source := getattr(outcome, "source", None)
                ) in outcomes or source is None:
                    apart.append(outcome)
                else:
                    outcomes[source] = outcome

        if self.transfers == "copy":
            refused = [
                j for j in jobs if isinstance(outcomes.get(j.path), FileExistsError)
            ]
            replaced = await asyncio.gather(
                *(self.replace_refused(j, outcomes[j.path]) for j in refused),  # type: ignore[arg-type]
                return_exceptions=True,
            )
            for job, error in zip(refused, replaced, strict=True):
                outcomes[job.path] = error if isinstance(error, Exception) else None

        self._transferred[number] = (outcomes, apart)

    async def replace_refused(self, job: _Job, refused: FileExistsError) -> None:
        """Replace a file a batch's copy was refused, under a permit, by the source's ``cp_file``."""
        try:
            async with self.concurrency_limit:
                cp_entry = cast("_Transfer", self.handler.cp_file)
                await self.replace_file(cp_entry, job.source, job.destination, refused)
        except Exception as e:
            if e is not refused:  # The batch's own error notes its paths already.
                e.add_note(f"{self.doing} {job.path!r} to {job.destination!r} failed.")
            raise

    def finish(self, task: asyncio.Task[Any]) -> list[Any]:
        """Handle a task done: a batch of files gives each one's outcome to `done` or `failed`."""
        batch = self.deleting.get(task)
        if batch is None or batch.kind != "transfers":
            return super().finish(task)

        del self.deleting[task]
        jobs = self.sent.pop(batch.source["batch"])
        self.current = batch.path
        outcome: list[Any] = []
        try:
            task.result()
        except Exception as e:
            # Failed whole: one error for all its files, each failed by it.
            e.add_note(f"{self.doing} {_keys_of(batch.paths)} failed.")
            for job in jobs:
                outcome += [
                    o for o in self.failed(job, e) if all(o is not x for x in outcome)
                ]
            self.errors += sum(isinstance(o, Exception) for o in outcome)
            return outcome

        outcomes, apart = self._transferred.pop(batch.source["batch"])
        for job in jobs:
            if job.path not in outcomes:
                self.dropped(job)  # Neither done nor failed, e.g. cancelled.
            elif (error := outcomes[job.path]) is None:
                outcome += self.done(job)
            else:
                failed = self.failed(job, error)
                self.errors += sum(isinstance(o, Exception) for o in failed)
                outcome += failed
        self.errors += len(apart)
        return outcome + apart
