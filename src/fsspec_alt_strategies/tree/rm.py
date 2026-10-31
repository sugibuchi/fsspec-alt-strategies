# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive delete that walks the tree and deletes while it walks.

`tree_rm` is an alternative to fsspec's recursive ``_rm``. fsspec's lists the whole
tree first, then deletes each path. `tree_rm` deletes each folder's files as soon as
the folder is listed, and each folder once it is empty.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsspec.asyn import _get_batch_size

from ..control import Operation, State
from .walk import _listings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from fsspec.asyn import AsyncFileSystem

    from .walk import _Listing

__all__ = ["RmState", "tree_rm"]


@dataclass(frozen=True, kw_only=True)
class RmState(State):
    """The progress of a `tree_rm`.

    Attributes
    ----------
    files_found : int
        The files found so far. It grows as the walk goes on.
    files_deleted : int
        The files deleted so far.
    folders_deleted : int
        The folders deleted so far.
    """

    files_found: int = 0
    files_deleted: int = 0
    folders_deleted: int = 0


def tree_rm(
    fs: AsyncFileSystem,
    path: str,
    *,
    rm_folder: Callable[[str], Awaitable[Any]] | None = None,
    batch_size: int | None = None,
) -> Operation[str, RmState]:
    """Delete a folder and everything in it, deleting while the tree is walked.

    The tree is walked as `tree_walk` walks it. Each folder's files are deleted as
    soon as the folder is listed, while other folders are still being listed. A
    folder is deleted once everything in it is, so the deepest go first. A
    folder with anything in it that failed is kept.

    It yields each file and folder once deleted, and the error of each folder
    that failed to list or path that failed to be deleted. A path found already
    gone is not yielded.

    It wins over fsspec's ``_rm`` on storages that list one folder per request
    and have no recursive delete of their own, such as Databricks volumes. The
    deletions then start after the first listing, not after the last. It also
    keeps going past a failure, and reports each.

    It loses on object stores, which list a whole tree in one paginated request:
    fsspec's ``_rm`` makes fewer requests there. On a storage deleting a folder
    with everything in it in one request, that request wins.

    Once cancelled, it starts no new listing or deletion, and yields the
    outcomes of the deletions already sent. Closed, it waits for those
    deletions without yielding them, as their outcome would otherwise be
    unknown. Iterating it raises FileNotFoundError if ``path`` does not exist.

    Requests: one ``_ls`` per folder, one ``_rm_file`` per file, and one
    ``rm_folder`` per folder. At most ``batch_size`` listings and
    ``batch_size`` deletions run at a time.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path : str
        The folder to delete. If it is a file, that file is deleted.
    rm_folder : callable, optional
        An async function deleting an empty folder. By default, the
        filesystem's ``_rmdir`` if it has one, otherwise its ``_rm_file``, as
        fsspec's ``_rm`` does. A FileNotFoundError from it counts as deleted, as
        object stores have no folders to delete.
    batch_size : int, optional
        The most listings, and the most deletions, at a time. fsspec's default
        if None.

    Returns
    -------
    Operation
        Yielding each path deleted, or an error. Its state is an `RmState`.

    See Also
    --------
    tree_walk : The walk under it.

    Examples
    --------
    >>> async with tree_rm(fs, "data/2024", batch_size=32) as op:
    ...     async for path in op:
    ...         if isinstance(path, Exception):
    ...             op.cancel()  # Stop at the first failure.
    ...             print(path)
    >>> op.state.files_deleted
    1234
    """

    async def run(op: Operation[str, RmState]) -> AsyncGenerator[str | Exception]:
        deletion = _Deletion(fs, rm_folder, batch_size, op)
        try:
            async with aclosing(deletion.steps(path)) as steps:
                async for outcomes in steps:
                    op.state = deletion.state()
                    for outcome in outcomes:
                        yield outcome
        finally:
            # Closing the steps waits for the deletions sent, and counts them.
            op.state = deletion.state()

    return Operation(run, RmState())


class _Deletion:
    """The bookkeeping of one `tree_rm` operation, kept apart from the function."""

    def __init__(
        self,
        fs: AsyncFileSystem,
        rm_folder: Callable[[str], Awaitable[Any]] | None,
        batch_size: int | None,
        op: Operation[str, RmState],
    ) -> None:
        self.fs = fs
        self.rm_folder = rm_folder or getattr(fs, "_rmdir", None) or fs._rm_file
        self.batch_size = batch_size or fs.batch_size or _get_batch_size(nofiles=True)
        self.op = op
        self.errors = 0
        self.files_found = 0
        self.files_deleted = 0
        self.folders_deleted = 0
        self.current: str | None = None
        # The paths in each folder listed that are not deleted yet. A folder
        # whose count reaches zero is empty and can go. One with a failure in
        # it never reaches zero, and nor do the folders above it.
        self.remaining: dict[str, int] = {}
        self.parents: dict[str, str] = {}
        # Deletions to start, and those running: whether a folder, and the path.
        self.waiting: deque[tuple[bool, str]] = deque()
        self.running: dict[asyncio.Task[Any], tuple[bool, str]] = {}
        # The walk's next listing, awaited along with the deletions.
        self.next: asyncio.Task[_Listing] | None = None

    async def steps(self, path: str) -> AsyncGenerator[list[str | Exception]]:
        """Delete, yielding what each step deleted or failed.

        A step is a folder listed or a deletion finished. Closing this stops the
        deletion: it starts nothing new, and waits for the deletions sent, as
        their outcome would otherwise be unknown.
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
            # deletions are cancelled only when this task itself is.
            tasks = [*self.running, *([self.next] if self.next else [])]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await listings.aclose()

    async def begin(
        self, path: str, listings: AsyncGenerator[_Listing]
    ) -> list[str | Exception]:
        """List ``path``, or queue its own deletion if a file: the first step."""
        try:
            first = await anext(listings)
        except NotADirectoryError:
            self.files_found = 1
            self.waiting.append((False, self.fs._strip_protocol(path)))
            return []
        self.next = asyncio.ensure_future(anext(listings))
        return self.add(first)

    def take(
        self, task: asyncio.Task[Any], listings: AsyncGenerator[_Listing]
    ) -> list[str | Exception] | None:
        """Handle a finished task, returning its step, or None if the walk ended."""
        if task is not self.next:
            return self.finish(task)
        self.next = None
        try:
            listing = task.result()
        except StopAsyncIteration:
            return None
        self.next = asyncio.ensure_future(anext(listings))
        return self.add(listing)

    def add(self, listing: _Listing) -> list[str | Exception]:
        """Queue the deletion of a listed folder's files, and return its error."""
        self.current = listing.folder
        if listing.error is not None:
            self.errors += 1
            return [listing.error]
        self.remaining[listing.folder] = len(listing.entries)
        for entry in listing.entries:
            name = entry["name"].rstrip("/")
            self.parents[name] = listing.folder
            if entry["type"] != "directory":
                self.files_found += 1
                self.waiting.append((False, name))
        if not listing.entries:
            self.waiting.append((True, listing.folder))
        return []

    def start(self) -> None:
        if self.op.cancelling():
            self.waiting.clear()
        while self.waiting and len(self.running) < self.batch_size:
            is_folder, path = self.waiting.popleft()
            coro = self.rm_folder(path) if is_folder else self.fs._rm_file(path)
            self.running[asyncio.ensure_future(coro)] = is_folder, path

    def finish(self, task: asyncio.Task[Any]) -> list[str | Exception]:
        """Count a finished deletion, queue its folder's if now empty, and return it."""
        is_folder, path = self.running.pop(task)
        self.current = path
        outcome: list[str | Exception] = []
        try:
            task.result()
        except FileNotFoundError:
            # Already gone: an object store's folder, or deleted by another.
            pass
        except Exception as e:
            # Yielded among paths, the error must tell its path itself.
            e.add_note(f"Deleting {path!r} failed.")
            self.errors += 1
            return [e]
        else:
            if is_folder:
                self.folders_deleted += 1
            else:
                self.files_deleted += 1
            outcome.append(path)
        parent = self.parents.get(path)
        if parent is not None:
            self.remaining[parent] -= 1
            if self.remaining[parent] == 0:
                self.waiting.append((True, parent))
        return outcome

    def state(self) -> RmState:
        return RmState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_deleted=self.files_deleted,
            folders_deleted=self.folders_deleted,
        )
