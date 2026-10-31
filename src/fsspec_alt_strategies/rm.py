# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive delete that walks the tree and deletes while it walks.

`rm` is an alternative to fsspec's recursive ``_rm``. fsspec's lists the whole
tree first, then deletes each path. `rm` deletes each folder's files as soon as
the folder is listed, and each folder once it is empty.

`irm` is the same deletion as an async iterator, yielding each path as it is
deleted.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsspec.asyn import _get_batch_size

from .control import State, _finish, _report
from .walk import _listings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable

    from fsspec.asyn import AsyncFileSystem

    from .control import Callback, Stop
    from .walk import _Listing

__all__ = ["RmState", "irm", "rm"]


@dataclass(frozen=True, kw_only=True)
class RmState(State):
    """The state of an `rm`, as its callback gets it.

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


async def rm(
    fs: AsyncFileSystem,
    path: str,
    *,
    rm_folder: Callable[[str], Awaitable[Any]] | None = None,
    batch_size: int | None = None,
    callback: Callback[RmState] | None = None,
) -> tuple[list[str], list[Exception]]:
    """Delete a folder and everything in it, deleting while the tree is walked.

    The tree is walked as `iwalk` walks it. Each folder's files are deleted as
    soon as the folder is listed, while other folders are still being listed. A
    folder is deleted once everything in it is, so the deepest go first. A
    folder with anything in it that failed is kept.

    It wins over fsspec's ``_rm`` on storages that list one folder per request
    and have no recursive delete of their own, such as Databricks volumes. The
    deletions then start after the first listing, not after the last. It also
    keeps going past a failure, and reports each.

    It loses on object stores, which list a whole tree in one paginated request:
    fsspec's ``_rm`` makes fewer requests there. On a storage deleting a folder
    with everything in it in one request, that request wins.

    Requests: one ``_ls`` per folder, one ``_rm_file`` per file, and one
    ``rm_folder`` per folder. At most ``batch_size`` listings and
    ``batch_size`` deletions run at a time.

    The callback is called after each folder listed and each path deleted or
    failed. Once stopped, no new listing or deletion starts, and the deletions
    already sent are waited for. ``Stop(cleanup=True)`` undoes nothing, as a
    deletion cannot be undone.

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
    callback : callable, optional
        Called with an `RmState` after each step. It may return a `Stop`.

    Returns
    -------
    deleted : list of str
        The files and folders deleted, in the order they were. A path found
        already gone is not in it.
    errors : list of Exception
        The error of each folder that failed to list, and of each path that
        failed to be deleted, in the order they failed.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    OperationStopped
        If the callback returned ``Stop(raise_error=True)``.

    See Also
    --------
    irm : The same deletion, as an async iterator.
    iwalk : The walk under it.

    Examples
    --------
    >>> deleted, errors = await rm(fs, "data/2024", batch_size=32)
    >>> len(deleted), errors
    (1234, [])
    """
    deletion = _Deletion(fs, rm_folder, batch_size)
    stop: Stop | None = None
    async with aclosing(deletion.steps(path)) as steps:
        async for _ in steps:
            if (stop := await _report(callback, deletion.state())) is not None:
                break
    # Closing the steps waited for the deletions sent, and counted them.
    return _finish((deletion.deleted, deletion.errors), stop, deletion.errors)


async def irm(
    fs: AsyncFileSystem,
    path: str,
    *,
    rm_folder: Callable[[str], Awaitable[Any]] | None = None,
    batch_size: int | None = None,
) -> AsyncIterator[str | Exception]:
    """Delete a folder and everything in it, yielding each path as it goes.

    It deletes as `rm` does, with the same requests. It yields each path once
    deleted, and the error of each folder that failed to list or path that
    failed to be deleted.

    Like `iwalk`, it takes no callback: the caller drives it. Breaking out of
    the loop stops it: no new listing or deletion starts, and the deletions
    already sent are waited for, though not yielded. Close it with
    `contextlib.aclosing`, so that this happens at once. New deletions start
    only as the caller asks for paths, so a slow loop body slows the deletion.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path : str
        The folder to delete. If it is a file, that file is deleted.
    rm_folder : callable, optional
        An async function deleting an empty folder, as with `rm`.
    batch_size : int, optional
        The most listings, and the most deletions, at a time. fsspec's default
        if None.

    Yields
    ------
    str or Exception
        Each file or folder deleted, or an error.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.

    See Also
    --------
    rm : The same deletion, collected, with a callback.
    iwalk : The walk under it.

    Examples
    --------
    >>> from contextlib import aclosing
    >>> async with aclosing(irm(fs, "data/2024")) as deleted:
    ...     async for path in deleted:
    ...         if isinstance(path, Exception):
    ...             raise path
    ...         print(path)
    data/2024/01/part-0.parquet
    """
    deletion = _Deletion(fs, rm_folder, batch_size)
    async with aclosing(deletion.steps(path)) as steps:
        async for outcomes in steps:
            for outcome in outcomes:
                yield outcome


class _Deletion:
    """The bookkeeping of one `rm` or `irm` call, kept apart from the functions."""

    def __init__(
        self,
        fs: AsyncFileSystem,
        rm_folder: Callable[[str], Awaitable[Any]] | None,
        batch_size: int | None,
    ) -> None:
        self.fs = fs
        self.rm_folder = rm_folder or getattr(fs, "_rmdir", None) or fs._rm_file
        self.batch_size = batch_size or fs.batch_size or _get_batch_size(nofiles=True)
        self.deleted: list[str] = []
        self.errors: list[Exception] = []
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
        listings = _listings(self.fs, path, batch_size=self.batch_size)
        try:
            yield await self.begin(path, listings)
            while self.next or self.running or self.waiting:
                self.start()
                done, _ = await asyncio.wait(
                    [*self.running, *([self.next] if self.next else [])],
                    return_when=asyncio.FIRST_COMPLETED,
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
            self.errors.append(listing.error)
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
            self.errors.append(e)
            return [e]
        else:
            if is_folder:
                self.folders_deleted += 1
            else:
                self.files_deleted += 1
            self.deleted.append(path)
            outcome.append(path)
        parent = self.parents.get(path)
        if parent is not None:
            self.remaining[parent] -= 1
            if self.remaining[parent] == 0:
                self.waiting.append((True, parent))
        return outcome

    def state(self) -> RmState:
        return RmState(
            errors=len(self.errors),
            current=self.current,
            files_found=self.files_found,
            files_deleted=self.files_deleted,
            folders_deleted=self.folders_deleted,
        )
