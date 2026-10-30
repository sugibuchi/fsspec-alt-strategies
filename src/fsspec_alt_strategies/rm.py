# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive delete that walks the tree and deletes while it walks.

`tree_rm` is an alternative to fsspec's recursive ``_rm``. fsspec's lists the
whole tree first, then deletes each path. `tree_rm` deletes each folder's files
as soon as the folder is listed, and each folder once it is empty.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsspec.asyn import _get_batch_size

from .control import Result, State, _finish, _report
from .walk import _listings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from fsspec.asyn import AsyncFileSystem

    from .control import Callback, Stop
    from .walk import _Listing

__all__ = ["RmResult", "RmState", "tree_rm"]


@dataclass(frozen=True, kw_only=True)
class RmState(State):
    """The state of a `tree_rm`, as its callback gets it.

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


@dataclass(kw_only=True)
class RmResult(Result):
    """What a `tree_rm` deleted.

    A folder is kept when anything in it failed: its path and the failure are in
    ``errors``, along with those of files.

    Attributes
    ----------
    files_deleted : int
        The files deleted.
    folders_deleted : int
        The folders deleted, the starting one included.
    """

    files_deleted: int = 0
    folders_deleted: int = 0


async def tree_rm(
    fs: AsyncFileSystem,
    path: str,
    *,
    rm_folder: Callable[[str], Awaitable[Any]] | None = None,
    batch_size: int | None = None,
    callback: Callback[RmState] | None = None,
) -> RmResult:
    """Delete a folder and everything in it, deleting while the tree is walked.

    The tree is walked as `iwalk` walks it. Each folder's files are deleted as soon as
    the folder is listed, while other folders are still being listed. A folder
    is deleted once everything in it is, so the deepest go first.

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
    RmResult
        What was deleted, and the paths that failed.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    OperationStopped
        If the callback returned ``Stop(raise_error=True)``.

    See Also
    --------
    iwalk : The walk under it.

    Examples
    --------
    >>> result = await tree_rm(fs, "data/2024", batch_size=32)
    >>> result.complete, result.files_deleted
    (True, 1234)
    """
    if rm_folder is None:
        rm_folder = getattr(fs, "_rmdir", None) or fs._rm_file
    batch_size = batch_size or fs.batch_size or _get_batch_size(nofiles=True)
    return _finish(await _Deletion(fs, rm_folder, batch_size, callback).run(path))


class _Deletion:
    """The bookkeeping of one `tree_rm` call, kept apart from the function."""

    def __init__(
        self,
        fs: AsyncFileSystem,
        rm_folder: Callable[[str], Awaitable[Any]],
        batch_size: int,
        callback: Callback[RmState] | None,
    ) -> None:
        self.fs = fs
        self.rm_folder = rm_folder
        self.batch_size = batch_size
        self.callback = callback
        self.result = RmResult()
        self.files_found = 0
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

    async def run(self, path: str) -> RmResult:
        listings = _listings(self.fs, path, batch_size=self.batch_size)
        try:
            stop = await self.delete(path, listings)
            if stop is not None:
                self.result.stop = stop
                # Deletions already sent may have happened: wait for their
                # outcome, but start nothing new.
                self.waiting.clear()
                while self.running:
                    done, _ = await asyncio.wait(self.running)
                    for task in done:
                        self.finish(task)
            return self.result
        finally:
            # Listings are only reads, so cancelling them loses nothing. The
            # deletions are cancelled only when this task itself is.
            tasks = [*self.running, *([self.next] if self.next else [])]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await listings.aclose()

    async def delete(
        self, path: str, listings: AsyncGenerator[_Listing]
    ) -> Stop | None:
        """Delete until all is done, or a callback stops, and return its stop."""
        try:
            self.add(await anext(listings))
        except NotADirectoryError:
            self.files_found = 1
            self.waiting.append((False, self.fs._strip_protocol(path)))
        else:
            self.next = asyncio.ensure_future(anext(listings))
        stop = await self.report()
        while (self.next or self.running or self.waiting) and stop is None:
            self.start()
            done, _ = await asyncio.wait(
                [*self.running, *([self.next] if self.next else [])],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task is self.next:
                    self.next = None
                    try:
                        self.add(task.result())
                    except StopAsyncIteration:
                        continue
                    self.next = asyncio.ensure_future(anext(listings))
                else:
                    self.finish(task)
                stop = await self.report()
                if stop is not None:
                    break
        return stop

    def add(self, listing: _Listing) -> None:
        """Queue the deletion of a listed folder's files."""
        self.current = listing.folder
        if listing.error is not None:
            self.result.errors[listing.folder] = listing.error
            return
        self.remaining[listing.folder] = len(listing.entries)
        for entry in listing.entries:
            name = entry["name"].rstrip("/")
            self.parents[name] = listing.folder
            if entry["type"] != "directory":
                self.files_found += 1
                self.waiting.append((False, name))
        if not listing.entries:
            self.waiting.append((True, listing.folder))

    def start(self) -> None:
        while self.waiting and len(self.running) < self.batch_size:
            is_folder, path = self.waiting.popleft()
            coro = self.rm_folder(path) if is_folder else self.fs._rm_file(path)
            self.running[asyncio.ensure_future(coro)] = is_folder, path

    def finish(self, task: asyncio.Task[Any]) -> None:
        """Count a finished deletion, and queue its folder's if now empty."""
        is_folder, path = self.running.pop(task)
        self.current = path
        try:
            task.result()
        except FileNotFoundError:
            # Already gone: an object store's folder, or deleted by another.
            pass
        except Exception as e:
            self.result.errors[path] = e
            return
        else:
            if is_folder:
                self.result.folders_deleted += 1
            else:
                self.result.files_deleted += 1
        parent = self.parents.get(path)
        if parent is not None:
            self.remaining[parent] -= 1
            if self.remaining[parent] == 0:
                self.waiting.append((True, parent))

    async def report(self) -> Stop | None:
        return await _report(
            self.callback,
            RmState(
                errors=len(self.result.errors),
                current=self.current,
                files_found=self.files_found,
                files_deleted=self.result.files_deleted,
                folders_deleted=self.result.folders_deleted,
            ),
        )
