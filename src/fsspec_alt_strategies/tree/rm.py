# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive delete that walks the tree and deletes while it walks.

`tree_rm` is an alternative to fsspec's recursive ``_rm``. fsspec's lists the
whole tree first, then deletes each path. `tree_rm` deletes each folder's files
as soon as the folder is listed, and each folder once it is empty.

This module depends on `control`, `walk` and `_jobs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..control import Operation, State
from ._jobs import _Job, _Jobs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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

    The tree is walked as `tree_walk` walks it. Each folder's files are deleted
    as soon as the folder is listed, while other folders are still being
    listed. A folder is deleted once everything in it is, so the deepest go
    first. A folder with anything in it that failed is kept.

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
    return Operation(
        lambda op: _Deletion(fs, path, rm_folder, batch_size, op).outcomes(),
        RmState(),
    )


class _Deletion(_Jobs):
    """The jobs of one `tree_rm`: a deletion per file, then per folder."""

    def __init__(
        self,
        fs: AsyncFileSystem,
        path: str,
        rm_folder: Callable[[str], Awaitable[Any]] | None,
        batch_size: int | None,
        op: Operation[str, RmState],
    ) -> None:
        super().__init__(fs, batch_size, op)
        self.path = path
        self.rm_folder = rm_folder or getattr(fs, "_rmdir", None) or fs._rm_file
        self.files_found = 0
        self.files_deleted = 0
        self.folders_deleted = 0

    async def prepare(self) -> str:
        return self.path

    def state(self) -> RmState:
        return RmState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_deleted=self.files_deleted,
            folders_deleted=self.folders_deleted,
        )

    def file_root(self, path: str) -> None:
        self.files_found = 1
        self.waiting.append(_Job("file", path))

    def add(self, listing: _Listing) -> None:
        self.track(listing)
        for entry in listing.entries:
            if entry["type"] != "directory":
                self.files_found += 1
                self.waiting.append(_Job("file", entry["name"].rstrip("/")))
        if not listing.entries:
            self.waiting.append(_Job("folder", listing.folder))

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "folder":
            return self.rm_folder(job.source)
        return self.fs._rm_file(job.source)

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "folder":
            self.folders_deleted += 1
        else:
            self.files_deleted += 1
        self.after(job.source)
        return [job.source]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if isinstance(error, FileNotFoundError):
            # Already gone: an object store's folder, or deleted by another.
            self.after(job.source)
            return []
        # Yielded among paths, the error must tell its path itself.
        error.add_note(f"Deleting {job.source!r} failed.")
        return [error]

    def after(self, path: str) -> None:
        """Queue the deletion of a path's folder, if it is now empty."""
        if (folder := self.child_done(path)) is not None:
            self.waiting.append(_Job("folder", folder))
