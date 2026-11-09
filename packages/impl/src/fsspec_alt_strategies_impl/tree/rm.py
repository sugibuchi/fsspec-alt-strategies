# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive delete that walks the tree and deletes while it walks.

`_tree_rm` is an alternative to fsspec's recursive ``_rm``. fsspec's lists the
whole tree first, then deletes each path. `_tree_rm` deletes each folder's files
as soon as the folder is listed, and each folder once it is empty.

This module depends on `_jobs`, and `filesystem`, `jobs` and `paths` of
`common`.
"""

from __future__ import annotations

import asyncio
from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import RmState

from ..common.filesystem import _profile_of
from ..common.jobs import _Job, _Roots
from ..common.paths import _check_maxdepth
from ._jobs import _TreeJobs

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from ..common.operation import Operation
    from .walk import _Listing

__all__: list[str] = []


def _tree_rm(
    fs: Any,
    path: str | Sequence[str],
    *,
    maxdepth: int | None = None,
) -> Operation[str, RmState]:
    """Delete a folder and everything in it, deleting while the tree is walked.

    The tree is walked as `_tree_walk` walks it. Each folder's files are deleted
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
    with everything in it in one request, that request wins: the filesystem
    gives it as its ``_rm_tree``, called ``rm_tree`` below.

    With ``rm_tree``, each folder root is first deleted whole by one
    ``rm_tree``, with no walk. A root whose deletion fails, e.g. as the
    storage refuses so large a folder, is walked and deleted file by file, as
    without it. A real failure then shows on its files. A root deleted whole
    is yielded once: what was in it is neither listed, yielded nor counted.

    A glob pattern deletes each path it matches, and all in it, as fsspec's
    ``_rm`` with ``recursive=True`` does. So does a list of paths and
    patterns. They are matched as `_tree_glob` matches them, in one search,
    then the folders matched are walked at once.

    With ``maxdepth``, it deletes what fsspec's ``_rm`` deletes with it: the
    paths at most ``maxdepth`` levels below ``path``, or below a pattern's
    matches one level less, then ``path`` itself. Folders deeper are not
    listed. A folder at the limit is found but not listed, so it is deleted
    by one ``rm_folder``, as fsspec deletes each path it finds. One with
    anything in it then fails, and the folders above it are kept.
    ``rm_tree`` is not used, as it would delete what is deeper.

    Once cancelled, it starts no new listing or deletion, and yields the
    outcomes of the deletions already sent. Closed, it waits for those
    deletions without yielding them, as their outcome would otherwise be
    unknown. Iterating it raises FileNotFoundError if ``path`` does not exist,
    or matches nothing.

    Requests: one ``_ils`` per folder, one ``rm_file`` per file, and one
    ``rm_folder`` per folder. Patterns, or a list, add the requests of their
    `_tree_glob`. With ``rm_tree``, a root deleted whole costs one ``rm_tree``
    instead. A single ``path`` then costs an ``info`` first, as ``rm_tree``
    is given folders only. Each request holds a permit of the filesystem's
    ``operation_limit`` while it runs.

    The filesystem's profile's ``implicit_folders=True`` saves the ``rm_folder`` of each
    folder, as a folder goes with its last file. Such a folder is not yielded,
    as one found already gone is not.

    The filesystem's primitives are called below by what they do: ``_ils`` is
    its ``_ils``, ``info`` its ``_info``, ``rm_file`` its ``_rm_file``, and
    ``rm_folder`` its ``_rmdir``. A FileNotFoundError from ``rm_folder``
    counts as deleted, as object stores have no folders to delete.

    Parameters
    ----------
    fs : TreeFileSystem
        The filesystem to delete from.
    path : str or sequence of str
        The folder to delete, as ``_ils`` takes it. If it is a file, that file
        is deleted. If it is a glob pattern, or a list of paths and patterns,
        all they match is deleted.
    maxdepth : int, optional
        The deepest level to delete: 1 deletes only what is in ``path``, and
        ``path`` if it is then empty. No limit by default.

    Returns
    -------
    Operation
        Yielding each path deleted, or an error. Its state is an `RmState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    _tree_walk : The walk under it.
    """
    _check_maxdepth(maxdepth)
    return _Deletion.operation(fs=fs, path=path, maxdepth=maxdepth)


class _Deletion(_TreeJobs):
    """The jobs of one `_tree_rm`: a deletion per file, then per folder."""

    def __init__(self, *, path: str | Sequence[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)

        fs, maxdepth = self.fs, self.maxdepth
        self.path = path
        # A folder deleted whole would take what is deeper than maxdepth.
        self.rm_tree: Callable[[str], Awaitable[None]] | None = (
            getattr(fs, "_rm_tree", None) if maxdepth is None else None
        )

        # The roots deleted whole, yielded with the first step.
        self.whole: list[str] = []
        self.implicit_folders = bool(_profile_of(fs).implicit_folders)
        self.files_found = 0
        self.files_deleted = 0
        self.folders_deleted = 0

    async def prepare(self) -> _Roots:
        """Find the roots, then delete whole the folders that can be: the rest are walked."""
        if not isinstance(self.path, str) or has_magic(self.path):
            roots = await self.glob_roots(self.path)
            known = True  # The glob found the folders to be folders.
        else:
            roots = _Roots([self.path])
            known = False  # Of unknown kind, until asked.

        if self.rm_tree is None or self.op.cancelling():
            return roots

        sources = list(roots.folders)
        kinds = await asyncio.gather(*(self.delete_whole(s, known) for s in sources))
        self.whole = [s for s, k in zip(sources, kinds, strict=True) if k == "whole"]
        self.folders_deleted = len(self.whole)
        files = [s for s, k in zip(sources, kinds, strict=True) if k == "file"]

        return _Roots(
            [s for s, k in zip(sources, kinds, strict=True) if k == "walk"],
            [*roots.files, *files],
        )

    async def delete_whole(self, folder: str, known: bool) -> str:
        """Delete a root by one ``rm_tree``, and tell how it went: whole, file or walk.

        A root of unknown kind is asked first, as ``rm_tree`` takes folders
        only. A missing root, or a failed deletion, is walked instead: the walk
        then raises, or shows the failure on the files.
        """
        try:
            if not known:
                async with self.concurrency_limit:
                    if (await self.fs._info(folder))["type"] != "directory":
                        return "file"

            async with self.concurrency_limit:
                await self.rm_tree(folder)  # type: ignore[misc]
        except Exception:
            return "walk"

        return "whole"

    async def begin(
        self, roots: _Roots, source: AsyncGenerator[_Listing] | None
    ) -> list[Any]:
        whole, self.whole = self.whole, []
        return [*whole, *await super().begin(roots, source)]

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

    def at_limit(self, folder: str) -> None:
        # Deleted only if empty: one that is not fails, as fsspec's deletion
        # of each path found does. An implicit folder is no path to delete.
        if not self.implicit_folders:
            self.waiting.append(_Job("folder", folder))

    def add(self, listing: _Listing) -> None:
        self.track(listing)

        for entry in listing.entries:
            if entry["type"] != "directory":
                self.files_found += 1
                self.waiting.append(_Job("file", entry["name"].rstrip("/")))

        # Empty, or all in it deleted before its last part came.
        if listing.last and self.finished(listing.folder):
            self.remove(listing.folder)

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "folder":
            return self.fs._rmdir(job.source)
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
            self.remove(folder)

    def remove(self, folder: str) -> None:
        """Queue the deletion of an empty folder, unless it went with its last file."""
        if self.implicit_folders:
            self.after(folder)  # Gone already, as a FileNotFoundError tells.
        else:
            self.waiting.append(_Job("folder", folder))
