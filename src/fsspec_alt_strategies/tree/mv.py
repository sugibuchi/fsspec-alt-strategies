# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive move that walks the tree and moves while it walks.

`tree_mv` is an alternative to fsspec's recursive ``mv``. fsspec's copies the
whole tree, then deletes it, each stage listing the tree first. `tree_mv` moves
each folder's files as soon as the folder is listed, each by a copy and a
delete, and deletes each source folder once it is empty.

It is for storages without a rename of their own. Where a storage renames a
file or a folder in one request, that request wins.

This module depends on `control`, `profile`, `cp`, and `jobs` and `states` of
`common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.jobs import _Job
from ..common.states import MvState
from ..control import Operation
from ..profile import StorageProfile
from .cp import _Copy

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Sequence

    from ..common.operations import _Info, _Ls, _Request
    from .walk import _Listing

__all__ = ["MvState", "tree_mv"]


def tree_mv(
    path1: str | Sequence[str],
    path2: str,
    *,
    ls: _Ls,
    info: _Info,
    cp_file: _Request,
    makedirs: _Request,
    rm_file: _Request,
    rm_folder: _Request,
    profile: StorageProfile | None = None,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[tuple[str, str], MvState]:
    """Move a folder and everything in it, moving while the tree is walked.

    It places the paths as fsspec's ``mv`` with ``recursive=True`` does. If
    ``path2`` is an existing folder, or ends with "/", and ``path1`` does not
    end with "/", ``path1`` goes into it: ``path1/x`` becomes
    ``path2/<name of path1>/x``. Otherwise ``path1`` becomes ``path2``:
    ``path1/x`` becomes ``path2/x``. A file moves the same way. Existing files
    are overwritten, as fsspec's ``mv`` does.

    A glob pattern moves each path it matches, and all in it, placed as
    fsspec's ``mv`` places them. Their paths below the matches' common
    prefix are kept under ``path2``, but a single file matched goes into
    ``path2``. The pattern is matched as `tree_glob` matches it, then the
    folders matched are walked at once.

    A list of paths and patterns moves each path it names or matches into
    ``path2``, under its name, with all in it, as `tree_cp` copies it. Unlike
    fsspec's ``mv``, it keeps the folders' trees. A path whose name another
    path takes first yields a FileExistsError, and stays where it is.

    The tree is walked as `tree_walk` walks it. Each folder's files are moved
    as soon as the folder is listed, each by a ``cp_file`` then an
    ``rm_file``. A source folder is deleted once everything in it is moved,
    so the deepest go first. A folder is created at the destination before
    files are copied into it, and for each empty folder.

    Unlike fsspec's ``mv``, it keeps going past a failure, and reports each. A
    file that failed to copy stays where it was, and so do the folders above
    it. A file copied but not deleted is then in both places, as its error
    notes.

    It wins over fsspec's ``mv`` on storages that list one folder per request
    and rename nothing in one request. The moves then start after the first
    listing, not after the whole tree is listed twice.

    Once cancelled, it starts no new listing or move, and yields the outcomes
    of the moves already sent. Closed, it waits for those moves without
    yielding them, as their outcome would otherwise be unknown. Iterating it
    raises FileNotFoundError if ``path1`` does not exist, or matches nothing,
    and ValueError if ``path2`` is inside it.

    Requests: as `tree_cp`, with one ``rm_file`` more per file, and one
    ``rm_folder`` per folder. Each request holds a permit of
    ``concurrency_limit`` while it runs.

    The profile's ``parents_required=False`` saves the ``makedirs`` of each
    folder with files, as their copies create it.

    Parameters
    ----------
    path1 : str or sequence of str
        The folder or file to move, a glob pattern, or a list of them, as
        ``ls`` takes paths.
    path2 : str
        Where to move it, as above.
    ls, info, cp_file, makedirs : callable
        The storage's operations, as with `tree_cp`.
    rm_file : callable
        Deletes a file, called as ``rm_file(path)``: e.g. the filesystem's
        ``_rm_file``.
    rm_folder : callable
        Deletes an empty folder, as with `tree_rm`.
    profile : StorageProfile, optional
        What the storage does. Unknown if None.
    concurrency_limit : asyncio.Semaphore
        Held by each request while it runs. Share one to limit several
        operations, or files, together.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each file and folder moved, or
        an error. Its state is an `MvState`.

    See Also
    --------
    tree_cp : The same, keeping the source.
    tree_rm : The deletion it does as it goes.

    Examples
    --------
    >>> mv = tree_mv(
    ...     "data/2024",
    ...     "archive/",
    ...     ls=fs._ls,
    ...     info=fs._info,
    ...     cp_file=fs._cp_file,
    ...     makedirs=fs._makedirs,
    ...     rm_file=fs._rm_file,
    ...     rm_folder=fs._rmdir,
    ...     concurrency_limit=asyncio.Semaphore(32),
    ... )
    >>> async with mv as op:
    ...     async for moved in op:
    ...         if isinstance(moved, Exception):
    ...             print("failed:", moved)
    >>> op.state.files_moved
    1234
    """
    return Operation(
        lambda op: _Move(
            path1,
            path2,
            ls=ls,
            info=info,
            cp_file=cp_file,
            makedirs=makedirs,
            rm_file=rm_file,
            rm_folder=rm_folder,
            profile=profile or StorageProfile(),
            concurrency_limit=concurrency_limit,
            op=op,
        ).outcomes(),
        MvState(folders_moved=0),
    )


class _Move(_Copy):
    """The jobs of one `tree_mv`: a copy's, deleting each source as it goes."""

    verb = "move"
    doing = "Moving"

    def __init__(
        self,
        path1: str | Sequence[str],
        path2: str,
        *,
        rm_folder: _Request,
        **kwargs: Any,
    ) -> None:
        super().__init__(path1, path2, **kwargs)
        self.rm_folder = rm_folder
        self.folders_moved = 0

    def state(self) -> MvState:
        return MvState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_moved=self.files_done,
            folders_moved=self.folders_moved,
        )

    def add(self, listing: _Listing) -> None:
        self.track(listing)
        super().add(listing)

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "rmdir":
            return self.rm_folder(job.source)
        return super().run(job)

    async def copy_file(self, source: str, destination: str) -> None:
        await super().copy_file(source, destination)
        try:
            await self.rm_file(source)
        except FileNotFoundError:
            pass  # Deleted by another: the file is moved all the same.
        except Exception as e:
            e.add_note(
                f"Copied {source!r} to {destination!r}, but deleting {source!r}"
                " failed: the file is in both places."
            )
            raise

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            outcome = super().done(job)
            # An empty folder: made at the destination, it can go.
            if job.source and self.remaining[job.source] == 0:
                self.waiting.append(_Job("rmdir", job.source, job.destination))
            return outcome
        if job.kind == "rmdir":
            self.folders_moved += 1
        else:
            self.files_done += 1
        if (folder := self.child_done(job.source)) is not None:
            self.waiting.append(_Job("rmdir", folder, self.target(folder)))
        return [(job.source, job.destination)]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if job.kind != "rmdir":
            return super().failed(job, error)
        if isinstance(error, FileNotFoundError):
            return self.done(job)  # Already gone, as tree_rm counts it.
        error.add_note(f"Deleting the folder {job.source!r} failed.")
        return [error]
