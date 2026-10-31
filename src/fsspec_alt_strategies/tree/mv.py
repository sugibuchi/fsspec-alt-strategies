# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive move that walks the tree and moves while it walks.

`tree_mv` is an alternative to fsspec's recursive ``mv``. fsspec's copies the
whole tree, then deletes it, each stage listing the tree first. `tree_mv` moves
each folder's files as soon as the folder is listed, each by a copy and a
delete, and deletes each source folder once it is empty.

It is for storages without a rename of their own. Where a storage renames a
file or a folder in one request, that request wins.

This module depends on `control`, `profile`, `walk`, `_jobs` and `cp`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..control import Operation, State
from ..profile import _resolve
from ._jobs import _Job
from .cp import _Copy

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fsspec.asyn import AsyncFileSystem

    from ..profile import StorageProfile
    from .walk import _Listing

__all__ = ["MvState", "tree_mv"]


@dataclass(frozen=True, kw_only=True)
class MvState(State):
    """The progress of a `tree_mv`.

    Attributes
    ----------
    files_found : int
        The files found so far. It grows as the walk goes on.
    files_moved : int
        The files moved so far.
    folders_moved : int
        The source folders deleted so far, once all in them was moved.
    """

    files_found: int = 0
    files_moved: int = 0
    folders_moved: int = 0


def tree_mv(
    fs: AsyncFileSystem,
    path1: str,
    path2: str,
    *,
    profile: StorageProfile | None = None,
    rm_folder: Callable[[str], Awaitable[Any]] | None = None,
    batch_size: int | None = None,
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

    The tree is walked as `tree_walk` walks it. Each folder's files are moved
    as soon as the folder is listed, each by a ``_cp_file`` then an
    ``_rm_file``. A source folder is deleted once everything in it is moved,
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
    raises FileNotFoundError if ``path1`` does not exist, and ValueError if
    ``path2`` is inside it.

    Requests: as `tree_cp`, with one ``_rm_file`` more per file, and one
    ``rm_folder`` per folder. At most ``batch_size`` listings and
    ``batch_size`` moves run at a time.

    The profile's ``parents_required=False`` saves the ``_makedirs`` of each
    folder with files, as their copies create it.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path1 : str
        The folder or file to move, or a glob pattern.
    path2 : str
        Where to move it, as above.
    profile : StorageProfile, optional
        What the storage does. The filesystem's ``storage_profile`` if None, if
        it has one.
    rm_folder : callable, optional
        An async function deleting an empty folder, as with `tree_rm`.
    batch_size : int, optional
        The most listings, and the most moves, at a time. fsspec's default if
        None.

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
    >>> async with tree_mv(fs, "data/2024", "archive/") as op:
    ...     async for moved in op:
    ...         if isinstance(moved, Exception):
    ...             print("failed:", moved)
    >>> op.state.files_moved
    1234
    """
    resolved = _resolve(fs, profile)
    return Operation(
        lambda op: _Move(
            fs, path1, path2, resolved, rm_folder, batch_size, op
        ).outcomes(),
        MvState(),
    )


class _Move(_Copy):
    """The jobs of one `tree_mv`: a copy's, deleting each source as it goes."""

    verb = "move"
    doing = "Moving"

    def __init__(
        self,
        fs: AsyncFileSystem,
        path1: str,
        path2: str,
        profile: StorageProfile,
        rm_folder: Callable[[str], Awaitable[Any]] | None,
        batch_size: int | None,
        op: Operation[Any, Any],
    ) -> None:
        super().__init__(fs, path1, path2, profile, batch_size, op)
        self.rm_folder = rm_folder or getattr(fs, "_rmdir", None) or fs._rm_file
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
            await self.fs._rm_file(source)
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
