# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive move that walks the tree and moves while it walks.

`tree_mv` is an alternative to fsspec's recursive ``mv``. fsspec's copies the
whole tree, then deletes it, each stage listing the tree first. `tree_mv` moves
each folder's files as soon as the folder is listed, each by one ``mv_file``,
and deletes each source folder once it is empty.

Where a storage renames a file or a folder in one request, `tree_mv` takes
that rename, and moves each root whole with it where it can. It walks only
what it cannot move so.

This module depends on `control`, `profile`, `cp`, and `jobs` and `states` of
`common`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..common.jobs import _Job, _Roots
from ..common.states import MvState
from ..control import Operation
from ..profile import StorageProfile
from .cp import _Transfer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Sequence

    from ..common.operations import _Info, _Ls, _Request
    from .walk import _Listing

__all__ = ["MvState", "tree_mv"]


def tree_mv(
    path1: str | Sequence[str],
    path2: str,
    *,
    ls: _Ls,
    info: _Info,
    mv_file: _Request,
    makedirs: _Request,
    rm_folder: _Request,
    rename: _Request | None = None,
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
    as soon as the folder is listed, each by one ``mv_file``. A source folder is deleted once everything in it is moved,
    so the deepest go first. A folder is created at the destination before
    files are moved into it, and for each empty folder.

    Unlike fsspec's ``mv``, it keeps going past a failure, and reports each. A
    file that failed to move stays where it was, and so do the folders above
    it. A storage refusing to overwrite yields a FileExistsError for each
    file already at its destination, as fsspec's ``mv`` raises one.

    It wins over fsspec's ``mv`` on storages that list one folder per request
    and rename nothing in one request. The moves then start after the first
    listing, not after the whole tree is listed twice.

    Given ``rename``, a storage's own move of a whole folder or file, each root
    is first moved whole, in one request, where it can be. That wins by far on
    storages renaming a folder at once, such as Azure Data Lake Storage Gen2,
    HDFS or a local disk. A root goes whole only to a destination that does not
    exist, as a rename onto an existing path differs among storages. Into an
    existing folder, its contents merge, and its paths are moved one by one, as
    without ``rename``. So are those of a root whose ``rename`` fails, e.g.
    across containers, and those of ``path1`` ending with "/", whose contents go
    onto ``path2``. A root moved whole is yielded once, as ``(root, target)``:
    what is in it is neither listed nor yielded.

    Once cancelled, it starts no new listing or move, and yields the outcomes
    of the moves already sent. Closed, it waits for those moves without
    yielding them, as their outcome would otherwise be unknown. Iterating it
    raises FileNotFoundError if ``path1`` does not exist, or matches nothing,
    and ValueError if ``path2`` is inside it.

    Requests: as `tree_cp`, with one ``mv_file`` per file instead of its
    ``cp_file``, and one ``rm_folder`` per folder. Given ``rename``, each root first costs an
    ``info`` of its target, unless ``path2`` was found missing already, a
    ``makedirs`` of its parent, and the ``rename``: a root moved whole costs
    nothing more. Each request holds a permit of ``concurrency_limit`` while
    it runs.

    The profile's ``parents_required=False`` saves the ``makedirs`` of each
    folder with files, as their moves create it. Its ``implicit_folders=True``
    saves those, and the ``rm_folder`` of each source folder too, as a folder
    goes with its last file. Either saves the ``makedirs`` before a
    ``rename``.

    Parameters
    ----------
    path1 : str or sequence of str
        The folder or file to move, a glob pattern, or a list of them, as
        ``ls`` takes paths.
    path2 : str
        Where to move it, as above.
    ls, info, makedirs : callable
        The storage's operations, as with `tree_cp`.
    mv_file : callable
        Moves a file, called as ``mv_file(source, destination)``: e.g. the
        filesystem's ``_mv_file``. fsspec's own copies, then deletes. A storage
        renaming a file in one request overrides it so.
    rm_folder : callable
        Deletes an empty folder, as with `tree_rm`.
    rename : callable, optional
        Moves a folder, with all in it, or a file, in one request, called as
        ``rename(path1, path2)``. It raises if it cannot, moving nothing. It is
        no fsspec method: a storage's own, e.g. its rename of a path.
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
    ...     mv_file=fs._mv_file,
    ...     makedirs=fs._makedirs,
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
            mv_file=mv_file,
            makedirs=makedirs,
            rm_folder=rm_folder,
            rename=rename,
            profile=profile or StorageProfile(),
            concurrency_limit=concurrency_limit,
            op=op,
        ).outcomes(),
        MvState(folders_moved=0, roots_renamed=None if rename is None else 0),
    )


class _Move(_Transfer):
    """The jobs of one `tree_mv`: a ``mv_file`` per file, deleting each folder emptied."""

    verb = "move"
    doing = "Moving"

    def __init__(
        self,
        path1: str | Sequence[str],
        path2: str,
        *,
        mv_file: _Request,
        rm_folder: _Request,
        rename: _Request | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(path1, path2, **kwargs)
        self.mv_file = mv_file
        self.rm_folder = rm_folder
        self.rename = rename
        self.folders_moved = 0
        # The roots moved whole, yielded with the first step.
        self.renamed: list[tuple[str, str]] = []
        self.roots_renamed = None if rename is None else 0

    def state(self) -> MvState:
        return MvState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_moved=self.files_done,
            folders_moved=self.folders_moved,
            roots_renamed=self.roots_renamed,
        )

    async def prepare(self) -> _Roots | None:
        """Place the roots, then move whole those that can be: the rest are walked."""
        roots = await super().prepare()
        if roots is None or self.rename is None:
            return roots
        if isinstance(self.path1, str) and self.path1.endswith("/"):
            return roots  # Its contents go onto path2, among what is there.
        sources = [*roots.folders, *roots.files]
        moved = await asyncio.gather(*(self.move_whole(s) for s in sources))
        whole = {s for s, ok in zip(sources, moved, strict=True) if ok}
        self.renamed = [(s, self.target(s)) for s in sources if s in whole]
        self.roots_renamed = len(self.renamed)
        return _Roots(
            [f for f in roots.folders if f not in whole],
            [f for f in roots.files if f not in whole],
            roots.errors,
        )

    async def move_whole(self, source: str) -> bool:
        """Move a root with one ``rename``, and tell whether it went.

        A root whose target exists merges into it, and a failed rename moved
        nothing: both are walked instead, and a real failure then shows on its
        paths, as without ``rename``.
        """
        target = self.target(source)
        parent = target.rpartition("/")[0]
        if target not in self.missing:
            try:
                async with self.concurrency_limit:
                    await self.info(target)
            except FileNotFoundError:
                pass  # Free: the root can go there whole.
            except Exception:
                return False
            else:
                return False  # Taken: the root merges into it.
        try:
            if parent and self.parents_required:
                async with self.concurrency_limit:
                    await self.makedirs(parent, exist_ok=True)
            async with self.concurrency_limit:
                await self.rename(source, target)  # type: ignore[misc]
        except Exception:
            return False
        return True

    async def begin(
        self, roots: _Roots, source: AsyncGenerator[_Listing] | None
    ) -> list[Any]:
        renamed, self.renamed = self.renamed, []
        return [*renamed, *await super().begin(roots, source)]

    def add(self, listing: _Listing) -> None:
        self.track(listing)
        super().add(listing)

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "rmdir":
            return self.rm_folder(job.source)
        return super().run(job)

    async def transfer_file(self, source: str, destination: str) -> None:
        try:
            await self.mv_file(source, destination)
        except Exception as e:
            # Yielded among paths, the error must tell its paths itself.
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            outcome = super().done(job)
            # An empty folder: made at the destination, it can go.
            if job.source and self.remaining[job.source] == 0:
                outcome += self.remove(job.source)
            return outcome
        if job.kind == "rmdir":
            self.folders_moved += 1
        else:
            self.files_done += 1
        outcome: list[Any] = [(job.source, job.destination)]
        if (folder := self.child_done(job.source)) is not None:
            outcome += self.remove(folder)
        return outcome

    def remove(self, folder: str) -> list[Any]:
        """Delete a source folder now empty, and return what that moved at once.

        An implicit folder went with its last file: it is moved, with no
        request, as one found already gone is.
        """
        job = _Job("rmdir", folder, self.target(folder))
        if self.implicit_folders:
            return self.done(job)
        self.waiting.append(job)
        return []

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if job.kind != "rmdir":
            return super().failed(job, error)
        if isinstance(error, FileNotFoundError):
            return self.done(job)  # Already gone, as tree_rm counts it.
        error.add_note(f"Deleting the folder {job.source!r} failed.")
        return [error]
