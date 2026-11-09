# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive move that walks the tree and moves while it walks.

`_tree_mv` is an alternative to fsspec's recursive ``mv``. fsspec's copies the
whole tree, then deletes it, each stage listing the tree first. `_tree_mv` moves
each folder's files as soon as the folder is listed, each by one ``mv_file``,
and deletes each source folder once it is empty.

Where ``mv_file`` moves a folder too, in one request, as a storage's rename
does, the profile says so. `_tree_mv` then moves each folder whole with it
where it can, and walks only what it cannot move so.

This module depends on `cp`, and `jobs` and `paths` of `common`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import MvState

from ..common.jobs import _Job, _Roots, _transfer_of
from ..common.paths import _check_maxdepth
from .cp import _CopyOrMove

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Sequence

    from ..common.jobs import _Transfer
    from ..common.operation import Operation
    from .walk import _Listing

__all__: list[str] = []


def _tree_mv(
    src_fs: Any,
    dst_fs: Any,
    path1: str | Sequence[str],
    path2: str,
    *,
    mv_file: _Transfer | None = None,
    same_storage: bool | None = None,
    maxdepth: int | None = None,
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
    ``path2``. The pattern is matched as `_tree_glob` matches it, then the
    folders matched are walked at once.

    A list of paths and patterns moves each path it names or matches into
    ``path2``, under its name, with all in it, as `_tree_cp` copies it. Unlike
    fsspec's ``mv``, it keeps the folders' trees. A path whose name another
    path takes first yields a FileExistsError, and stays where it is.

    The tree is walked as `_tree_walk` walks it. Each folder's files are moved
    as soon as the folder is listed, each by one ``mv_file``. A source folder
    is deleted once everything in it is moved, so the deepest go first. A
    folder is created at the destination before files are moved into it, and
    for each empty folder.

    The destination is described by ``dst_fs`` and its
    ``storage_profile``, unknown if it has none. Within
    one storage, ``dst_fs`` is ``src_fs``, and ``same_storage`` is True.
    Between two, ``mv_file`` copies a file to the other storage, then
    deletes it: e.g. the router's ``_cross_fs_mv_file``. A path may then go
    to the same path, or under it, and nothing is moved whole.

    Unlike fsspec's ``mv``, it keeps going past a failure, and reports each. A
    file that failed to move stays where it was, and so do the folders above
    it. A storage refusing to overwrite yields a FileExistsError for each
    file already at its destination, as fsspec's ``mv`` raises one.

    It wins over fsspec's ``mv`` on storages that list one folder per request
    and rename nothing in one request. The moves then start after the first
    listing, not after the whole tree is listed twice.

    Within one storage, with the profile's ``renames_folders=True``,
    ``mv_file`` moves a folder with all in it, in one request. Each folder root is then first moved whole
    by one ``mv_file``, where it can be. That wins by far on storages renaming
    a folder at once, such as Azure Data Lake Storage Gen2, HDFS or a local
    disk. A root goes whole only to a destination that does not exist, as a
    rename onto an existing path differs among storages. Into an existing
    folder, its contents merge, and its files are moved one by one. So are
    those of a root whose move fails, e.g. across containers, and those of
    ``path1`` ending with "/", whose contents go onto ``path2``. A root moved
    whole is yielded once, as ``(root, target)``: what is in it is neither
    listed nor yielded.

    With ``maxdepth``, it moves the files `_tree_cp` copies with it, and
    nothing is moved whole. A source folder is deleted once emptied, so those
    keeping a folder at the limit stay, and empty ones stay too. fsspec's
    ``mv`` copies as its ``copy`` does with ``maxdepth``, but then deletes the
    whole source tree, deeper files too. This keeps them.

    Once cancelled, it starts no new listing or move, and yields the outcomes
    of the moves already sent. Closed, it waits for those moves without
    yielding them, as their outcome would otherwise be unknown. Iterating it
    raises FileNotFoundError if ``path1`` does not exist, or matches nothing,
    and ValueError if ``path2`` is inside it within one storage.

    Requests: as `_tree_cp`, with one ``mv_file`` per file instead of its
    ``cp_file``, and one ``rm_folder`` per source folder. With
    ``renames_folders``, each folder root first costs a ``dst_info`` of its
    target, unless ``path2`` was found missing already, a ``dst_makedirs`` of
    its parent, and its ``mv_file``: a root moved whole costs nothing more.
    Each request holds a permit of the source's ``operation_limit`` while it
    runs.

    The destination's profile's ``parents_required=False`` saves the ``dst_makedirs`` of
    each folder with files, as their moves create it. Its
    ``implicit_folders=True`` saves those, and within one storage the
    ``rm_folder`` of each source folder too, as a folder goes with its last
    file. Either saves the ``dst_makedirs`` before a folder is moved whole.

    The primitives are called above by what they do: ``_ils``, ``info`` and
    ``rm_folder`` are the source's ``_ils``, ``_info`` and ``_rmdir``,
    ``dst_info`` and ``dst_makedirs`` the destination's ``_info`` and
    ``_makedirs``.

    ``src_fs`` is the filesystem moved from, and ``dst_fs`` the one moved to,
    ``src_fs`` again within one storage. ``path1`` is a folder, a file, a glob
    pattern, or a list of them, as the source's ``_ils`` takes paths.
    ``mv_file(source, destination)`` moves a file: the source's ``_mv_file``
    by default, within one storage. fsspec's own ``_mv_file`` copies, then
    deletes. A storage renaming a path in one request overrides it so, and may
    move a folder with it too, as the profile's ``renames_folders`` declares.
    ``same_storage`` is whether ``dst_fs`` is ``src_fs``, unless given.
    ``maxdepth`` of 1 moves only the files in ``path1``.

    The operation yields ``(source, destination)`` for each file and folder
    moved, or an error, and its state is an `MvState`. The call raises
    ValueError if ``maxdepth`` is less than 1, or ``mv_file`` is missing
    between two storages. `_tree_cp` does the same, keeping the source.
    """
    _check_maxdepth(maxdepth)
    mv_file, same_storage = _transfer_of(
        src_fs, dst_fs, mv_file, same_storage, "mv_file"
    )
    return _Move.operation(
        fs=src_fs,
        dst_fs=dst_fs,
        path1=path1,
        path2=path2,
        same_storage=same_storage,
        maxdepth=maxdepth,
        mv_file=mv_file,
    )


class _Move(_CopyOrMove):
    """The jobs of one `_tree_mv`: a ``mv_file`` per file, deleting each folder emptied."""

    verb = "move"
    doing = "Moving"

    def __init__(self, *, mv_file: _Transfer, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.mv_file = mv_file

        # The profile is the destination's: only within one storage does it
        # tell how the source renames, or deletes its folders. A folder moved
        # whole would take what is deeper than maxdepth along.
        self.renames_folders = (
            self.same_storage
            and bool(self.profile.renames_folders)
            and self.maxdepth is None
        )
        self.implicit_folders = self.same_storage and self.implicit_folders
        self.folders_moved = 0

        # The roots moved whole, yielded with the first step.
        self.renamed: list[tuple[str, str]] = []
        self.roots_renamed = 0 if self.renames_folders else None

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
        """Place the roots, then move whole the folders that can be: the rest are walked.

        A file root goes by one ``mv_file`` all the same, as the walk moves it.
        """
        roots = await super().prepare()
        if roots is None or not self.renames_folders:
            return roots

        if isinstance(self.path1, str) and self.path1.endswith("/"):
            return roots  # Its contents go onto path2, among what is there.

        sources = list(roots.folders)
        moved = await asyncio.gather(*(self.move_whole(s) for s in sources))
        whole = {s for s, ok in zip(sources, moved, strict=True) if ok}
        self.renamed = [(s, self.target(s)) for s in sources if s in whole]
        self.roots_renamed = len(self.renamed)

        return _Roots([f for f in roots.folders if f not in whole], roots.files)

    async def move_whole(self, source: str) -> bool:
        """Move a root with one ``mv_file``, and tell whether it went.

        A root whose target exists merges into it, and a failed move moved
        nothing: both are walked instead, and a real failure then shows on its
        files.
        """
        target = self.target(source)
        parent = target.rpartition("/")[0]

        if target not in self.missing:
            try:
                async with self.concurrency_limit:
                    await self.dst_fs._info(target)
            except FileNotFoundError:
                pass  # Free: the root can go there whole.
            except Exception:
                return False
            else:
                return False  # Taken: the root merges into it.

        try:
            if parent and self.parents_required:
                async with self.concurrency_limit:
                    await self.dst_fs._makedirs(parent, exist_ok=True)

            async with self.concurrency_limit:
                await self.mv_file(source, target)
        except Exception:
            return False

        return True

    async def begin(
        self, roots: _Roots, source: AsyncGenerator[_Listing] | None
    ) -> list[Any]:
        renamed, self.renamed = self.renamed, []
        return [*renamed, *await super().begin(roots, source)]

    def add(self, listing: _Listing) -> list[Any] | None:
        self.track(listing)
        super().add(listing)

        # All in it moved before its last part came. An empty folder goes once
        # made at the destination, in done.
        empty = listing.first and not listing.entries
        if listing.last and not empty and self.finished(listing.folder):
            return self.remove(listing.folder)
        return None

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "rmdir":
            return self.fs._rmdir(job.source)
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
            if job.source and self.finished(job.source):
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
            return self.done(job)  # Already gone, as _tree_rm counts it.
        error.add_note(f"Deleting the folder {job.source!r} failed.")
        return [error]
