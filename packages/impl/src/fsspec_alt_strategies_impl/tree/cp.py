# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy that walks the tree and copies while it walks.

`_tree_cp` is an alternative to fsspec's recursive ``copy``. fsspec's lists the
whole tree first, then copies each path. `_tree_cp` copies each directory's files
as soon as the directory is listed. The destination may be another storage:
within one, it is the same filesystem given twice.

Where the filesystem's ``cp_tree`` copies a directory in one request, the
profile says so. `_tree_cp` then copies each root whole with it where it can,
and walks only what it cannot copy so.

This module depends on `_jobs`, and `jobs` and `paths` of `common`.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import CpState

from ..common.jobs import _CopyJobs, _Job, _transfer_of
from ..common.paths import _check_maxdepth
from ._jobs import _TreeJobs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from ..common.handler import Entry, StorageHandler, StoragePath
    from ..common.jobs import _Roots, _Transfer
    from ..common.operation import Operation
    from .handler import TreeFileOperationHandler
    from .walk import _Listing

__all__: list[str] = []


def _tree_cp(
    src: TreeFileOperationHandler,
    dst: StorageHandler,
    path1: StoragePath | Sequence[StoragePath],
    path2: StoragePath,
    *,
    cp_entry: _Transfer | None = None,
    same_storage: bool | None = None,
    maxdepth: int | None = None,
    root_entry: dict[str, Any] | None = None,
    ils_kwargs: dict[str, Any] | None = None,
) -> Operation[tuple[str, str], CpState]:
    """Copy a directory and everything in it, copying while the tree is walked.

    It places the paths as fsspec's ``copy`` with ``recursive=True`` does. If
    ``path2`` is an existing directory, or ends with "/", and ``path1`` does not
    end with "/", ``path1`` goes into it: ``path1/x`` becomes
    ``path2/<name of path1>/x``. Otherwise ``path1`` becomes ``path2``:
    ``path1/x`` becomes ``path2/x``. A file is copied the same way. Existing
    files are overwritten, as fsspec's ``copy`` does.

    A glob pattern copies each path it matches, and all in it, placed as
    fsspec's ``copy`` places them. Their paths below the matches' common
    prefix are kept under ``path2``, but a single file matched goes into
    ``path2``. The pattern is matched as `_tree_glob` matches it, then the
    directories matched are walked at once.

    A list of paths and patterns copies each path it names or matches into
    ``path2``, under its name, with all in it. That is how ``cp -r`` places
    them. fsspec's ``copy`` places the files named as this does. It also
    flattens the directories' trees, though: each file in them goes straight into
    ``path2``. Two paths of the same name would go to the same place. The
    first, in path order, is copied, and each other yields a FileExistsError
    instead, as GNU's ``cp`` reports it. fsspec's ``copy`` overwrites one
    with the other. A path inside another one in the list is copied with that
    one only.

    The tree is walked as `_tree_walk` walks it. Each directory's files are
    copied as soon as the directory is listed, each by a ``cp_entry``. A
    directory is created at the destination before files are copied into it,
    and for each empty directory.

    With ``maxdepth``, it copies the files fsspec's ``copy`` copies with it:
    those at most ``maxdepth`` levels below ``path1``, or below a pattern's
    matches one level less. Directories deeper are not listed. As fsspec's, it
    then makes no empty directory. Unlike fsspec's, it places the files as
    without ``maxdepth``: fsspec places them by the common prefix of the files
    found, which moves them when all are in one subdirectory.

    The destination is described by ``dst`` and its ``profile``,
    unknown if it has none. Within one storage, ``dst`` is ``src``, and
    ``same_storage`` is True: a path copied onto itself, or under itself, is
    refused. Between two, ``cp_entry`` copies from one to the other, by default
    `transfer._across`, its data through the caller. A
    path may then go to the same path, or under it. The copies run
    concurrently, so the destination's paths are created in another order than
    the source's. A storage listing in creation order shows it.

    Unlike fsspec's ``copy``, it keeps going past a failure, and reports each.
    A copy cancelled or failed leaves what was copied so far: it deletes
    nothing.

    It wins over fsspec's ``copy`` on storages that list one directory per
    request. The copies then start after the first listing, not after the
    whole tree is listed. Between two storages, it wins over fsspec's way too,
    a ``get`` to local files then a ``put``: the data needs no local disk.

    Within one storage, with the profile's ``copies_directories=True``,
    ``cp_tree`` copies a directory with all in it, in one request. Each
    directory root is then copied whole by one ``cp_tree``, where it can be,
    as `_tree_mv` moves one by ``mv_tree``: only to a destination that does
    not exist, and never with ``maxdepth``. That wins by far on storages
    copying a folder at once, such as Dropbox. A root whose copy fails yields
    the error, with its cause. A root copied whole is yielded once, as
    ``(root, target)``: what is in it is neither listed nor yielded.

    With the profile's ``case_sensitive=False``, within one storage, paths
    are compared folded. A path onto itself in another case is left as it
    is, as one in the same case is, and one into itself is refused.

    Once cancelled, it starts no new listing or copy, and yields the outcomes
    of the copies already sent. Closed, it waits for those copies without
    yielding them. Iterating it raises FileNotFoundError if ``path1`` does not
    exist, or matches nothing, and ValueError if ``path2`` is inside it within
    one storage.

    Requests: one ``dst_info`` for ``path2``, unless it ends with "/" or
    ``path1`` is a pattern or a list. Patterns, or a list, add the requests of
    their `_tree_glob`. One ``ils_recursive`` per root, which a storage
    without a recursive listing answers with one per directory, or one
    ``ils`` per directory with ``maxdepth``. One ``cp_entry`` per file, and
    one ``dst_makedirs`` per destination directory with files or empty. A
    destination refusing to overwrite costs a ``dst_info``, a ``dst_rm_file``
    and a ``cp_entry`` more per file it refused. The ``dst_info`` checks it is
    a file, as ``dst_rm_file`` is given only paths known to be files. Each
    request holds a permit of the source's ``operation_limit`` while it runs,
    a ``cp_entry`` between two storages too. With ``copies_directories``, each
    directory root first costs a ``dst_info`` of its target, unless ``path2``
    was found missing or existing already. One copied whole then costs a
    ``dst_makedirs`` of its parent, and its ``cp_tree``: nothing more.

    The destination's profile's ``parents_required=False`` between two
    storages, ``copy_parents_required=False`` within one, or
    ``implicit_directories=True``, saves the ``dst_makedirs`` of each directory
    with files, as their copies create it. Its ``parents_required=True``
    makes every directory, each once its parent is made, as such a storage
    makes a missing parent by more requests. That is one ``dst_makedirs``
    per directory, those holding only directories too, unless ``maxdepth``
    is given. A directory failing to be made then leaves out all below it,
    reported by its error alone. A directory known to exist needs
    none: the ``dst_info`` of ``path2`` shows it or its parent, and within one
    storage, ``path1``'s parent exists.

    ``root_entry``, the entry of ``path1`` as the caller found it, saves the
    listing of a single file: it is copied as the entry says.

    The primitives are named above as the handlers' own: ``ils`` and
    ``cp_tree`` are the source's, and ``dst_info``, ``dst_makedirs`` and
    ``dst_rm_file`` the destination's ``info``, ``makedirs`` and ``rm_file``.

    ``src`` is the handler copied from, and ``dst`` the one copied
    to, ``src`` again within one storage. ``path1`` is a directory, a file,
    a glob pattern, or a list of them, as the source's ``ils`` takes paths.
    ``cp_entry(entry, destination)`` copies a file: by default, the source's
    ``cp_file`` within one storage, and `transfer._across` between two. ``same_storage`` is
    whether ``dst`` is ``src``, unless given. ``maxdepth`` of 1 copies
    only the files in ``path1``. ``ils_kwargs`` go to each of the source's
    ``ils``, a pattern's glob's too.

    Within one storage, unless ``cp_entry`` is given, the files go to the
    source's ``cp_files`` in batches of its ``cp_batch_size``: by default each
    by ``cp_file``. A file joins a batch once its destination's directory is
    ready, whatever the directory, and a batch goes once full, or once the walk
    has ended with no directory left to make.

    The operation yields ``(source, destination)`` for each file copied, or an
    error, and its state is a `CpState`. The call raises ValueError if
    ``maxdepth`` is less than 1. `_tree_mv` does the same, deleting the source as it goes.
    """
    _check_maxdepth(maxdepth)
    given = cp_entry
    cp_entry, same_storage = _transfer_of(src, dst, cp_entry, same_storage)
    return _Copy.operation(
        batching=same_storage and given is None,
        handler=src,
        dst=dst,
        path1=path1,
        path2=path2,
        same_storage=same_storage,
        maxdepth=maxdepth,
        cp_entry=cp_entry,
        root_entry=root_entry,
        ils_kwargs=ils_kwargs,
    )


class _CopyOrMove(_CopyJobs, _TreeJobs):
    """The jobs of a copy or a move: a directory made where needed, and one per file.

    The tree walked is the source's, ``handler``. A file's job waits for its
    destination directory, when the storage needs it. A subclass says how a file
    goes, in `transfer_file`.

    Where the storage copies or moves a directory in one request, as the
    subclass reads it from the profile into ``goes_whole``, each directory
    root goes whole where it can, by `transfer_whole`. The rest is walked.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.goes_whole = False  # Until the subclass's setup reads the profile.
        self.refused_dirs: set[str] = (
            set()
        )  # Those whose names the destination refused.
        self.roots_whole: int | None = None

    async def prepare(self) -> _Roots | None:
        """Place the roots, and queue each directory that can go whole: the rest are walked.

        A root can go whole to a target found missing. One whose target
        exists merges into it, so it is walked. One whose target could not be
        asked yields that error, with its cause, and stays where it is.
        """
        roots = await super().prepare()
        if roots is None or not self.goes_whole:
            return roots

        if isinstance(self.path1, str) and self.path1.endswith("/"):
            return roots  # Its contents go onto path2, among what is there.

        sources = list(roots.directories)
        whole = await asyncio.gather(*(self.whole(s) for s in sources))
        walked = []
        for root, outcome in zip(sources, whole, strict=True):
            if isinstance(outcome, Exception):
                self.found.append(outcome)
                continue
            if not outcome:
                walked.append(root)
                continue

            # A job each, as a file's, after its parent is made: the storage's
            # request is not to make it.
            target = self.target(root["name"])
            parent = self.dst.parent(target)
            job = _Job("whole", root, target)
            self.queue(parent, "", [job], always=not self.implicit_directories)

        self.roots_whole = 0
        return roots._replace(directories=walked)

    async def whole(self, root: dict[str, Any]) -> bool | Exception:
        """Whether a root can go whole, as its target is missing.

        An error asking is returned, noted: the root's, as its walk would only
        meet it again on each file.
        """
        source = root["name"]
        target = self.target(source)
        if target in self.missing:
            return True
        if target in self.existing:
            return False
        try:
            async with self.concurrency_limit:
                await self.dst.info(self.dst.entry(target))
        except FileNotFoundError:
            return True
        except Exception as e:
            e.add_note(f"{self.doing} {source!r} to {target!r} failed.")
            return e
        else:
            return False

    def file_root(self, entry: dict[str, Any]) -> None:
        self.files_found += 1
        path = entry["name"]
        target = self.target(path)
        job = _Job("copy", entry, target)
        self.queue(self.dst.parent(target), "", [job])

    def add(self, listing: _Listing) -> list[Any] | None:
        try:
            directory = self.target(listing.directory)
        except ValueError as e:
            # A name the destination refuses, e.g. "..": nothing in it goes. A
            # directory below one refused is no error of its own.
            above = self.below_refused(listing.directory)
            self.refused_dirs.add(listing.directory)
            if listing.first and not above:
                return [self.refuse_name(listing.directory, e)]
            return None

        copies, refused = [], []
        for entry in listing.entries:
            if entry["type"] != "directory":
                name = entry["name"].rstrip("/")
                self.files_found += 1
                try:
                    target = self.target(name)
                except ValueError as e:
                    refused.append(self.refuse_name(name, e))
                    continue
                # The entry goes with its copy, so that cp_entry need not ask it.
                copies.append(_Job("copy", {**entry, "name": name}, target))

        if listing.first and listing.last and not listing.entries:
            # An empty directory is made, whatever the storage. With maxdepth,
            # fsspec's copy leaves out directories, and copies files only.
            if self.maxdepth is not None:
                return None
            self.queue(directory, listing.directory, [], always=True)
        elif copies or (listing.first and self.parents_first):
            # Parents first, each directory is made once its parent is. Else, a
            # directory holding only directories needs no mkdir: theirs create it.
            self.queue(directory, listing.directory, copies)
        return refused or None

    def below_refused(self, directory: str) -> bool:
        """Whether a source directory is in one whose name the destination refused.

        Its parent may be no path the source verifies, e.g. "data/..": it is
        then in no directory refused.
        """
        try:
            return self.handler.parent(directory) in self.refused_dirs
        except ValueError:
            return False

    @property
    def parents_first(self) -> bool:
        """Whether each directory is made after its parent, by one mkdir each.

        A storage declaring ``parents_required=True`` makes a directory only
        in an existing parent, as SFTP and Azure Files do. Its ``makedirs``
        then makes a missing parent by more requests: e.g. one refused, then
        the parent's, then the directory's again. The walk lists each
        directory before what is in it, so making them in that order costs
        one mkdir each. A storage not declaring it may make the parents in
        the same request: only the directories with files are made then, and
        the empty ones. So it is with ``maxdepth``, as fsspec's copy makes no
        empty directory.
        """
        return (
            self.parents_required
            and self.profile.parents_required is True
            and self.maxdepth is None
        )

    def queue(
        self,
        directory: StoragePath,
        source: str,
        copies: list[_Job],
        *,
        always: bool = False,
    ) -> None:
        """Queue copies into a destination directory, made first if needed: never the root.

        Parents first, a directory's mkdir waits for its parent's, if that is
        not done yet.
        """
        if (
            directory == self.dst.root
            or not directory
            or directory in self.made
            or not (always or self.parents_required)
        ):
            self.release(copies)
            return

        if directory not in self.after_mkdir:
            job = _Job("mkdir", {"name": source}, directory)
            parent = self.dst.parent(directory)
            if self.parents_first and parent in self.after_mkdir:
                self.after_mkdir[parent].append(job)
            else:
                self.waiting.append(job)
        self.after_mkdir.setdefault(directory, []).extend(copies)

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "mkdir":
            return self.dst.makedirs(
                job.destination,
                exist_ok=True,
                missing=self.known_missing(job.destination),
            )
        if job.kind == "whole":
            return self.transfer_whole(job.source, job.destination)
        else:
            return self.transfer_file(job.source, job.destination)

    @abstractmethod
    async def transfer_file(
        self, entry: dict[str, Any], destination: StoragePath
    ) -> None:
        """Copy or move a file, by its entry. An error raised notes its paths, as it is yielded."""

    @abstractmethod
    def transfer_whole(self, entry: Entry, destination: StoragePath) -> Awaitable[None]:
        """Copy or move a root with all in it, by the storage's one request."""

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            self.made.add(job.destination)
            self.release(self.after_mkdir.pop(job.destination))
            return []
        if job.kind == "whole":
            self.roots_whole = (self.roots_whole or 0) + 1
            return [(job.path, job.destination)]
        self.files_done += 1
        return [(job.path, job.destination)]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if job.kind == "whole":
            # Declared to go whole, the root failed so: its cause, not one
            # failure per file.
            error.add_note(f"{self.doing} {job.path!r} to {job.destination!r} failed.")
            return [error]
        # A file's error is noted where it happened, in transfer_file.
        if job.kind == "mkdir":
            # Its copies, and those of the directory's later parts, are not sent:
            # each would fail as it did. Parents first, nor are its directories'.
            error.add_note(f"Creating the directory {job.destination!r} failed.")
        return [error]


class _Copy(_CopyOrMove):
    """The jobs of one `_tree_cp`: a ``cp_entry`` per file."""

    def __init__(self, *, cp_entry: _Transfer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cp_entry = cp_entry

    async def setup(self) -> None:
        await super().setup()

        # Only within one storage does the profile tell how the source copies. A
        # directory copied whole would take what is deeper than maxdepth along.
        self.goes_whole = (
            self.same_storage
            and bool(self.profile.copies_directories)
            and self.maxdepth is None
        )
        self.roots_whole = 0 if self.goes_whole else None

    def state(self) -> CpState:
        return CpState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_copied=self.files_done,
            roots_copied=self.roots_whole,
        )

    async def transfer_file(
        self, entry: dict[str, Any], destination: StoragePath
    ) -> None:
        await self.copy_file(self.cp_entry, entry, destination)

    def transfer_whole(self, entry: Entry, destination: StoragePath) -> Awaitable[None]:
        return self.handler.cp_tree(entry, destination)
