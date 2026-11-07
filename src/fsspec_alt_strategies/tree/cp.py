# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy that walks the tree and copies while it walks.

`tree_cp` is an alternative to fsspec's recursive ``copy``. fsspec's lists the
whole tree first, then copies each path. `tree_cp` copies each folder's files
as soon as the folder is listed.

This module depends on `control`, `profile`, `walk` and `_jobs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec.utils import common_prefix

from ..control import Operation, State
from ..profile import StorageProfile
from ._jobs import _expand, _Job, _Jobs, _Roots
from .walk import _trim

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Sequence

    from ._jobs import _Request
    from .walk import _Info, _Listing, _Ls

__all__ = ["CpState", "tree_cp"]


@dataclass(frozen=True, kw_only=True)
class CpState(State):
    """The progress of a `tree_cp`.

    Attributes
    ----------
    files_found : int
        The files found so far. It grows as the walk goes on.
    files_copied : int
        The files copied so far.
    """

    files_found: int = 0
    files_copied: int = 0


def tree_cp(
    path1: str | Sequence[str],
    path2: str,
    *,
    ls: _Ls,
    info: _Info,
    cp_file: _Request,
    makedirs: _Request,
    rm_file: _Request,
    profile: StorageProfile | None = None,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[tuple[str, str], CpState]:
    """Copy a folder and everything in it, copying while the tree is walked.

    It places the paths as fsspec's ``copy`` with ``recursive=True`` does. If
    ``path2`` is an existing folder, or ends with "/", and ``path1`` does not
    end with "/", ``path1`` goes into it: ``path1/x`` becomes
    ``path2/<name of path1>/x``. Otherwise ``path1`` becomes ``path2``:
    ``path1/x`` becomes ``path2/x``. A file is copied the same way. Existing
    files are overwritten, as fsspec's ``copy`` does.

    A glob pattern copies each path it matches, and all in it, placed as
    fsspec's ``copy`` places them. Their paths below the matches' common
    prefix are kept under ``path2``, but a single file matched goes into
    ``path2``. The pattern is matched as `tree_glob` matches it, then the
    folders matched are walked at once.

    A list of paths and patterns copies each path it names or matches into
    ``path2``, under its name, with all in it. That is how ``cp -r`` places
    them. fsspec's ``copy`` places the files named as this does. It also
    flattens the folders' trees, though: each file in them goes straight into
    ``path2``. Two paths of the same name would go to the same place. The
    first, in path order, is copied, and each other yields a FileExistsError
    instead, as GNU's ``cp`` reports it. fsspec's ``copy`` overwrites one
    with the other. A path inside another one in the list is copied with that
    one only.

    The tree is walked as `tree_walk` walks it. Each folder's files are copied
    as soon as the folder is listed, each by a ``cp_file``. A folder is
    created at the destination before files are copied into it, and for each
    empty folder.

    Unlike fsspec's ``copy``, it keeps going past a failure, and reports each.
    A copy cancelled or failed leaves what was copied so far: it deletes
    nothing.

    It wins over fsspec's ``copy`` on storages that list one folder per
    request. The copies then start after the first listing, not after the
    whole tree is listed.

    Once cancelled, it starts no new listing or copy, and yields the outcomes
    of the copies already sent. Closed, it waits for those copies without
    yielding them. Iterating it raises FileNotFoundError if ``path1`` does not
    exist, or matches nothing, and ValueError if ``path2`` is inside it.

    Requests: one ``info`` for ``path2``, unless it ends with "/" or ``path1``
    is a pattern or a list. Patterns, or a list, add the requests of their
    `tree_glob`. One ``ls`` per folder, one ``cp_file`` per file, and one
    ``makedirs`` per destination folder with files or empty. A storage
    refusing to overwrite costs an ``rm_file`` and a ``cp_file`` more per file
    it refused. Each request holds a permit of ``concurrency_limit`` while it
    runs.

    The profile's ``parents_required=False`` saves the ``makedirs`` of each
    folder with files, as their copies create it.

    Parameters
    ----------
    path1 : str or sequence of str
        The folder or file to copy, a glob pattern, or a list of them, as
        ``ls`` takes paths.
    path2 : str
        Where to copy it, as above.
    ls : callable
        Lists a folder, called as ``ls(path, detail=True)``: e.g. the
        filesystem's ``_ls``.
    info : callable
        Describes a path, called as ``info(path)``: e.g. the filesystem's
        ``_info``.
    cp_file : callable
        Copies a file, called as ``cp_file(source, destination)``: e.g. the
        filesystem's ``_cp_file``.
    makedirs : callable
        Creates a folder and its parents, called as
        ``makedirs(path, exist_ok=True)``: e.g. the filesystem's ``_makedirs``.
    rm_file : callable
        Deletes a file, called as ``rm_file(path)``: e.g. the filesystem's
        ``_rm_file``. Only a storage refusing to overwrite needs it.
    profile : StorageProfile, optional
        What the storage does. Unknown if None.
    concurrency_limit : asyncio.Semaphore
        Held by each request while it runs. Share one to limit several
        operations, or files, together.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each file copied, or an error.
        Its state is a `CpState`.

    See Also
    --------
    tree_walk : The walk under it.
    tree_mv : The same, deleting the source as it goes.

    Examples
    --------
    >>> cp = tree_cp(
    ...     "data/2024",
    ...     "backup/",
    ...     ls=fs._ls,
    ...     info=fs._info,
    ...     cp_file=fs._cp_file,
    ...     makedirs=fs._makedirs,
    ...     rm_file=fs._rm_file,
    ...     concurrency_limit=asyncio.Semaphore(32),
    ... )
    >>> async with cp as op:
    ...     async for copied in op:
    ...         if isinstance(copied, Exception):
    ...             print("failed:", copied)
    >>> op.state.files_copied
    1234
    """
    return Operation(
        lambda op: _Copy(
            path1,
            path2,
            ls=ls,
            info=info,
            cp_file=cp_file,
            makedirs=makedirs,
            rm_file=rm_file,
            profile=profile or StorageProfile(),
            concurrency_limit=concurrency_limit,
            op=op,
        ).outcomes(),
        CpState(),
    )


class _Copy(_Jobs):
    """The jobs of one `tree_cp`: a folder made where needed, and a copy per file.

    A file's copy waits for its destination folder, when the storage needs it.
    """

    verb = "copy"
    doing = "Copying"

    def __init__(
        self,
        path1: str | Sequence[str],
        path2: str,
        *,
        ls: _Ls,
        info: _Info,
        cp_file: _Request,
        makedirs: _Request,
        rm_file: _Request,
        profile: StorageProfile,
        concurrency_limit: asyncio.Semaphore,
        op: Operation[Any, Any],
    ) -> None:
        super().__init__(ls, info, concurrency_limit, op)
        self.path1 = path1
        self.path2 = path2
        self.cp_file = cp_file
        self.makedirs = makedirs
        self.rm_file = rm_file
        self.parents_required = profile.parents_required is not False
        self.prefix = ""
        self.base = ""
        # For a list: where each root goes, by the root.
        self.places: dict[str, str] = {}
        self.files_found = 0
        self.files_done = 0
        # The copies waiting for their destination folder, by that folder.
        self.after_mkdir: dict[str, list[_Job]] = {}

    def state(self) -> Any:
        return CpState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_copied=self.files_done,
        )

    async def prepare(self) -> _Roots | None:
        """Set where the paths go, as fsspec's ``copy`` places them.

        A path under ``self.prefix`` goes to the same place under
        ``self.base``. For a list, a path under a root of ``self.places`` goes
        there instead.
        """
        destination = _trim(self.path2)
        if not isinstance(self.path1, str):
            roots = await self._place(destination)
            sources = [*roots.folders, *roots.files]
        elif has_magic(self.path1):
            roots = await _expand(
                self.ls, self.info, self.path1, self.concurrency_limit
            )
            sources = [*roots.folders, *roots.files]
            if not sources:
                return roots  # Only errors, to yield.
            self.prefix = common_prefix(sources)
            if sources == list(roots.files) and len(sources) == 1:
                # A single file goes into the destination.
                self.prefix = self.prefix.rpartition("/")[0]
            self.base = destination
        else:
            root = _trim(self.path1)
            if root == destination:
                return None
            roots = _Roots([root])
            sources = [root]
            self.prefix = root
            self.base = await self._base(root, destination)
        for source in sources:
            target = self.target(source)
            if target == source and len(sources) == 1:
                return None
            if not source or target == source or target.startswith(source + "/"):
                raise ValueError(
                    f"Cannot {self.verb} {source!r} into itself: {target!r}"
                )
        return roots

    async def _place(self, destination: str) -> _Roots:
        """Place each root of a list into ``destination``, under its name.

        Two roots of the same name would go to the same place. As GNU's cp
        does, the first, in path order, goes, and each other is an error. That
        error is the root's own: what is in it is not copied either.
        """
        roots = await _expand(self.ls, self.info, self.path1, self.concurrency_limit)
        folders = set(roots.folders)
        placed: dict[str, str] = {}
        errors = list(roots.errors)
        for source in sorted([*roots.folders, *roots.files]):
            name = source.rpartition("/")[2]
            target = f"{destination}/{name}" if destination else name
            if (other := placed.get(target)) is not None:
                error = FileExistsError(target)
                error.add_note(
                    f"{self.doing} {source!r} to {target!r} skipped: {other!r} goes"
                    " there too."
                )
                errors.append(error)
                continue
            placed[target] = source
            self.places[source] = target
        return _Roots(
            [s for s in self.places if s in folders],
            [s for s in self.places if s not in folders],
            errors,
        )

    async def _base(self, root: str, destination: str) -> str:
        """Where ``root``, a single path, goes: into ``destination``, or onto it."""
        if self.path2.endswith("/"):
            into = True
        else:
            try:
                into = (await self.info(destination))["type"] == "directory"
            except FileNotFoundError:
                into = False
        if not into or self.path1.endswith("/"):
            return destination
        name = root.rsplit("/", 1)[-1]
        return f"{destination}/{name}" if destination else name

    def target(self, path: str) -> str:
        """Where a path goes."""
        if self.places:
            # The root it is under, found from the path up.
            root = path
            while root not in self.places and "/" in root:
                root = root.rpartition("/")[0]
            return self.places[root] + path[len(root) :]
        if self.prefix:
            return self.base + path[len(self.prefix) :]
        return f"{self.base}/{path}" if self.base else path

    def file_root(self, path: str) -> None:
        self.files_found += 1
        target = self.target(path)
        self.queue(target.rpartition("/")[0], "", [_Job("copy", path, target)])

    def add(self, listing: _Listing) -> None:
        copies = []
        for entry in listing.entries:
            if entry["type"] != "directory":
                name = entry["name"].rstrip("/")
                self.files_found += 1
                copies.append(_Job("copy", name, self.target(name)))
        if not listing.entries:
            # An empty folder is made, whatever the storage.
            self.queue(self.target(listing.folder), listing.folder, [], always=True)
        elif copies:
            # A folder holding only folders needs no mkdir: theirs create it.
            self.queue(self.target(listing.folder), listing.folder, copies)

    def queue(
        self, folder: str, source: str, copies: list[_Job], *, always: bool = False
    ) -> None:
        """Queue copies into a destination folder, made first if needed."""
        if not folder or not (always or self.parents_required):
            self.waiting.extend(copies)
            return
        if folder not in self.after_mkdir:
            self.waiting.append(_Job("mkdir", source, folder))
        self.after_mkdir.setdefault(folder, []).extend(copies)

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "mkdir":
            return self.makedirs(job.destination, exist_ok=True)
        return self.copy_file(job.source, job.destination)

    async def copy_file(self, source: str, destination: str) -> None:
        try:
            try:
                await self.cp_file(source, destination)
            except FileExistsError:
                # The storage refuses to overwrite, which fsspec's copy does.
                await self.rm_file(destination)
                await self.cp_file(source, destination)
        except Exception as e:
            # Yielded among paths, the error must tell its paths itself.
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            self.waiting.extend(self.after_mkdir.pop(job.destination))
            return []
        self.files_done += 1
        return [(job.source, job.destination)]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        # A file's error is noted where it happened, in copy_file.
        if job.kind == "mkdir":
            error.add_note(f"Creating the folder {job.destination!r} failed.")
        return [error]
