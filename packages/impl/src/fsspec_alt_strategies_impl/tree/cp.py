# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy that walks the tree and copies while it walks.

`_tree_cp` is an alternative to fsspec's recursive ``copy``. fsspec's lists the
whole tree first, then copies each path. `_tree_cp` copies each folder's files
as soon as the folder is listed. The destination may be another storage:
within one, it is the same storage's operations given twice.

This module depends on `_jobs`, and `jobs`, `operation`, `paths` and
`placement` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import CpState

from ..common.filesystem import _profile_of
from ..common.jobs import _Job
from ..common.operation import Operation
from ..common.paths import _check_maxdepth
from ..common.placement import _Placement
from ._jobs import _TreeJobs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from fsspec_alt_strategies.profile import StorageProfile

    from ..common.jobs import _Roots
    from ..common.types import _Info, _Request
    from .walk import _Listing

__all__: list[str] = []


def _tree_cp(
    source: Any,
    destination: Any,
    path1: str | Sequence[str],
    path2: str,
    *,
    cp_file: _Request | None = None,
    same_storage: bool | None = None,
    maxdepth: int | None = None,
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
    ``path2``. The pattern is matched as `_tree_glob` matches it, then the
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

    The tree is walked as `_tree_walk` walks it. Each folder's files are copied
    as soon as the folder is listed, each by a ``cp_file``. A folder is
    created at the destination before files are copied into it, and for each
    empty folder.

    With ``maxdepth``, it copies the files fsspec's ``copy`` copies with it:
    those at most ``maxdepth`` levels below ``path1``, or below a pattern's
    matches one level less. Folders deeper are not listed. As fsspec's, it
    then makes no empty folder. Unlike fsspec's, it places the files as
    without ``maxdepth``: fsspec places them by the common prefix of the files
    found, which moves them when all are in one subfolder.

    The destination is described by ``destination`` and its
    ``storage_profile``, unknown if it has none. Within
    one storage, ``destination`` is ``source``, and ``same_storage`` is True:
    a path copied onto itself, or under itself, is refused. Between two,
    ``cp_file`` copies from one to the other, e.g. the router's
    ``_cross_fs_cp_file``, and every byte may go through the caller. A path
    may then go to the same path, or under it.
    The copies run concurrently, so the destination's paths are created in
    another order than the source's. A storage listing in creation order
    shows it.

    Unlike fsspec's ``copy``, it keeps going past a failure, and reports each.
    A copy cancelled or failed leaves what was copied so far: it deletes
    nothing.

    It wins over fsspec's ``copy`` on storages that list one folder per
    request. The copies then start after the first listing, not after the
    whole tree is listed. Between two storages, it wins over fsspec's way too,
    a ``get`` to local files then a ``put``: the data needs no local disk.

    Once cancelled, it starts no new listing or copy, and yields the outcomes
    of the copies already sent. Closed, it waits for those copies without
    yielding them. Iterating it raises FileNotFoundError if ``path1`` does not
    exist, or matches nothing, and ValueError if ``path2`` is inside it within
    one storage.

    Requests: one ``dst_info`` for ``path2``, unless it ends with "/" or
    ``path1`` is a pattern or a list. Patterns, or a list, add the requests of
    their `_tree_glob`. One ``ils`` per folder, one ``cp_file`` per file, and
    one ``dst_makedirs`` per destination folder with files or empty. A
    destination refusing to overwrite costs a ``dst_info``, a ``dst_rm_file``
    and a ``cp_file`` more per file it refused. The ``dst_info`` checks it is
    a file, as ``dst_rm_file`` is given only paths known to be files. Each
    request holds a permit of the source's ``operation_limit`` while it runs,
    a ``cp_file`` between two storages too.

    The destination's profile's ``parents_required=False``, or
    ``implicit_folders=True``,
    saves the ``dst_makedirs`` of each folder with files, as their copies
    create it.

    The primitives are called above by what they do: ``ils`` and ``info`` are
    the source's ``_ils`` and ``_info``, ``dst_info``, ``dst_makedirs`` and
    ``dst_rm_file`` the destination's ``_info``, ``_makedirs`` and
    ``_rm_file``.

    Parameters
    ----------
    source : TreeFileSystem
        The filesystem copied from.
    destination : AsyncFileSystem
        The filesystem copied to: ``source`` within one storage.
    path1 : str or sequence of str
        The folder or file to copy, a glob pattern, or a list of them, as
        the source's ``_ils`` takes paths.
    path2 : str
        Where to copy it, as above.
    cp_file : callable, optional
        Copies a file to the destination, called as
        ``cp_file(source, destination)``: the source's ``_cp_file`` by
        default, which copies within one storage.
    same_storage : bool, optional
        Whether the destination is the source's storage: whether
        ``destination`` is ``source`` by default.
    maxdepth : int, optional
        The deepest level to copy: 1 copies only the files in ``path1``. No
        limit by default.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each file copied, or an error.
        Its state is a `CpState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    _tree_walk : The walk under it.
    _tree_mv : The same, deleting the source as it goes.
    """
    _check_maxdepth(maxdepth)
    options = _options(source, destination, same_storage, maxdepth)
    cp_file = source._cp_file if cp_file is None else cp_file
    return Operation(
        lambda op: _Copy(path1, path2, cp_file=cp_file, op=op, **options).outcomes(),
        CpState(),
    )


def _options(
    source: Any,
    destination: Any,
    same_storage: bool | None,
    maxdepth: int | None,
) -> dict[str, Any]:
    """The options of a copy or a move, taken from its filesystems.

    The profile is the destination's, as what a copy saves depends on where
    it writes. Within one storage, it is the source's too.
    """
    return {
        "source": source,
        "destination": destination,
        "same_storage": source is destination if same_storage is None else same_storage,
        "profile": _profile_of(destination),
        "maxdepth": maxdepth,
    }


class _CopyOrMove(_TreeJobs):
    """The jobs of a copy or a move: a folder made where needed, and one per file.

    A file's job waits for its destination folder, when the storage needs it.
    A subclass says how a file goes, in `transfer_file`.
    """

    verb = "copy"
    doing = "Copying"

    def __init__(
        self,
        path1: str | Sequence[str],
        path2: str,
        *,
        source: Any,
        destination: Any,
        same_storage: bool,
        profile: StorageProfile,
        maxdepth: int | None,
        op: Operation[Any, Any],
    ) -> None:
        super().__init__(source, op, maxdepth)

        self.path1 = path1
        self.path2 = path2

        # The walk is the source's. The profile is the destination's.
        self.destination = destination
        destination_info: _Info = destination._info
        self.makedirs: _Request = destination._makedirs

        # Unknown, folders are made first. Implicit folders need none.
        self.parents_required = (
            profile.parents_required is not False and not profile.implicit_folders
        )
        self.implicit_folders = bool(profile.implicit_folders)

        self.destination_info = destination_info
        self.placement = _Placement(
            path1,
            path2,
            destination_info=destination_info,
            same_storage=same_storage,
            expand=self.glob_roots,
            found=self.found,
            verb=self.verb,
            doing=self.doing,
        )

        self.files_found = 0
        self.files_done = 0

        # The copies waiting for their destination folder, by that folder.
        self.after_mkdir: dict[str, list[_Job]] = {}

        # The destination folders made, or tried: a folder listed in several
        # parts is made once, and its later copies need not wait.
        self.made: set[str] = set()

    def state(self) -> Any:
        return CpState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_copied=self.files_done,
        )

    async def prepare(self) -> _Roots | None:
        return await self.placement.prepare()

    def target(self, path: str) -> str:
        """Where a path goes."""
        return self.placement.target(path)

    def file_root(self, path: str) -> None:
        self.files_found += 1
        target = self.target(path)
        self.queue(target.rpartition("/")[0], "", [_Job("copy", path, target)])

    def add(self, listing: _Listing) -> list[Any] | None:
        copies = []
        for entry in listing.entries:
            if entry["type"] != "directory":
                name = entry["name"].rstrip("/")
                self.files_found += 1
                copies.append(_Job("copy", name, self.target(name)))

        if listing.first and listing.last and not listing.entries:
            # An empty folder is made, whatever the storage. With maxdepth,
            # fsspec's copy leaves out folders, and copies files only.
            if self.maxdepth is not None:
                return None
            self.queue(self.target(listing.folder), listing.folder, [], always=True)
        elif copies:
            # A folder holding only folders needs no mkdir: theirs create it.
            self.queue(self.target(listing.folder), listing.folder, copies)
        return None

    def queue(
        self, folder: str, source: str, copies: list[_Job], *, always: bool = False
    ) -> None:
        """Queue copies into a destination folder, made first if needed."""
        if not folder or folder in self.made or not (always or self.parents_required):
            self.waiting.extend(copies)
            return

        if folder not in self.after_mkdir:
            self.waiting.append(_Job("mkdir", source, folder))
        self.after_mkdir.setdefault(folder, []).extend(copies)

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "mkdir":
            return self.makedirs(job.destination, exist_ok=True)
        return self.transfer_file(job.source, job.destination)

    @abstractmethod
    async def transfer_file(self, source: str, destination: str) -> None:
        """Copy or move a file. An error raised notes its paths, as it is yielded."""

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            self.made.add(job.destination)
            self.waiting.extend(self.after_mkdir.pop(job.destination))
            return []
        self.files_done += 1
        return [(job.source, job.destination)]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        # A file's error is noted where it happened, in transfer_file.
        if job.kind == "mkdir":
            # Its copies, and those of the folder's later parts, are not sent:
            # each would fail as it did.
            error.add_note(f"Creating the folder {job.destination!r} failed.")
        return [error]


class _Copy(_CopyOrMove):
    """The jobs of one `_tree_cp`: a ``cp_file`` per file."""

    def __init__(
        self,
        path1: str | Sequence[str],
        path2: str,
        *,
        cp_file: _Request,
        **kwargs: Any,
    ) -> None:
        super().__init__(path1, path2, **kwargs)
        self.cp_file = cp_file
        self.rm_file: _Request = self.destination._rm_file

    async def transfer_file(self, source: str, destination: str) -> None:
        try:
            try:
                await self.cp_file(source, destination)
            except FileExistsError:
                # The destination refuses to overwrite, which fsspec's copy
                # does. A folder there is no file to replace: rm_file takes
                # files only. The job holds its permit already.
                if (await self.destination_info(destination))["type"] == "directory":
                    raise

                await self.rm_file(destination)
                await self.cp_file(source, destination)
        except Exception as e:
            # Yielded among paths, the error must tell its paths itself.
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise
