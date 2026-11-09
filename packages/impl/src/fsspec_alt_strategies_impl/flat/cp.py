# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy that copies the keys of a prefix as it lists them.

`_flat_cp` is an alternative to fsspec's recursive ``copy`` for flat storages.
fsspec's lists every key first, then copies them. `_flat_cp` copies each key
as soon as the listing gives it. The destination may be another storage, flat
or a tree: within one, it is the same filesystem given twice.

This module depends on `_jobs`, and `copying`, `jobs`, `operation` and `paths`
of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import CpState

from ..common.jobs import _CopyJobs, _Job
from ..common.operation import Operation
from ..common.paths import _check_maxdepth
from ._jobs import _FlatJobs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from ..common.jobs import _Transfer

__all__: list[str] = []


def _flat_cp(
    source: Any,
    destination: Any,
    path1: str | Sequence[str],
    path2: str,
    *,
    cp_file: _Transfer | None = None,
    same_storage: bool | None = None,
    maxdepth: int | None = None,
) -> Operation[tuple[str, str], CpState]:
    """Copy the keys under a prefix, copying as the listing goes on.

    It places the paths as fsspec's ``copy`` with ``recursive=True`` does, as
    `_tree_cp` places them: into ``path2`` if it is an existing folder or ends
    with "/", and ``path1`` does not end with "/", otherwise onto it. A glob
    pattern copies each path it matches, relative to the matches' common
    prefix. A list copies each path it names or matches into ``path2``, under
    its name, as ``cp -r`` does. Existing files are overwritten.

    The keys are listed as `_flat_find` lists them, and each is copied by one
    ``cp_file`` as soon as the listing gives it.

    The destination is described by ``destination`` and its
    ``storage_profile``, unknown if it has none, and may be flat or a tree.
    Within one storage, ``destination`` is ``source``, and ``same_storage`` is
    True: a path copied onto itself, or under itself, is refused. Between two,
    ``cp_file`` copies from one to the other, e.g. the router's
    ``_cross_fs_cp_file``, and every byte may go through the caller. A path
    may then go to the same path, or under it.

    Unless the profile says it needs none, a folder is made at the destination
    before the first key copied into it, by one ``dst_makedirs``. A folder's
    marker, a key ending with "/", is no file to copy: its folder is made at
    the destination instead, so that a tree keeps it even when empty, unless
    the destination's folders are implicit.

    With ``maxdepth``, it copies the keys fsspec's ``copy`` copies with it:
    those at most ``maxdepth`` levels below ``path1``, or below a pattern's
    matches one level less. The keys deeper are listed, but not copied. As
    fsspec's, it then makes no folder for a marker. It places the keys as
    without ``maxdepth``, as `_tree_cp` does.

    The copies run concurrently, so the destination's keys are created in
    another order than the source's. A storage listing in creation order
    shows it.

    Unlike fsspec's ``copy``, it keeps going past a failure, and reports each.
    A copy cancelled or failed leaves what was copied so far: it deletes
    nothing. A folder failing to be made is reported, and the keys into it
    are still copied, each reporting its own failure.

    It wins over fsspec's ``copy`` on large prefixes of flat storages: the
    copies start after the first page of the listing, not after the last.
    Between two storages, it wins over fsspec's way too, a ``get`` to local
    files then a ``put``: the data needs no local disk. It loses on sources
    with a hierarchical namespace, whose folders a listing by prefix does not
    see: `_tree_cp` suits those.

    Once cancelled, it takes nothing more from the listing, and starts no new
    copy, but yields the outcomes of those sent. Closed, it waits for those
    without yielding them. Iterating it raises FileNotFoundError if ``path1``
    does not exist, or matches nothing, and ValueError if ``path2`` is inside
    it within one storage.

    Requests: one ``dst_info`` for ``path2``, unless it ends with "/" or
    ``path1`` is a pattern or a list. Patterns, or a list, add the requests
    of their `_flat_glob`. The pages of one ``_iexpand`` per folder copied,
    one after another, one ``cp_file`` per key, and one ``dst_makedirs`` per
    destination folder with keys or a marker. A destination refusing to
    overwrite costs a ``dst_info``, a ``dst_rm_file`` and a ``cp_file`` more
    per file it refused. Each request but the listing holds a permit of the
    source's ``operation_limit`` while it runs, a ``cp_file`` between two
    storages too.

    The destination's profile's ``parents_required=False``, or
    ``implicit_folders=True``, saves the ``dst_makedirs`` of each folder with
    keys, as their copies create it. ``implicit_folders=True`` saves those of
    the markers too.

    The primitives are called above by what they do: ``_iexpand`` and
    ``info`` are the source's ``_iexpand`` and ``_info``, ``dst_info``,
    ``dst_makedirs`` and ``dst_rm_file`` the destination's ``_info``,
    ``_makedirs`` and ``_rm_file``.

    Parameters
    ----------
    source : FlatFileSystem
        The filesystem copied from.
    destination : AsyncFileSystem
        The filesystem copied to: ``source`` within one storage.
    path1 : str or sequence of str
        The folder or file to copy, a glob pattern, or a list of them, as
        the source's ``_iexpand`` takes paths.
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
        The deepest level to copy: 1 copies only the keys right in ``path1``.
        No limit by default.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each key copied, or an error.
        Its state is a `CpState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    _tree_cp : The same, for storages listing one folder per request.
    _flat_mv : The same, deleting the sources as it goes.
    """
    _check_maxdepth(maxdepth)
    cp_file = source._cp_file if cp_file is None else cp_file
    if same_storage is None:
        same_storage = source is destination
    return Operation(
        lambda op: _Copy(
            fs=source,
            destination=destination,
            path1=path1,
            path2=path2,
            same_storage=same_storage,
            maxdepth=maxdepth,
            cp_file=cp_file,
            op=op,
        ).outcomes(),
        CpState(),
    )


class _CopyOrMove(_CopyJobs, _FlatJobs):
    """The jobs of a copy or a move: a folder made where needed, and one per key.

    The keys listed are the source's, ``fs``. A key's job waits for its
    destination folder, when the destination needs it. The keys of one folder
    come apart in the listing, so the folders made are remembered, and each
    is made once. A subclass says how a key goes, in `transfer_file`.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # The destination folders made, or tried: their keys need not wait.
        self.made: set[str] = set()

        # The copies waiting for their destination folder, by that folder.
        self.after_mkdir: dict[str, list[_Job]] = {}

    def add(self, entry: dict[str, Any]) -> None:
        name = entry["name"]
        if name.endswith("/"):
            # A folder's marker: a tree destination keeps the folder. With
            # maxdepth, fsspec's copy leaves out folders, and copies files only.
            if not self.implicit_folders and self.maxdepth is None:
                self.queue(self.target(name.rstrip("/")), [], always=True)
            return

        self.files_found += 1
        target = self.target(name)
        self.queue(target.rpartition("/")[0], [_Job("copy", name, target)])

    def queue(self, folder: str, copies: list[_Job], *, always: bool = False) -> None:
        """Queue copies into a destination folder, made first if needed."""
        if not folder or folder in self.made or not (always or self.parents_required):
            self.waiting.extend(copies)
            return

        if folder not in self.after_mkdir:
            self.waiting.append(_Job("mkdir", "", folder))
        self.after_mkdir.setdefault(folder, []).extend(copies)

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "mkdir":
            return self.destination._makedirs(job.destination, exist_ok=True)
        return self.transfer_file(job.source, job.destination)

    @abstractmethod
    async def transfer_file(self, source: str, destination: str) -> None:
        """Copy or move a key. An error raised notes its paths, as it is yielded."""

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            self.made.add(job.destination)
            self.waiting.extend(self.after_mkdir.pop(job.destination))
            return []

        self.files_done += 1
        return [(job.source, job.destination)]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if job.kind == "mkdir":
            # The copies still go: each then reports its own failure, if any.
            self.made.add(job.destination)
            self.waiting.extend(self.after_mkdir.pop(job.destination))
            error.add_note(f"Creating the folder {job.destination!r} failed.")

        # A key's error is noted where it happened, in transfer_file.
        return [error]


class _Copy(_CopyOrMove):
    """The jobs of one `_flat_cp`: a ``cp_file`` per key."""

    def __init__(self, *, cp_file: _Transfer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cp_file = cp_file

    async def transfer_file(self, source: str, destination: str) -> None:
        await self.copy_file(self.cp_file, source, destination)
