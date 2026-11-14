# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy that copies the keys of a prefix as it lists them.

`_flat_cp` is an alternative to fsspec's recursive ``copy`` for flat storages.
fsspec's lists every key first, then copies them. `_flat_cp` copies each key
as soon as the listing gives it. The destination may be another storage, flat
or a tree: within one, it is the same filesystem given twice.

This module depends on `_jobs`, and `jobs` and `paths` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ..common.jobs import _CopyJobs, _Job, _transfer_of
from ..common.paths import _check_maxdepth
from ._jobs import _FlatJobs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from fsspec_alt_strategies.states import CpState

    from ..common.handler import StorageHandler, StoragePath
    from ..common.jobs import _Transfer
    from ..common.operation import Operation
    from .handler import FlatFileOperationHandler

__all__: list[str] = []


def _flat_cp(
    src: FlatFileOperationHandler,
    dst: StorageHandler,
    path1: StoragePath | Sequence[StoragePath],
    path2: StoragePath,
    *,
    cp_entry: _Transfer | None = None,
    same_storage: bool | None = None,
    maxdepth: int | None = None,
    path1_entry: dict[str, Any] | None = None,
    ils_kwargs: dict[str, Any] | None = None,
) -> Operation[tuple[str, str], CpState]:
    """Copy the keys under a prefix, copying as the listing goes on.

    It places the paths as fsspec's ``copy`` with ``recursive=True`` does, as
    `_tree_cp` places them: into ``path2`` if it is an existing directory or
    ends with "/", and ``path1`` does not end with "/", otherwise onto it. A glob
    pattern copies each path it matches, relative to the matches' common
    prefix. A list copies each path it names or matches into ``path2``, under
    its name, as ``cp -r`` does. Existing files are overwritten.

    The keys are listed as `_flat_find` lists them, and each is copied by one
    ``cp_entry`` as soon as the listing gives it.

    The destination is described by ``dst`` and its ``profile``,
    unknown if it has none, and may be flat or a tree. Within one storage,
    ``dst`` is ``src``, and ``same_storage`` is True: a path copied onto
    itself, or under itself, is refused. Between two, ``cp_entry`` copies from
    one to the other, by default `transfer._across`, its data through the
    caller. A path may then go to the same path, or under it.

    Unless the profile says it needs none, a directory is made at the
    destination before the first key copied into it, by one ``dst_makedirs``.
    A directory's marker, a key ending with "/", is no file to copy: its
    directory is made at the destination instead, so that a tree keeps it even
    when empty, unless the destination's directories are implicit.

    With ``maxdepth``, it copies the keys fsspec's ``copy`` copies with it:
    those at most ``maxdepth`` levels below ``path1``, or below a pattern's
    matches one level less. The keys deeper are listed, but not copied. As
    fsspec's, it then makes no directory for a marker. It places the keys as
    without ``maxdepth``, as `_tree_cp` does.

    The copies run concurrently, so the destination's keys are created in
    another order than the source's. A storage listing in creation order
    shows it.

    Unlike fsspec's ``copy``, it keeps going past a failure, and reports each.
    A copy cancelled or failed leaves what was copied so far: it deletes
    nothing. A directory failing to be made is reported, and the keys into it
    are still copied, each reporting its own failure.

    It wins over fsspec's ``copy`` on large prefixes of flat storages: the
    copies start after the first page of the listing, not after the last.
    Between two storages, it wins over fsspec's way too, a ``get`` to local
    files then a ``put``: the data needs no local disk. It loses on sources
    with a hierarchical namespace, whose directories a listing by prefix does
    not see: `_tree_cp` suits those.

    Once cancelled, it takes nothing more from the listing, and starts no new
    copy, but yields the outcomes of those sent. Closed, it waits for those
    without yielding them. Iterating it raises FileNotFoundError if ``path1``
    does not exist, or matches nothing, and ValueError if ``path2`` is inside
    it within one storage.

    Requests: one ``dst_info`` for ``path2``, unless it ends with "/" or
    ``path1`` is a pattern or a list. Patterns, or a list, add the requests
    of their `_flat_glob`. The pages of one ``ils_recursive`` per root, one
    after another, one ``cp_entry`` per key, and one ``dst_makedirs`` per
    destination directory with keys or a marker. A destination refusing to
    overwrite costs a ``dst_info``, a ``dst_rm_file`` and a ``cp_entry`` more
    per file it refused. Each request but the listing holds a permit of the
    source's ``operation_limit`` while it runs, a ``cp_entry`` between two
    storages too.

    The destination's profile's ``parents_required=False`` between two
    storages, ``copy_parents_required=False`` within one, or
    ``implicit_directories=True``, saves the ``dst_makedirs`` of each directory
    with keys, as their copies create it. ``implicit_directories=True`` saves
    those of the markers too. A directory known to exist needs none: the
    ``dst_info`` of ``path2`` shows it or its parent, and within one storage,
    ``path1``'s parent exists.

    ``path1_entry``, the entry of ``path1`` as the caller found it, saves the
    listing of a single file: it is copied as the entry says.

    The primitives are named above as the handlers' own: ``ils`` is the
    source's ``ils_recursive``, and ``dst_info``, ``dst_makedirs`` and
    ``dst_rm_file`` the destination's ``info``, ``makedirs`` and ``rm_file``.

    ``src`` is the handler copied from, and ``dst`` the one copied
    to, ``src`` again within one storage. ``path1`` is a directory, a file,
    a glob pattern, or a list of them, as the source's ``ils`` takes paths.
    ``cp_entry(entry, destination)`` copies a file: by default, the source's
    ``cp_file`` within one storage, and `transfer._across` between two.
    ``same_storage`` is
    whether ``dst`` is ``src``, unless given. ``maxdepth`` of 1 copies
    only the keys right in ``path1``. ``ils_kwargs`` go to each of the
    source's ``ils_recursive``, a pattern's glob's too.

    Within one storage, unless ``cp_entry`` is given, the files go to the
    source's ``cp_files`` in batches of its ``cp_batch_size``: by default each
    by ``cp_file``. A file joins a batch once its destination's directory is
    ready, whatever the directory, and a batch goes once full, or once the walk
    has ended with no directory left to make.

    The operation yields ``(source, destination)`` for each key copied, or an
    error, and its state is a `CpState`. The call raises ValueError if
    ``maxdepth`` is less than 1. `_flat_mv` does the same, deleting the sources as it goes.
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
        path1_entry=path1_entry,
        ils_kwargs=ils_kwargs,
    )


class _CopyOrMove(_CopyJobs, _FlatJobs):
    """The jobs of a copy or a move: a directory made where needed, and one per key.

    The keys listed are the source's, ``handler``. A key's job waits for its
    destination directory, when the destination needs it. The keys of one
    directory come apart in the listing, so the directories made are
    remembered, and each is made once. A subclass says how a key goes, in `transfer_file`.
    """

    def add(self, entry: dict[str, Any]) -> None:
        name = entry["name"]
        if name.endswith("/"):
            # A directory's marker: a tree destination keeps the directory. With
            # maxdepth, fsspec's copy leaves out directories, and copies files only.
            if not self.implicit_directories and self.maxdepth is None:
                self.queue(self.target(name.rstrip("/")), [], always=True)
            return

        self.files_found += 1
        target = self.target(name)
        # The entry goes with its copy, so that cp_entry need not ask it.
        job = _Job("copy", entry, target)
        self.queue(self.dst.parent(target), [job])

    def queue(
        self, directory: StoragePath, copies: list[_Job], *, always: bool = False
    ) -> None:
        """Queue copies into a destination directory, made first if needed: never the root."""
        if not self.needs_mkdir(directory, always=always):
            self.release(copies)
            return

        if directory not in self.after_mkdir:
            self.waiting.append(_Job("mkdir", {"name": ""}, directory))
        self.after_mkdir.setdefault(directory, []).extend(copies)

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "mkdir":
            return self.make_directory(job.destination)
        else:
            return self.transfer_file(job.source, job.destination)

    @abstractmethod
    async def transfer_file(
        self, entry: dict[str, Any], destination: StoragePath
    ) -> None:
        """Copy or move a key, by its entry. An error raised notes its paths, as it is yielded."""

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            self.made.add(job.destination)
            self.release(self.after_mkdir.pop(job.destination))
            return []

        self.files_done += 1
        return [(job.path, job.destination)]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if job.kind == "mkdir":
            # The copies still go: each then reports its own failure, if any.
            self.made.add(job.destination)
            self.release(self.after_mkdir.pop(job.destination))
            error.add_note(f"Creating the directory {job.destination!r} failed.")

        # A key's error is noted where it happened, in transfer_file.
        return [error]


class _Copy(_CopyOrMove):
    """The jobs of one `_flat_cp`: a ``cp_entry`` per key."""

    def __init__(self, *, cp_entry: _Transfer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cp_entry = cp_entry

    async def transfer_file(
        self, entry: dict[str, Any], destination: StoragePath
    ) -> None:
        await self.copy_file(self.cp_entry, entry, destination)
