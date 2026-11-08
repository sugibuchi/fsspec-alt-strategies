# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive copy from one storage to another, copying the keys as it lists them.

`flat_transfer` is `flat_cp` between two storages. It lists the source's keys
under a prefix, and copies each to the destination as soon as the listing
gives it, by a ``copy_file`` given by the caller. `common.routing` has one
streaming the data through the two storages' file objects.

The destination may be flat or a tree: its profile says whether a folder must
be made before the files copied into it.

This module depends on `cp`, and `jobs`, `operation` and `placement` of
`common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies_common.states import CpState

from ..common.jobs import _Job
from ..common.operation import Operation
from .cp import _Copy

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Sequence

    from fsspec_alt_strategies_common.profile import StorageProfile

    from ..common.types import _Ils, _Info, _Request

__all__ = ["CpState", "flat_transfer"]


def flat_transfer(
    path1: str | Sequence[str],
    path2: str,
    *,
    ils: _Ils,
    info: _Info,
    copy_file: _Request,
    dst_info: _Info,
    dst_makedirs: _Request,
    dst_rm_file: _Request,
    profile: StorageProfile,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[tuple[str, str], CpState]:
    """Copy the keys under a prefix to another storage, as the listing goes on.

    It places the paths as `flat_cp` does, as fsspec's ``copy`` with
    ``recursive=True`` places them: into ``path2`` if it is an existing
    folder of the destination, or ends with "/", and onto it otherwise.
    Glob patterns and lists of paths are placed as `flat_cp` places them,
    and two paths of a list with the same name are reported as it does.
    Existing files are overwritten.

    The two storages are apart, so a path may be copied to the same path, or
    under itself: neither is refused, unlike in `flat_cp`.

    The keys are listed as `flat_cp` lists them, with ``ils`` and ``info``.
    Each is copied by one ``copy_file`` as soon as the listing gives it.

    The destination may be flat or a tree. Unless its profile says it needs
    none, a folder is made at the destination before the first key copied
    into it, by one ``dst_makedirs``. A folder's marker, a key ending with
    "/", is no file to copy: its folder is made at the destination instead,
    so that a tree keeps it even when empty.

    It assumes that ``copy_file`` copies a whole file, and that the
    destination behaves as ``profile`` says. The copies run concurrently, so
    the destination's keys are created in another order than the source's.

    It wins over fsspec's way, a ``get`` to local files then a ``put``, or a
    ``copy`` listing every key first. The copies start after the first page
    of the listing, and the data needs no local disk. Within one storage
    copying on its server, it loses to `flat_cp`: here every byte goes
    through the caller. On a source with a hierarchical namespace, whose
    folders a listing by prefix does not see, `tree_transfer` suits better.

    Unlike fsspec's ``copy``, it keeps going past a failure, and reports each.
    A copy cancelled or failed leaves what was copied so far: it deletes
    nothing. A folder failing to be made is reported, and the keys into it
    are still copied, each reporting its own failure.

    Once cancelled, it takes nothing more from the listing, and starts no new
    copy, but yields the outcomes of those sent. Closed, it waits for those
    without yielding them. Iterating it raises FileNotFoundError if ``path1``
    does not exist, or matches nothing.

    Requests: one ``dst_info`` for ``path2``, unless it ends with "/" or
    ``path1`` is a pattern or a list. Patterns, or a list, add the requests
    of their `flat_glob` on the source. The pages of one ``ils`` per folder
    copied, one after another, one ``copy_file`` per key, and one
    ``dst_makedirs`` per destination folder with keys or a marker. A
    destination refusing to overwrite costs a ``dst_info``, a
    ``dst_rm_file`` and a ``copy_file`` more per file it refused. Each
    request but the listing holds a permit of ``concurrency_limit`` while it
    runs, a ``copy_file`` too, though it uses both storages.

    The profile's ``parents_required=False``, or ``implicit_folders=True``,
    saves the ``dst_makedirs`` of each folder with keys, as their copies
    create it.

    Parameters
    ----------
    path1 : str or sequence of str
        The source's folder or file to copy, a glob pattern, or a list of
        them, as ``ils`` takes paths.
    path2 : str
        Where to copy it in the destination, as ``dst_info`` takes paths.
    ils : callable
        Lists every source key starting with a prefix, called as
        ``ils(prefix)``: an async iterator of entries as fsspec's ``_ls``
        gives them, in any order, each key once.
    info : callable
        Describes a source path, called as ``info(path)``.
    copy_file : callable
        Copies a file from the source to the destination, called as
        ``copy_file(source, destination)``: e.g. one made by
        ``common.routing.streaming_copy`` from both ``open_async``.
    dst_info : callable
        Describes a destination path, called as ``dst_info(path)``.
    dst_makedirs : callable
        Creates a destination folder and its parents, called as
        ``dst_makedirs(path, exist_ok=True)``. On a flat destination, it may
        do nothing.
    dst_rm_file : callable
        Deletes a destination file, called as ``dst_rm_file(path)``. Only a
        storage refusing to overwrite needs it.
    profile : StorageProfile
        What the destination does: ``StorageProfile()`` if it is unknown.
    concurrency_limit : asyncio.Semaphore
        Held by each request but the listing while it runs, on either
        storage.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each key copied, or an error.
        Its state is a `CpState`.

    See Also
    --------
    flat_cp : The same, within one storage.
    tree_transfer : The same, from a storage listing one folder per request.

    Examples
    --------
    >>> transfer = flat_transfer(
    ...     "data/2024",
    ...     "backup/",
    ...     ils=src._ils,
    ...     info=src._info,
    ...     copy_file=streaming_copy(src.open_async, dst.open_async),
    ...     dst_info=dst._info,
    ...     dst_makedirs=dst._makedirs,
    ...     dst_rm_file=dst._rm_file,
    ...     profile=StorageProfile(),
    ...     concurrency_limit=asyncio.Semaphore(32),
    ... )
    >>> async with transfer as op:
    ...     async for copied in op:
    ...         if isinstance(copied, Exception):
    ...             print("failed:", copied)
    >>> op.state.files_copied
    1234
    """
    return Operation(
        lambda op: _CrossCopy(
            path1,
            path2,
            ils=ils,
            info=info,
            copy_file=copy_file,
            dst_info=dst_info,
            dst_makedirs=dst_makedirs,
            dst_rm_file=dst_rm_file,
            profile=profile,
            concurrency_limit=concurrency_limit,
            op=op,
        ).outcomes(),
        CpState(),
    )


class _CrossCopy(_Copy):
    """The jobs of one `flat_transfer`: a folder made where needed, a copy per key.

    A key's copy waits for its destination folder, when the destination needs
    it. The keys of one folder come apart in the listing, so the folders made
    are remembered, and each is made once.
    """

    def __init__(
        self,
        path1: str | Sequence[str],
        path2: str,
        *,
        copy_file: _Request,
        dst_info: _Info,
        dst_makedirs: _Request,
        dst_rm_file: _Request,
        profile: StorageProfile,
        **kwargs: Any,
    ) -> None:
        super().__init__(path1, path2, cp_file=copy_file, **kwargs)

        # The destination is another storage's: placed by its own info, and a
        # path may go to the same path, or under it.
        self._placing(dst_info, same_storage=False)

        self.makedirs = dst_makedirs
        self.dst_rm_file = dst_rm_file

        # Unknown, folders are made first. Implicit folders need none.
        self.parents_required = (
            profile.parents_required is not False and not profile.implicit_folders
        )

        # The destination folders made, or tried: their keys need not wait.
        self.made: set[str] = set()

        # The copies waiting for their destination folder, by that folder.
        self.after_mkdir: dict[str, list[_Job]] = {}

    def add(self, entry: dict[str, Any]) -> None:
        name = entry["name"]
        if name.endswith("/"):
            # A folder's marker: a tree destination keeps the folder.
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
            return self.makedirs(job.destination, exist_ok=True)
        return self.copy_file(job.source, job.destination)

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "mkdir":
            self.made.add(job.destination)
            self.waiting.extend(self.after_mkdir.pop(job.destination))
            return []
        return super().done(job)

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if job.kind == "mkdir":
            # The copies still go: each then reports its own failure, if any.
            self.made.add(job.destination)
            self.waiting.extend(self.after_mkdir.pop(job.destination))
            error.add_note(f"Creating the folder {job.destination!r} failed.")
        return [error]

    async def copy_file(self, source: str, destination: str) -> None:
        try:
            try:
                await self.cp_file(source, destination)
            except FileExistsError:
                # The destination refuses to overwrite, as tree_transfer handles
                # it. A folder there is no file to replace. The job holds its
                # permit already.
                entry = await self.destination_info(destination)
                if entry["type"] == "directory":
                    raise

                await self.dst_rm_file(destination)
                await self.cp_file(source, destination)
        except Exception as e:
            # Yielded among paths, the error must tell its paths itself.
            e.add_note(f"{self.doing} {source!r} to {destination!r} failed.")
            raise
