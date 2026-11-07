# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive move of the keys of a prefix, each moved as soon as it is listed.

`flat_mv` is an alternative to fsspec's recursive ``mv`` for flat storages.
fsspec's copies every key, then deletes them all, each stage listing the
prefix first. `flat_mv` moves each key as soon as the listing gives it: by one
``mv_file``, or by a copy, deleting the copied keys in batches while the
listing goes on.

This module depends on `_jobs`, `cp`, and `jobs` and `operation` of `common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies_common.states import MvState

from ..common.jobs import _Job
from ..common.operation import Operation
from ._jobs import _keys_of
from .cp import _Copy

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Sequence

    from fsspec_alt_strategies_common.profile import StorageProfile

    from ..common.types import _Ils, _Info, _Request, _Requests

__all__ = ["MvState", "flat_mv"]


def flat_mv(
    path1: str | Sequence[str],
    path2: str,
    *,
    ils: _Ils,
    info: _Info,
    mv_file: _Request | None = None,
    cp_file: _Request | None = None,
    rm_files: _Requests | None = None,
    batch_size: int | None = None,
    profile: StorageProfile,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[tuple[str, str], MvState]:
    """Move the keys under a prefix, each as soon as it is listed.

    It places the paths as fsspec's ``mv`` with ``recursive=True`` does, as
    `flat_cp` places them. Existing files are overwritten.

    It moves each key in one of two ways, by the operations given, and by the
    profile if both are:

    - ``mv_file``: each key is moved by one ``mv_file`` as soon as the
      listing gives it, and yielded once moved. It wins on storages renaming a
      key in one request, which copies no data.
    - ``cp_file``, ``rm_files`` and ``batch_size``: each key is copied by one
      ``cp_file`` as soon as the listing gives it. The keys copied are deleted
      in batches of ``batch_size``, each by one ``rm_files``, and the last,
      smaller batch once all is copied. A key is moved, and yielded, once its
      batch is deleted. It wins on object stores, which rename nothing, but
      delete many keys in one request.

    Given both, it moves by ``mv_file`` only if the profile's
    ``renames_files`` is True. Unknown, it copies and deletes in batches,
    which is right on any storage: about one request per key either way,
    but a rename copies no data.

    A folder's marker, a key ending with "/", is moved too, but not yielded. In
    batches, it is deleted without being copied, as `flat_cp` copies none.

    Unlike fsspec's ``mv``, it keeps going past a failure, and reports each. A
    key that failed to move, or to copy, stays where it was. If a batch fails
    to be deleted, its keys are then in both places, as its error notes.

    It wins over fsspec's ``mv`` on large prefixes of flat storages: the moves
    start after the first page of the listing, and the prefix is listed once,
    not twice.

    Once cancelled, it takes nothing more from the listing, and starts no new
    move or copy. In batches, it still deletes the keys already copied, so
    that none is in both places. It yields the outcomes of what was sent.
    Closed, it waits for the requests in flight without yielding them: keys
    copied but not yet deleted are then in both places. Iterating it raises
    FileNotFoundError if ``path1`` does not exist, or matches nothing, and
    ValueError if ``path2`` is inside it.

    Requests: as `flat_cp`, with one ``mv_file`` per key instead of its
    ``cp_file``, or with one ``rm_files`` more per batch.

    Parameters
    ----------
    path1 : str or sequence of str
        The folder or file to move, a glob pattern, or a list of them, as
        ``ils`` takes paths.
    path2 : str
        Where to move it, as with `flat_cp`.
    ils, info : callable
        The storage's operations, as with `flat_cp`.
    mv_file : callable, optional
        Moves a file, called as ``mv_file(source, destination)``: e.g. the
        filesystem's ``_mv_file``.
    cp_file : callable, optional
        Copies a file, as with `flat_cp`.
    rm_files : callable, optional
        Deletes several files in one request, as with `flat_rm`.
    batch_size : int, optional
        The most keys of one ``rm_files``, e.g. 1000 on S3. ``cp_file``,
        ``rm_files`` and ``batch_size`` are given together, or not at all.
    profile : StorageProfile
        What the storage does: ``StorageProfile()`` if it is unknown. Its
        ``renames_files`` chooses the way when both are given.
    concurrency_limit : asyncio.Semaphore
        Held by each request but the listing while it runs. Share one to
        limit several operations, or files, together.

    Returns
    -------
    Operation
        Yielding ``(source, destination)`` for each key moved, or an error. Its
        state is an `MvState`.

    Raises
    ------
    ValueError
        If neither ``mv_file`` nor all of ``cp_file``, ``rm_files`` and
        ``batch_size`` are given, only some of those three, or
        ``batch_size`` is less than 1.

    See Also
    --------
    flat_cp : The same, keeping the sources.
    tree_mv : The same, for storages listing one folder per request.

    Examples
    --------
    >>> mv = flat_mv(
    ...     "data/2024",
    ...     "archive/",
    ...     ils=my_ils,
    ...     info=fs._info,
    ...     cp_file=fs._cp_file,
    ...     rm_files=my_rm_files,
    ...     batch_size=1000,
    ...     profile=StorageProfile(),
    ...     concurrency_limit=asyncio.Semaphore(32),
    ... )
    >>> async for moved in mv:
    ...     if isinstance(moved, Exception):
    ...         print("failed:", moved)
    """
    batched = [o is not None for o in (cp_file, rm_files, batch_size)]
    if any(batched) and not all(batched):
        raise ValueError("cp_file, rm_files and batch_size are given together")
    if mv_file is None and not all(batched):
        raise ValueError("Give mv_file, or all of cp_file, rm_files and batch_size")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be at least 1: {batch_size}")
    # Both given, a rename wins only where the storage is known to rename.
    if all(batched) and profile.renames_files is not True:
        mv_file = None
    elif mv_file is not None:
        cp_file = rm_files = batch_size = None
    return Operation(
        lambda op: _Move(
            path1,
            path2,
            ils=ils,
            info=info,
            mv_file=mv_file,
            cp_file=cp_file,
            rm_files=rm_files,
            batch_size=batch_size,
            concurrency_limit=concurrency_limit,
            op=op,
        ).outcomes(),
        MvState(),
    )


class _Move(_Copy):
    """The jobs of one `flat_mv`: a move per key, or a copy per key and a deletion per batch.

    Moving by ``mv_file``, it holds it as its ``cp_file``: a "copy" job then
    moves the key.
    """

    verb = "move"
    doing = "Moving"

    def __init__(
        self,
        path1: str | Sequence[str],
        path2: str,
        *,
        mv_file: _Request | None,
        cp_file: _Request | None,
        rm_files: _Requests | None,
        batch_size: int | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(path1, path2, cp_file=mv_file or cp_file, **kwargs)
        # None when each key is moved alone, by mv_file.
        self.rm_files = rm_files
        self.batch_size = batch_size or 0
        # The keys to delete, each with where it was copied, None for a marker.
        self.batch: list[str] = []
        self.copied: dict[str, str | None] = {}
        # The copies queued or running, and whether the listing has ended: the
        # last batch goes once both say all is copied.
        self.copying = 0
        self.listed = False

    def state(self) -> MvState:
        return MvState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_moved=self.files_done,
        )

    def add(self, entry: dict[str, Any]) -> None:
        name = entry["name"]
        if name.endswith("/"):
            # A folder's marker: moved, or deleted only, and not yielded.
            if self.rm_files is None:
                self.waiting.append(_Job("marker", name, self.target(name)))
            else:
                self.delete_later(name, None)
            return
        if self.rm_files is not None:
            self.copying += 1
        super().add(entry)

    def exhausted(self) -> None:
        self.listed = True
        self.send_if_ready()

    def stopping(self) -> None:
        if self.rm_files is None:
            super().stopping()
            return
        # The copies not started are dropped, but the keys copied are deleted,
        # so that none is in both places.
        deletions = [j for j in self.waiting if j.kind == "delete"]
        self.waiting.clear()
        self.waiting.extend(deletions)
        self.copying = sum(j.kind == "copy" for j in self.running.values())
        self.send_if_ready()

    def delete_later(self, source: str, destination: str | None) -> None:
        """Add a key to the batch to delete, and send the batch if it is ready."""
        self.copied[source] = destination
        self.batch.append(source)
        self.send_if_ready()

    def send_if_ready(self) -> None:
        """Send the batch if full, or if nothing more can join it."""
        if not self.batch:
            return
        done = (self.listed or self.op.cancelling()) and self.copying == 0
        if len(self.batch) >= self.batch_size or done:
            paths, self.batch = tuple(self.batch), []
            self.waiting.append(_Job("delete", paths[-1], paths=paths))

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "delete":
            return self.rm_files(list(job.paths))  # type: ignore[misc]
        return super().run(job)

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "marker":
            return []
        if job.kind == "copy":
            if self.rm_files is None:
                return super().done(job)  # Moved, by mv_file.
            self.copying -= 1
            self.delete_later(job.source, job.destination)
            return []
        moved = []
        for path in job.paths:
            if (destination := self.copied.pop(path)) is not None:
                self.files_done += 1
                moved.append((path, destination))
        return moved

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        if job.kind == "marker" or (job.kind == "copy" and self.rm_files is None):
            return super().failed(job, error)  # Noted in copy_file.
        if job.kind == "copy":
            self.copying -= 1
            self.send_if_ready()
            return super().failed(job, error)
        for path in job.paths:
            self.copied.pop(path)
        error.add_note(
            f"Deleting {_keys_of(job.paths)} failed, once copied: they are in both"
            " places."
        )
        return [error]
