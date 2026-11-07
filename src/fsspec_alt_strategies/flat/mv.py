# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive move that copies the keys of a prefix, and deletes them in batches.

`flat_mv` is an alternative to fsspec's recursive ``mv`` for flat storages.
fsspec's copies every key, then deletes them all, each stage listing the
prefix first. `flat_mv` copies each key as soon as the listing gives it, and
deletes the copied keys in batches while the listing goes on.

It is for storages without a rename of their own, such as object stores.

This module depends on `control`, `_jobs`, `cp`, and `jobs` and `states` of
`common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.jobs import _Job
from ..common.states import MvState
from ..control import Operation
from ._jobs import _keys_of
from .cp import _Copy

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Sequence

    from ..common.operations import _Ils, _Info, _Request, _Requests

__all__ = ["MvState", "flat_mv"]


def flat_mv(
    path1: str | Sequence[str],
    path2: str,
    *,
    ils: _Ils,
    info: _Info,
    cp_file: _Request,
    rm_files: _Requests,
    batch_size: int,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[tuple[str, str], MvState]:
    """Move the keys under a prefix: copy each as listed, delete them in batches.

    It places the paths as fsspec's ``mv`` with ``recursive=True`` does, as
    `flat_cp` places them. Existing files are overwritten.

    Each key is copied by one ``cp_file`` as soon as the listing gives it. The
    keys copied are deleted in batches of ``batch_size``, each by one
    ``rm_files``, and the last, smaller batch once all is copied. A key is
    moved, and yielded, once its batch is deleted. A folder's marker, a key
    ending with "/", is deleted without being copied, as `flat_cp` copies
    none.

    Unlike fsspec's ``mv``, it keeps going past a failure, and reports each. A
    key that failed to copy stays where it was. If a batch fails to be
    deleted, its keys are then in both places, as its error notes.

    It wins over fsspec's ``mv`` on large prefixes of flat storages: the moves
    start after the first page of the listing, and the prefix is listed once,
    not twice. It loses on storages that rename a file or a folder in one
    request: that request wins.

    Once cancelled, it takes nothing more from the listing, and starts no new
    copy. It still deletes the keys already copied, so that none is in both
    places, and yields the outcomes of what was sent. Closed, it waits for the
    requests in flight without yielding them: keys copied but not yet deleted
    are then in both places. Iterating it raises FileNotFoundError if
    ``path1`` does not exist, or matches nothing, and ValueError if ``path2``
    is inside it.

    Requests: as `flat_cp`, with one ``rm_files`` per batch.

    Parameters
    ----------
    path1 : str or sequence of str
        The folder or file to move, a glob pattern, or a list of them, as
        ``ils`` takes paths.
    path2 : str
        Where to move it, as with `flat_cp`.
    ils, info, cp_file : callable
        The storage's operations, as with `flat_cp`.
    rm_files : callable
        Deletes several files in one request, as with `flat_rm`.
    batch_size : int
        The most keys of one ``rm_files``, e.g. 1000 on S3.
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
        If ``batch_size`` is less than 1.

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
    ...     concurrency_limit=asyncio.Semaphore(32),
    ... )
    >>> async for moved in mv:
    ...     if isinstance(moved, Exception):
    ...         print("failed:", moved)
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1: {batch_size}")
    return Operation(
        lambda op: _Move(
            path1,
            path2,
            ils=ils,
            info=info,
            cp_file=cp_file,
            rm_files=rm_files,
            batch_size=batch_size,
            concurrency_limit=concurrency_limit,
            op=op,
        ).outcomes(),
        MvState(),
    )


class _Move(_Copy):
    """The jobs of one `flat_mv`: a copy per key, then a deletion per batch."""

    verb = "move"
    doing = "Moving"

    def __init__(
        self,
        path1: str | Sequence[str],
        path2: str,
        *,
        rm_files: _Requests,
        batch_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(path1, path2, **kwargs)
        self.rm_files = rm_files
        self.batch_size = batch_size
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
            self.delete_later(name, None)  # A folder's marker: deleted only.
            return
        self.copying += 1
        super().add(entry)

    def exhausted(self) -> None:
        self.listed = True
        self.send_if_ready()

    def stopping(self) -> None:
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
        done = (self.listed or self.op.cancelling()) and self.copying == 0
        if len(self.batch) >= self.batch_size or (self.batch and done):
            paths, self.batch = tuple(self.batch), []
            self.waiting.append(_Job("delete", paths[-1], paths=paths))

    def run(self, job: _Job) -> Awaitable[Any]:
        if job.kind == "delete":
            return self.rm_files(list(job.paths))
        return super().run(job)

    def done(self, job: _Job) -> list[Any]:
        if job.kind == "copy":
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
