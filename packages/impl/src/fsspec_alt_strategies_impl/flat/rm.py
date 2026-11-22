# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive delete that deletes the keys of a prefix in batches, as it lists them.

`_flat_rm` is an alternative to fsspec's recursive ``_rm`` for flat storages.
fsspec's, and the object stores' own, list every key first, then delete them,
in batches where the storage can. `_flat_rm` sends each batch as soon as the
listing has given its keys.

This module depends on `_jobs`, and `handler`, `jobs` and `paths` of `common`.
"""

from __future__ import annotations

from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import RmState

from ..common.jobs import _Job, _keys_of, _Roots
from ..common.paths import _check_maxdepth, _trim
from ._jobs import _FlatJobs

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from ..common.handler import StoragePath
    from ..common.operation import Operation
    from .handler import FlatFileOperationHandler

__all__: list[str] = []


def _flat_rm(
    handler: FlatFileOperationHandler,
    path: StoragePath | Sequence[StoragePath],
    *,
    maxdepth: int | None = None,
    ils_kwargs: dict[str, Any] | None = None,
) -> Operation[str, RmState]:
    """Delete the keys under a prefix, in batches sent as the listing goes on.

    It deletes what fsspec's ``_rm`` with ``recursive=True`` deletes: the
    keys under ``path``, or ``path`` itself if it is a file. A directory's
    marker, a key ending with "/", is deleted as any key is. A glob pattern, or
    a list of paths and patterns, deletes all they match, as with fsspec's
    ``_rm``. They are matched as `_flat_glob` matches them, then the keys under
    the directories matched are listed, one directory after another.

    With ``maxdepth``, it deletes the keys fsspec's ``_rm`` deletes with it:
    those at most ``maxdepth`` levels below ``path``, or below a pattern's
    matches one level less. A directory's marker counts at its directory's
    level. The keys deeper are listed all the same, as a prefix lists every
    key under it, but kept.

    The keys are grouped in batches of the handler's ``rm_batch_size``, each
    deleted by one ``rm_files``, while the listing goes on. The last batch,
    smaller, goes once the listing ends. It yields each key once its batch is
    deleted, the error of each key its batch could not delete, noting the key,
    and the error of each batch that failed whole, noting its keys.

    It wins over fsspec's ``_rm`` on large prefixes of flat storages: the
    deletions start after the first page of the listing, not after the last,
    and the keys are held a batch at a time. It sends the same requests as a
    storage deleting in batches of ``rm_batch_size``. It loses on storages
    with a hierarchical namespace, whose directories a listing by prefix does
    not see: `_tree_rm` suits those.

    Once cancelled, it takes nothing more from the listing, and sends no new
    batch, but yields the outcomes of those sent. Closed, it waits for those
    without yielding them. Iterating it raises FileNotFoundError if ``path``
    does not exist, or matches nothing. An error from the listing ends it too.

    Requests: the pages of one recursive ``ils`` per root, one after another,
    and one ``rm_files`` per batch. One ``info`` for ``path`` if nothing is
    under it. Patterns, or a list, add the requests of their `_flat_glob`.
    Each request but the listing holds a permit of the handler's
    ``operation_limit`` while it runs.

    ``path`` is the directory deleted from ``handler``, as its ``ils`` takes
    paths. A file is deleted alone. A glob pattern, or a list of paths and
    patterns, deletes all they match. ``maxdepth`` of 1 deletes only the keys
    right in ``path``. The handler's ``rm_files`` takes a key missing as
    deleted, and yields each key deleted or the error of each it could not
    delete. ``ils_kwargs`` go to
    each recursive ``ils``, a pattern's glob's too.

    The operation yields each key deleted, or an error, and its state is an
    `RmState`. The call raises ValueError if ``maxdepth`` is less than 1, and
    iterating it if the handler's ``rm_batch_size`` is, read once it is open.
    """
    _check_maxdepth(maxdepth)

    return _Deletion.operation(
        handler=handler, path=path, maxdepth=maxdepth, ils_kwargs=ils_kwargs
    )


class _Deletion(_FlatJobs):
    """The jobs of one `_flat_rm`: a deletion per batch of keys."""

    def __init__(
        self, *, path: StoragePath | Sequence[StoragePath], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.path = path
        self.files_found = 0
        self.files_deleted = 0

    async def prepare(self) -> _Roots:
        if not isinstance(self.path, str) or has_magic(self.path):
            return await self.glob_roots(self.path)
        else:
            return _Roots([{"name": _trim(self.path)}])

    def state(self) -> RmState:
        return RmState(
            errors=self.errors,
            current=self.current,
            files_found=self.files_found,
            files_deleted=self.files_deleted,
        )

    def add(self, entry: dict[str, Any]) -> None:
        self.files_found += 1
        self.queue_deletion(entry)

    def exhausted(self) -> None:
        self.flush_deletions()

    def run(self, job: _Job) -> Awaitable[Any]:
        raise AssertionError(f"Only batches of deletions run: {job.kind}")

    def done(self, job: _Job) -> list[Any]:
        deleted, errors = self.deleted(job)
        self.files_deleted += len(deleted)
        return [*deleted, *errors]

    def failed(self, job: _Job, error: Exception) -> list[Any]:
        # Yielded among paths, the error must tell its paths itself.
        error.add_note(f"Deleting {_keys_of(job.paths)} failed.")
        return [error]
