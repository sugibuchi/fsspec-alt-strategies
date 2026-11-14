# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Existing filesystems, wrapped as a `FlatFileOperationHandler`.

`FsspecFlatHandler` wraps an fsspec filesystem, async or sync, so that a
`FileSystemWrapper` on it answers its recursive methods by the flat
strategies.

fsspec has no listing by prefix in pages, which the flat strategies list with.
The handler's ``ils_recursive`` lists with the wrapped filesystem's ``find``
of the directory, which object stores answer in one paginated listing. That
costs twice:

- ``find`` returns the whole listing at once, so the strategies start only
  once all of it has arrived, and it is held in memory, not streamed.
- It lists the whole directory: for "data" with the prefix "2024-", all of
  "data/", the keys not starting with the prefix then left out.

A subclass overriding ``ils_recursive`` with the storage's own listing by
prefix, e.g. from its SDK, streams it page by page, and lists only the prefix.

This module depends on `_base`, `rm_files` of `common`, and `handler` of `flat`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from ..common.rm_files import rm_by_request
from ..flat.handler import FlatFileOperationHandler
from ._base import _FsspecHandler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from fsspec import AbstractFileSystem

    from fsspec_alt_strategies.profile import StorageProfile
    from fsspec_alt_strategies.states import RmState

    from ..common.handler import Entry
    from ..common.operation import Operation

__all__ = ["FsspecFlatHandler"]


class FsspecFlatHandler(_FsspecHandler, FlatFileOperationHandler):
    """An fsspec filesystem, async or sync, as a flat handler.

    Its primitives, such as ``info``, ``cp_file`` and ``mv_file``, are the
    wrapped filesystem's: its async methods, awaited on its loop, where its
    ``async_impl`` says it has them, else its sync ones, each in a thread. So
    are its data methods, such as ``cat_file``, and its files.

    Its ``ils_recursive`` lists with the wrapped one's ``find`` of the
    directory. Its ``rm_files`` gives a batch, of ``rm_batch_size``, to the wrapped one's ``rm``,
    which s3fs, e.g., deletes in one request. That ``rm`` tells no key apart,
    so any failure but a missing key fails the whole batch.

    It wins on object stores, as the flat strategies do. A
    `FileSystemWrapper` made on it is not cached: each is new, and may be
    dropped once used.

    Parameters
    ----------
    fs : AbstractFileSystem
        The filesystem to wrap.
    operation_concurrency : int or asyncio.Semaphore, optional
        The requests of the strategies at a time: 1 by default.
        A semaphore is used as it is, shared with what else holds it.
    io_concurrency : int or asyncio.Semaphore, optional
        The data methods' calls at a time, such as ``cat_file``: 1 by default.
        A semaphore is used as it is, shared with what else holds it.
    storage_profile : StorageProfile, optional
        What the storage does: unknown by default.
    rm_batch_size : int, optional
        The most keys the wrapped ``rm`` is given at once, e.g. 1000 on S3: 1
        by default.
    cp_batch_size, mv_batch_size : int, optional
        The most files ``cp_files`` and ``mv_files`` take at once: 1 by
        default, as each goes by the wrapped one's own copy or move.

    Raises
    ------
    ValueError
        If a concurrency is less than 1. A batch size less than 1 is refused by
        the strategy batching by it, as it starts.
    See Also
    --------
    FsspecTreeHandler : The same, for storages with a hierarchical namespace.

    Examples
    --------
    >>> flat = FileSystemWrapper(
    ...     FsspecFlatHandler(
    ...         s3,
    ...         rm_batch_size=1000,
    ...         operation_concurrency=64,
    ...         io_concurrency=8,
    ...         storage_profile=UNKNOWN,
    ...     )
    ... )
    >>> with monitoring(Progress()):
    ...     flat.rm("bucket/data/2024", recursive=True)
    """

    def __init__(
        self,
        fs: AbstractFileSystem,
        *,
        operation_concurrency: int | asyncio.Semaphore | None = None,
        io_concurrency: int | asyncio.Semaphore | None = None,
        storage_profile: StorageProfile | None = None,
        rm_batch_size: int | None = None,
        cp_batch_size: int | None = None,
        mv_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            fs,
            operation_concurrency=operation_concurrency,
            io_concurrency=io_concurrency,
            storage_profile=storage_profile,
        )
        # Its strategies' batches, the Strategy mixin's: the class's where left out.
        if rm_batch_size is not None:
            self.rm_batch_size = rm_batch_size
        if cp_batch_size is not None:
            self.cp_batch_size = cp_batch_size
        if mv_batch_size is not None:
            self.mv_batch_size = mv_batch_size

    async def ils_recursive(
        self, entry: Entry, *, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[Entry]:
        # fsspec has no listing in pages: the wrapped one's find, all at once.
        path = entry["name"]
        found = await self.call("find", path, detail=True, **kwargs)
        for listed in _starting(list(found.values()), path, prefix):
            yield listed

    def rm_files(self, entries: list[Entry]) -> Operation[str, RmState]:
        # A batch by one call of the wrapped one's rm: see _rm_batch.
        return rm_by_request(self, entries, self._rm_batch)

    async def _rm_batch(self, entries: list[Entry]) -> Mapping[str, Exception] | None:
        """Delete a batch by the wrapped one's ``rm`` of all its paths, which s3fs, e.g., sends as one request."""
        paths = [e["name"] for e in entries]
        try:
            await self.call("rm", paths)
        except FileNotFoundError:
            # A key gone meanwhile: the batch again, key by key, a missing key no error.
            await asyncio.gather(*(self._rm_if_there(p) for p in paths))
        # fsspec's rm tells no key refused apart: a failure is the batch's.
        return None

    async def _rm_if_there(self, path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            await self.call("rm_file", path)


def _starting(
    listing: list[dict[str, Any]], path: str, prefix: str
) -> list[dict[str, Any]]:
    """A listing's entries starting with ``path/prefix``, in name order, as ``ils_recursive`` yields them."""
    if not prefix:
        return sorted(listing, key=lambda e: e["name"])
    start = f"{path}/{prefix}" if path else prefix
    return sorted(
        (e for e in listing if e["name"].startswith(start)), key=lambda e: e["name"]
    )
