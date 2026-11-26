# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Existing object stores, wrapped as a `TreeFileOperationHandler` listing keys.

`FsspecFlatHandler` wraps an fsspec filesystem, async or sync, of an object
store, so that a `FileSystemWrapper` on it answers its recursive methods by
the tree strategies, a subtree by one listing.

fsspec has no listing by prefix in pages. The handler's ``ils_recursive``
lists with the wrapped filesystem's ``find`` of the directory, which object
stores answer in one paginated listing, its directories made up by
`implied_directories`. That costs twice:

- ``find`` returns the whole listing at once, so the strategies start only
  once all of it has arrived, and it is held in memory, not streamed.
- A prefix lists its whole directory: for "data/2024-", all of "data/", the
  keys not starting with the prefix then left out.

A subclass overriding ``ils_recursive`` with the storage's own listing by
prefix, e.g. from its SDK, streams it page by page, and lists only the prefix.

This module depends on `_base`, `listing` and `rm_files` of `common`, and
`handler` of `tree`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.profile import StorageProfile

from ..common.listing import implied_directories
from ..common.rm_files import rm_by_request
from ..tree.handler import TreeFileOperationHandler
from ._base import _FsspecHandler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from fsspec import AbstractFileSystem

    from fsspec_alt_strategies.states import RmState

    from ..common.handler import Entry
    from ..common.operation import Operation

__all__ = ["FsspecFlatHandler"]


class FsspecFlatHandler(_FsspecHandler, TreeFileOperationHandler):
    """An fsspec filesystem of an object store, async or sync, as a handler listing keys.

    Its primitives, such as ``ils``, ``info``, ``cp_file`` and ``mv_file``,
    are the wrapped filesystem's: its async methods, awaited on its loop,
    where its ``async_impl`` says it has them, else its sync ones, each in a
    thread. So are its data methods, such as ``cat_file``, and its files.

    Its ``ils_recursive`` lists with the wrapped one's ``find`` of the
    directory. Its profile declares by default what an object store does:
    ``implicit_directories``, ``recursive_ls`` and
    ``recursive_ls_with_prefix``. A profile given declares them itself. Its
    ``rm_files`` gives a batch, of ``rm_batch_size``, to the wrapped one's
    ``rm``, which s3fs, e.g., deletes in one request. That ``rm`` tells no
    key apart, so any failure but a missing key fails the whole batch. Its
    ``rmdir`` deletes a directory's marker so.

    It wins on object stores, listing a subtree in one request. A
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
        What the storage does: by default, an object store's, listing by
        ``find``.
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

    profile = StorageProfile(
        implicit_directories=True, recursive_ls=True, recursive_ls_with_prefix=True
    )

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

    def ils_recursive(
        self, entry: Entry, prefix: bool = False, **kwargs: Any
    ) -> AsyncIterator[Entry]:
        """List all below a directory, or all starting with a prefix, by the wrapped one's ``find``: ``kwargs`` go to it."""
        name = entry["name"]
        top = name.rpartition("/")[0] if prefix else name
        return implied_directories(self._find(name, top, prefix, **kwargs), top)

    async def _find(
        self, name: str, top: str, prefix: bool, **kwargs: Any
    ) -> AsyncIterator[Entry]:
        """The keys below ``top``, starting with ``name`` for a prefix, in name order."""
        # fsspec has no listing in pages: the wrapped one's find, all at once.
        if not prefix:
            found = await self.call("find", name, detail=True, **kwargs)
            if not found:
                await self.call("info", name)  # Missing, it raises FileNotFoundError.
            for listed in sorted(found.values(), key=_name):
                yield listed
            return
        try:
            found = await self.call("find", top, detail=True, **kwargs)
        except FileNotFoundError:
            return  # Nothing starts with the prefix.
        for listed in sorted(found.values(), key=_name):
            if listed["name"].startswith(name):
                yield listed

    async def rmdir(self, entry: Entry) -> None:
        """Delete a directory's marker, the key of its name and "/", by the wrapped one's ``rm``: none is no error."""
        await self._rm_batch([{**entry, "name": f"{entry['name']}/"}])

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


def _name(entry: Entry) -> str:
    return entry["name"]
