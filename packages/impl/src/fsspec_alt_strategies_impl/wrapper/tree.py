# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Existing filesystems, wrapped as a `TreeFileOperationHandler`.

`FsspecTreeHandler` wraps an fsspec filesystem, async or sync, so that a
`FileSystemWrapper` on it answers its recursive methods by the tree
strategies. It lists a tree a directory per request, by the wrapped one's
``ls``.

This module depends on `_base`, and `handler` of `tree`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tree.handler import TreeFileOperationHandler
from ._base import _FsspecHandler

if TYPE_CHECKING:
    import asyncio

    from fsspec import AbstractFileSystem

    from fsspec_alt_strategies.profile import StorageProfile

    from ..common.handler import Entry, StoragePath

__all__ = ["FsspecTreeHandler"]


class FsspecTreeHandler(_FsspecHandler, TreeFileOperationHandler):
    """An fsspec filesystem, async or sync, as a tree handler.

    Its primitives, such as ``ils``, ``rm_file`` and ``mv_file``, are the
    wrapped filesystem's: its async methods, awaited on its loop, where its
    ``async_impl`` says it has them, else its sync ones, each in a thread. So
    are its data methods, such as ``cat_file``, and its files. So are the
    requests on a directory whole, each used where the profile says it is one
    request:

    - ``rm_tree``, the wrapped one's recursive ``rm``, where
      ``deletes_directories=True``;
    - ``mv_tree``, its ``mv``, or async its ``_mv_file``, where
      ``renames_directories=True``;
    - ``cp_tree``, its recursive ``copy``, where ``copies_directories=True``.

    It wins on storages listing one directory per request, such as a local
    disk or SFTP, as the tree strategies do. A `FileSystemWrapper` made on it
    is not cached: each is new, and may be dropped once used.

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
    rm_batch_size, cp_batch_size, mv_batch_size : int, optional
        The most files ``rm_files``, ``cp_files`` and ``mv_files`` take at
        once: 1 by default, as each goes by the wrapped one's own primitive.

    Raises
    ------
    ValueError
        If a concurrency is less than 1. A batch size less than 1 is refused by
        the strategy batching by it, as it starts.

    See Also
    --------
    FsspecFlatHandler : The same, for flat storages.

    Examples
    --------
    >>> local = fsspec.filesystem("file")
    >>> tree = FileSystemWrapper(
    ...     FsspecTreeHandler(
    ...         local,
    ...         operation_concurrency=8,
    ...         io_concurrency=4,
    ...         storage_profile=StorageProfile(renames_directories=True),
    ...     )
    ... )
    >>> tree.mv("data/2024", "archive/2024", recursive=True)
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

    async def rm_tree(self, entry: Entry) -> None:
        # The wrapped one's own recursive delete, as deletes_directories declares.
        path = entry["name"]
        if not self.asynchronous:
            await self.run_in_thread(self.fs.rm, path, recursive=True)
        else:
            await self.fs._rm(path, recursive=True)

    async def mv_tree(
        self, entry: Entry, path2: StoragePath, *, parent: Entry | None = None
    ) -> None:
        # The wrapped one's own rename of a directory, as renames_directories declares.
        path = entry["name"]
        if not self.asynchronous:
            await self.run_in_thread(self.fs.mv, path, path2)
        else:
            await self.fs._mv_file(path, path2)

    async def cp_tree(
        self, entry: Entry, path2: StoragePath, *, parent: Entry | None = None
    ) -> None:
        # The wrapped one's own copy of a directory, as copies_directories declares.
        path = entry["name"]
        if not self.asynchronous:
            await self.run_in_thread(self.fs.copy, path, path2, recursive=True)
        else:
            await self.fs._copy(path, path2, recursive=True)
