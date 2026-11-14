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

    - ``rm_tree``, the wrapped one's ``_rm_tree``, or else its recursive
      ``rm``, where ``deletes_directories=True``;
    - ``mv_tree``, the wrapped one's ``_mv_tree``, or else its ``mv``, where
      ``renames_directories=True``;
    - ``cp_tree``, the wrapped one's ``_cp_tree``, or else its recursive
      ``copy``, where ``copies_directories=True``.

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
        If a concurrency, or a batch size, is less than 1.

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

    async def rm_tree(self, entry: Entry) -> None:
        # The wrapped one's own recursive delete, as deletes_directories declares.
        path = entry["name"]
        if not self.asynchronous:
            await self._run_in_thread(self.fs.rm, path, recursive=True)
        elif (rm_tree := getattr(self.fs, "_rm_tree", None)) is not None:
            await rm_tree(path)
        else:
            await self.fs._rm(path, recursive=True)

    async def mv_tree(self, entry: Entry, path2: StoragePath) -> None:
        # The wrapped one's own rename of a directory, as renames_directories declares.
        path = entry["name"]
        if not self.asynchronous:
            await self._run_in_thread(self.fs.mv, path, path2)
        elif (mv_tree := getattr(self.fs, "_mv_tree", None)) is not None:
            await mv_tree(path, path2)
        else:
            await self.fs._mv_file(path, path2)

    async def cp_tree(self, entry: Entry, path2: StoragePath) -> None:
        # The wrapped one's own copy of a directory, as copies_directories declares.
        path = entry["name"]
        if not self.asynchronous:
            await self._run_in_thread(self.fs.copy, path, path2, recursive=True)
        elif (cp_tree := getattr(self.fs, "_cp_tree", None)) is not None:
            await cp_tree(path, path2)
        else:
            await self.fs._copy(path, path2, recursive=True)
