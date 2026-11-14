# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Existing filesystems, wrapped as a `TreeHandler`.

`FsspecTreeHandler` wraps an fsspec filesystem, async or sync, so that a
`TreeFileSystem` on it answers its recursive methods by the tree strategies.
It lists a tree a directory per request, by the wrapped one's ``ls``.

This module depends on `_base`, and `filesystem` and `handler` of `tree`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..tree.filesystem import TreeFileSystem
from ..tree.handler import TreeHandler
from ._base import _FsspecHandler, _unwrapped

if TYPE_CHECKING:
    import asyncio

    from fsspec import AbstractFileSystem

    from fsspec_alt_strategies.profile import StorageProfile

    from ..common.handler import Entry, StoragePath

__all__ = ["FsspecTreeHandler", "TreeFileSystemWrapper"]


class FsspecTreeHandler(_FsspecHandler, TreeHandler):
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
    disk or SFTP, as the tree strategies do. A `TreeFileSystem` made on it by
    ``TreeFileSystem.of`` is not cached: each is new, and may be dropped once
    used.

    Parameters
    ----------
    fs : AbstractFileSystem
        The filesystem to wrap.
    operation_concurrency : int or asyncio.Semaphore
        The requests of the strategies at a time.
        A semaphore is used as it is, shared with what else holds it.
    io_concurrency : int or asyncio.Semaphore
        The data methods' calls at a time, such as ``cat_file``.
        A semaphore is used as it is, shared with what else holds it.
    storage_profile : StorageProfile
        What the storage does. ``UNKNOWN`` if it is unknown.

    Raises
    ------
    ValueError
        If a concurrency is less than 1.

    See Also
    --------
    FsspecFlatHandler : The same, for flat storages.

    Examples
    --------
    >>> local = fsspec.filesystem("file")
    >>> tree = TreeFileSystemWrapper(
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
        fs: Any,
        *,
        operation_concurrency: int | asyncio.Semaphore,
        io_concurrency: int | asyncio.Semaphore,
        storage_profile: StorageProfile,
    ) -> None:
        super().__init__(
            fs,
            operation_concurrency=operation_concurrency,
            io_concurrency=io_concurrency,
            # A tree handler, whatever it wraps.
            storage_profile=replace(storage_profile, fs_type="tree"),
        )

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


class TreeFileSystemWrapper(TreeFileSystem):
    """A `TreeFileSystem` on a handler wrapping another filesystem, serving that one's paths.

    As it is made, it takes the handler's `protocol`, `root` and
    `strip_protocol`, the wrapped filesystem's: it takes the URLs that one
    takes, and gives its own. A `TreeFileSystem` takes none of its handler's
    paths: a storage's class names its own.

    Parameters
    ----------
    handler : FsspecTreeHandler
        The handler of the filesystem wrapped.
    **kwargs
        fsspec's own options, as `TreeFileSystem` takes them, e.g. ``asynchronous=True``.

    Examples
    --------
    >>> fs = TreeFileSystemWrapper(FsspecTreeHandler(fsspec.filesystem("file"), ...))
    """

    cachable = False
    """Not cached: fsspec would key it by its handler's repr, which two wrappers of one filesystem share."""

    handler: FsspecTreeHandler

    def __init__(self, handler: FsspecTreeHandler, **kwargs: Any) -> None:
        super().__init__(handler, **kwargs)

        # The wrapped filesystem's paths, which its handler holds.
        self.protocol = handler.protocol
        self.root_marker = handler.root
        self._strip_protocol = handler.strip_protocol  # type: ignore[method-assign]

    def _unwrapped(self) -> AbstractFileSystem:
        """The filesystem wrapped, as a monitor's `Call` names it: the one its user made."""
        return _unwrapped(self.handler.fs)  # type: ignore[no-any-return]
