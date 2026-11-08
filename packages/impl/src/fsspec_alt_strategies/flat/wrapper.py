# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Existing filesystems, wrapped as a `FlatFileSystem`, to use the flat strategies.

`AsyncFlatWrapper` wraps an async filesystem, `SyncFlatWrapper` a sync one. A
wrapper is light, and not cached: make one on demand, e.g. to watch a call
with a monitor, and drop it after. The developer chooses the wrapper: none is
chosen for a filesystem.

fsspec has no listing by prefix, which the flat strategies list with. A
wrapper's ``_ils`` lists with the wrapped filesystem's ``find`` of the
prefix's folder, which object stores answer in one paginated listing. That
costs twice:

- ``find`` returns the whole listing at once, so the strategies start only
  once all of it has arrived, and it is held in memory, not streamed.
- It lists the prefix's whole folder: for "data/2024-", all of "data/", the
  keys not starting with the prefix then left out.

A subclass overriding ``_ils`` with the storage's own listing by prefix, e.g.
from its SDK, streams it page by page, and lists only the prefix.

This module depends on `filesystem`, and `wrapper` of `common`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from ..common.wrapper import _AsyncFileSystemWrapper, _SyncFileSystemWrapper
from .filesystem import FlatFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fsspec.asyn import AsyncFileSystem
    from fsspec.spec import AbstractFileSystem

    from fsspec_alt_strategies_common.profile import StorageProfile

__all__ = ["AsyncFlatWrapper", "SyncFlatWrapper"]


class AsyncFlatWrapper(_AsyncFileSystemWrapper, FlatFileSystem):
    """An async filesystem, as a `FlatFileSystem`: its recursive methods the flat strategies.

    Its primitives, such as ``_info``, ``_cp_file`` and ``_mv_file``, are the
    wrapped filesystem's, awaited on its loop. So are its data methods, such
    as ``_cat_file``, and its files. The wrapper runs on the same loop, sync or
    not as the wrapped one is.

    Its ``_ils`` lists with the wrapped one's ``_find`` of the prefix's folder,
    and its ``_rm_files`` deletes with its ``_rm``, as s3fs deletes a batch in
    one request. It wins on object stores, as `FlatFileSystem` does. It is not
    cached: each is new, and may be dropped once used.

    Parameters
    ----------
    fs : AsyncFileSystem
        The filesystem to wrap.
    rm_batch_size : int
        The most keys the wrapped ``_rm`` deletes in one request, e.g. 1000 on
        S3.
    operation_concurrency : int or asyncio.Semaphore
        The requests of the strategies at a time.
        A semaphore is used as it is, shared with what else holds it.
    io_concurrency : int or asyncio.Semaphore
        The data methods' calls at a time, such as ``_cat_file``.
        A semaphore is used as it is, shared with what else holds it.
    storage_profile : StorageProfile
        What the storage does. ``StorageProfile()`` if it is unknown.
    **kwargs
        Passed to `fsspec.asyn.AsyncFileSystem`.

    Raises
    ------
    ValueError
        If a concurrency, or ``rm_batch_size``, is less than 1.

    See Also
    --------
    SyncFlatWrapper : The same, for a sync filesystem.

    Examples
    --------
    >>> flat = AsyncFlatWrapper(
    ...     s3,
    ...     rm_batch_size=1000,
    ...     operation_concurrency=64,
    ...     io_concurrency=8,
    ...     storage_profile=StorageProfile(),
    ... )
    >>> with monitoring(Progress()):
    ...     flat.rm("bucket/data/2024", recursive=True)
    """

    def __init__(
        self,
        fs: AsyncFileSystem,
        *,
        rm_batch_size: int,
        operation_concurrency: int | asyncio.Semaphore,
        io_concurrency: int | asyncio.Semaphore,
        storage_profile: StorageProfile,
        **kwargs: Any,
    ) -> None:
        self.rm_batch_size = rm_batch_size
        super().__init__(
            fs,
            operation_concurrency=operation_concurrency,
            io_concurrency=io_concurrency,
            storage_profile=storage_profile,
            **kwargs,
        )

    async def _ils(self, prefix: str) -> AsyncIterator[dict[str, Any]]:
        found = await self.fs._find(prefix.rpartition("/")[0], detail=True)
        for entry in _under(found, prefix):
            yield entry

    async def _rm_files(self, paths: list[str]) -> None:
        try:
            await self.fs._rm(list(paths))
        except FileNotFoundError:
            # A key gone meanwhile: the rest one by one, a missing key no error.
            await asyncio.gather(*(self._rm_if_there(p) for p in paths))

    async def _rm_if_there(self, path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            await self.fs._rm_file(path)


class SyncFlatWrapper(_SyncFileSystemWrapper, FlatFileSystem):
    """A sync filesystem, as a `FlatFileSystem`: its recursive methods the flat strategies.

    Its primitives, such as ``_info``, ``_cp_file`` and ``_mv_file``, call the
    wrapped filesystem's, each in a thread, so that several run at once. Its
    ``_mv_file`` is the wrapped one's ``mv``. Its data methods call the
    wrapped one's too, and its files are the wrapped one's, awaited through a
    `SyncFileAdapter`.

    Its ``_ils`` lists with the wrapped one's ``find`` of the prefix's folder,
    and its ``_rm_files`` deletes with its ``rm``. A filesystem unsafe to call
    from several threads needs both concurrencies at 1. It is not cached: each
    is new, and may be dropped once used.

    Parameters
    ----------
    fs : AbstractFileSystem
        The filesystem to wrap.
    rm_batch_size : int
        The most keys the wrapped ``rm`` is given at once.
    operation_concurrency : int or asyncio.Semaphore
        The requests of the strategies at a time, each in a thread.
        A semaphore is used as it is, shared with what else holds it.
    io_concurrency : int or asyncio.Semaphore
        The data methods' calls at a time, such as ``_cat_file``.
        A semaphore is used as it is, shared with what else holds it.
    storage_profile : StorageProfile
        What the storage does. ``StorageProfile()`` if it is unknown.
    **kwargs
        Passed to `fsspec.asyn.AsyncFileSystem`, e.g. ``asynchronous=True``.

    Raises
    ------
    ValueError
        If a concurrency, or ``rm_batch_size``, is less than 1.

    See Also
    --------
    AsyncFlatWrapper : The same, for an async filesystem.

    Examples
    --------
    >>> flat = SyncFlatWrapper(
    ...     fs,
    ...     rm_batch_size=1000,
    ...     operation_concurrency=8,
    ...     io_concurrency=4,
    ...     storage_profile=StorageProfile(),
    ... )
    >>> flat.find("bucket/data")
    """

    def __init__(
        self,
        fs: AbstractFileSystem,
        *,
        rm_batch_size: int,
        operation_concurrency: int | asyncio.Semaphore,
        io_concurrency: int | asyncio.Semaphore,
        storage_profile: StorageProfile,
        **kwargs: Any,
    ) -> None:
        self.rm_batch_size = rm_batch_size
        super().__init__(
            fs,
            operation_concurrency=operation_concurrency,
            io_concurrency=io_concurrency,
            storage_profile=storage_profile,
            **kwargs,
        )

    async def _ils(self, prefix: str) -> AsyncIterator[dict[str, Any]]:
        found = await asyncio.to_thread(
            self.fs.find, prefix.rpartition("/")[0], detail=True
        )
        for entry in _under(found, prefix):
            yield entry

    async def _rm_files(self, paths: list[str]) -> None:
        try:
            await asyncio.to_thread(self.fs.rm, list(paths))
        except FileNotFoundError:
            # A key gone meanwhile: the rest one by one, a missing key no error.
            await asyncio.gather(*(self._rm_if_there(p) for p in paths))

    async def _rm_if_there(self, path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            await asyncio.to_thread(self.fs.rm_file, path)


def _under(found: dict[str, dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """The entries of a ``find`` starting with ``prefix``, as ``ils`` yields them."""
    return [e for name, e in found.items() if name.startswith(prefix)]
