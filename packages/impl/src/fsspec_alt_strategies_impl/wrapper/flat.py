# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Existing filesystems, wrapped as a `FlatHandler`.

`FsspecFlatHandler` wraps an fsspec filesystem, async or sync, so that a
`FlatFileSystem` on it answers its recursive methods by the flat strategies.

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

This module depends on `_base`, and `filesystem` and `handler` of `flat`.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..flat.filesystem import FlatFileSystem
from ..flat.handler import FlatHandler
from ._base import _FsspecHandler, _unwrapped

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from fsspec import AbstractFileSystem

    from fsspec_alt_strategies.profile import StorageProfile

    from ..common.handler import Entry

__all__ = ["FlatFileSystemWrapper", "FsspecFlatHandler"]


class FsspecFlatHandler(_FsspecHandler, FlatHandler):
    """An fsspec filesystem, async or sync, as a flat handler.

    Its primitives, such as ``info``, ``cp_file`` and ``mv_file``, are the
    wrapped filesystem's: its async methods, awaited on its loop, where its
    ``async_impl`` says it has them, else its sync ones, each in a thread. So
    are its data methods, such as ``cat_file``, and its files.

    Its ``ils_recursive`` lists with the wrapped one's ``find`` of the
    directory. Its ``rm_files`` gives a batch to the wrapped one's ``rm``,
    which s3fs, e.g., deletes in one request. That ``rm`` tells no key apart,
    so any failure but a missing key fails the whole batch.

    It wins on object stores, as the flat strategies do. A `FlatFileSystem`
    made on it by ``FlatFileSystem.of`` is not cached: each is new, and may be
    dropped once used.

    Parameters
    ----------
    fs : AbstractFileSystem
        The filesystem to wrap.
    rm_batch_size : int
        The most keys the wrapped ``rm`` is given at once, e.g. 1000 on S3.
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
        If a concurrency, or ``rm_batch_size``, is less than 1.

    See Also
    --------
    FsspecTreeHandler : The same, for storages with a hierarchical namespace.

    Examples
    --------
    >>> flat = FlatFileSystemWrapper(
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
        fs: Any,
        *,
        rm_batch_size: int,
        operation_concurrency: int | asyncio.Semaphore,
        io_concurrency: int | asyncio.Semaphore,
        storage_profile: StorageProfile,
    ) -> None:
        if rm_batch_size < 1:
            raise ValueError(f"rm_batch_size must be at least 1: {rm_batch_size}")
        super().__init__(
            fs,
            operation_concurrency=operation_concurrency,
            io_concurrency=io_concurrency,
            # A flat handler, whatever it wraps.
            storage_profile=replace(storage_profile, fs_type="flat"),
        )
        self.rm_batch_size = rm_batch_size

    async def ils_recursive(  # type: ignore[override]
        self, entry: Entry, *, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[Entry]:
        # fsspec has no listing in pages: the wrapped one's find, all at once.
        path = entry["name"]
        found = await self.call("find", path, detail=True, **kwargs)
        for listed in _starting(list(found.values()), path, prefix):
            yield listed

    async def rm_files(self, entries: list[Entry]) -> Mapping[str, Exception] | None:
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


class FlatFileSystemWrapper(FlatFileSystem):
    """A `FlatFileSystem` on a handler wrapping another filesystem, serving that one's paths.

    As it is made, it takes the handler's `protocol`, `root` and
    `strip_protocol`, the wrapped filesystem's: it takes the URLs that one
    takes, and gives its own. A `FlatFileSystem` takes none of its handler's
    paths: a storage's class names its own.

    Parameters
    ----------
    handler : FsspecFlatHandler
        The handler of the filesystem wrapped.
    **kwargs
        fsspec's own options, as `TreeFileSystem` takes them, e.g. ``asynchronous=True``.

    Examples
    --------
    >>> fs = FlatFileSystemWrapper(FsspecFlatHandler(s3fs.S3FileSystem(), ...))
    """

    cachable = False
    """Not cached: fsspec would key it by its handler's repr, which two wrappers of one filesystem share."""

    handler: FsspecFlatHandler

    def __init__(self, handler: FsspecFlatHandler, **kwargs: Any) -> None:
        super().__init__(handler, **kwargs)

        # The wrapped filesystem's paths, which its handler holds.
        self.protocol = handler.protocol
        self.root_marker = handler.root
        self._strip_protocol = handler.strip_protocol  # type: ignore[method-assign]

    def _unwrapped(self) -> AbstractFileSystem:
        """The filesystem wrapped, as a monitor's `Call` names it: the one its user made."""
        return _unwrapped(self.handler.fs)  # type: ignore[no-any-return]
