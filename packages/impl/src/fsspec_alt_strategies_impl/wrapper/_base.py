# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What the wrapping handlers share: an fsspec filesystem's methods, awaited or run in threads.

`_FsspecHandler` calls the filesystem it wraps: its async methods, awaited on
its loop, where ``fs.async_impl`` says it has them, or else its sync ones, each
in a thread of the handler's own. So a sync filesystem, e.g. a
``DirFileSystem`` over a local one, is wrapped as sync, whatever its class. A
family's handler adds its own primitives.

This module depends on `handler` and `sync_client` of `common`, and
`fsspec_async` of `files`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.handler import StorageHandler
from ..common.sync_client import SyncClientSupport
from ..files.fsspec_async import SyncFileAdapter

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncIterator

    from fsspec.callbacks import Callback

    from fsspec_alt_strategies.profile import StorageProfile

    from ..common.handler import Entry, StoragePath

__all__: list[str] = []


def _async_impl(fs: Any) -> bool:
    """Whether a filesystem's async methods work: its ``async_impl``, and that of the one it chains over.

    A filesystem chained over another, e.g. fsspec's ``DirFileSystem`` in its
    ``fs``, declares ``async_impl`` on its class, but its async methods call
    the other's: they work only if that one's do.
    """
    if not getattr(fs, "async_impl", False):
        return False
    inner = getattr(fs, "fs", None)
    if inner is not None and hasattr(inner, "async_impl"):
        return _async_impl(inner)
    return True


class _FsspecHandler(SyncClientSupport, StorageHandler):
    """An fsspec filesystem's primitives, its data methods and its files, as a handler's.

    Its paths are the wrapped filesystem's: it verifies none, as the wrapped
    one checks its own, and strips and unstrips the protocol as it does.

    A sync filesystem's calls run in threads of the handler's own, as many as
    both concurrencies together, when both are numbers: a filesystem unsafe to
    call from several threads needs both at 1. `close` and `aclose` shut them down.
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
            operation_concurrency=operation_concurrency, io_concurrency=io_concurrency
        )
        self.fs = fs
        self.profile = storage_profile
        # The wrapped one's paths: its protocol, its root and its stripping.
        self.protocol = fs.protocol
        self.root = getattr(fs, "root_marker", "")
        # Its async methods only where they work.
        self.asynchronous = _async_impl(fs)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} of {self.fs!r}>"

    @property
    def wrapped(self) -> Any:
        """The filesystem wrapped, itself unwrapped if a wrapper: a monitor's `Call` names it."""
        unwrap = getattr(self.fs, "_unwrapped", None)
        return unwrap() if unwrap is not None else self.fs

    def _thread_count(self) -> int | None:
        # The data methods hold no permit of operation_limit: they need threads
        # of their own.
        concurrencies = (self.operation_concurrency, self.io_concurrency)
        if isinstance(concurrencies[0], int) and isinstance(concurrencies[1], int):
            return concurrencies[0] + concurrencies[1]
        else:
            return None

    async def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """The wrapped filesystem's method ``name``: its ``_name`` awaited, or ``name`` in a thread."""
        if self.asynchronous:
            return await getattr(self.fs, f"_{name}")(*args, **kwargs)
        else:
            return await self._run_in_thread(getattr(self.fs, name), *args, **kwargs)

    # Its paths, as the wrapped filesystem's.

    def verify_path(self, path: str) -> None:
        """Nothing: the wrapped filesystem checks its paths in its own methods."""

    def strip_protocol(self, path: Any) -> Any:  # type: ignore[override]
        # The wrapped filesystem's, which its class may override: an instance's.
        return self.fs._strip_protocol(path)

    def unstrip_protocol(self, name: str) -> str:
        return self.fs.unstrip_protocol(name)  # type: ignore[no-any-return]

    # The primitives.

    async def ils(self, entry: Entry, **kwargs: Any) -> AsyncIterator[Entry]:  # type: ignore[override]
        # fsspec has no listing in pages: the wrapped one's ls, all at once.
        for listed in await self.call("ls", entry["name"], detail=True, **kwargs):
            yield listed

    async def info(self, entry: Entry, **kwargs: Any) -> Entry:
        return await self.call("info", entry["name"], **kwargs)  # type: ignore[no-any-return]

    async def rm_file(self, entry: Entry) -> None:
        await self.call("rm_file", entry["name"])

    async def rmdir(self, entry: Entry) -> None:
        await self.call("rmdir", entry["name"])

    async def makedirs(self, path: StoragePath, exist_ok: bool = False) -> None:
        await self.call("makedirs", path, exist_ok=exist_ok)

    # fsspec's copy and move take paths: the entry's name alone, and no info first.

    async def cp_file(self, entry: Entry, path2: StoragePath) -> None:
        await self.call("cp_file", entry["name"], path2)

    async def mv_file(self, entry: Entry, path2: StoragePath) -> None:
        if self.asynchronous:
            await self.fs._mv_file(entry["name"], path2)
        else:
            # A sync filesystem's own move, e.g. a local rename, is its mv.
            await self._run_in_thread(self.fs.mv, entry["name"], path2)

    # The data methods and the files, the wrapped filesystem's as they are.

    async def cat_file(
        self,
        entry: Entry,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        return await self.call(  # type: ignore[no-any-return]
            "cat_file", entry["name"], start=start, end=end, **kwargs
        )

    async def pipe_file(self, path: StoragePath, value: bytes, **kwargs: Any) -> None:
        await self.call("pipe_file", path, value, **kwargs)

    async def get_file(
        self, entry: Entry, lpath: str, callback: Callback, **kwargs: Any
    ) -> None:
        await self.call("get_file", entry["name"], lpath, callback=callback, **kwargs)

    async def put_file(
        self, lpath: str, path: StoragePath, callback: Callback, **kwargs: Any
    ) -> None:
        await self.call("put_file", lpath, path, callback=callback, **kwargs)

    async def open_async_file(self, path: str, mode: str, **kwargs: Any) -> Any:
        if self.asynchronous:
            return await self.fs.open_async(path, mode, **kwargs)
        file = await self._run_in_thread(self.fs.open, path, mode, **kwargs)
        return SyncFileAdapter(file)

    def open_sync_file(self, path: str, mode: str, **kwargs: Any) -> Any:
        return self.fs._open(path, mode=mode, **kwargs)

    # The strategies' files, e.g. a copy's between two storages: the wrapped
    # one's own, awaited. They close as the wrapped one's do: one left by an
    # exception may keep what was written, so the profile declares no
    # atomic_writes unless the library's writer aborts.

    async def open_file(self, entry: Entry, mode: str = "rb", **kwargs: Any) -> Any:
        return await self.open_async_file(entry["name"], mode, **kwargs)

    # The sync methods fsspec leaves to each filesystem, the wrapped one's.

    def created(self, path: str) -> Any:
        return self.fs.created(path)

    def modified(self, path: str) -> Any:
        return self.fs.modified(path)

    async def sign(self, entry: Entry, expiration: int = 100, **kwargs: Any) -> str:
        # fsspec's sign is sync, also of an async filesystem: in a thread, as
        # its sync() refuses the loop running this.
        sign = self.fs.sign
        name = entry["name"]
        return await self._run_in_thread(sign, name, expiration=expiration, **kwargs)  # type: ignore[no-any-return]

    def invalidate_cache(self, path: str | None = None) -> None:
        self.fs.invalidate_cache(path)
