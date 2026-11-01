# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The file objects of this package, used from async code on their loop.

A file object's sync methods, such as ``read``, wait on the filesystem's loop.
Called on that loop, they would wait on themselves. Async code there awaits
the methods of an `AsyncFileAdapter` instead, as fsspec's ``open_async``
returns:

>>> async def open_async(self, path, mode="rb", **kwargs):
...     return AsyncFileAdapter(MyFile(self, path, mode))

The adapter only awaits what the file implements already: the file objects
stay files, with Python's sync API.
"""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Self

    from ._async_file import _AsyncFile

__all__ = ["AsyncFileAdapter"]


class AsyncFileAdapter:
    """A file object of this package, with methods to await on its loop.

    Each method awaits the file's async counterpart of the sync one, e.g.
    ``read`` awaits what the file's ``read`` waits for. As in fsspec's async
    files, ``read``, ``write``, ``flush`` and ``close`` are awaited. So are
    ``seek``, ``tell`` and ``truncate``, unlike in fsspec's: they may send a
    request, e.g. a `~fsspec_alt_strategies.files.buffered.LocallyBufferedFile`
    downloading first. ``readable``, ``writable`` and ``seekable`` send none.

    Leaving an ``async with`` block closes the file as leaving a ``with``
    block does, e.g. discarding a locally buffered file's changes on an
    exception. Close it: an open file closed by the garbage collector waits
    on the loop, which may be the running one.

    Parameters
    ----------
    file : AdaptiveReadableFile, AdaptiveWritableFile or LocallyBufferedFile
        The file. Its methods run on its ``loop``: the adapter's raise
        RuntimeError on another one.

    Examples
    --------
    >>> async with AsyncFileAdapter(MyFile(fs, path, "rb")) as f:
    ...     header = await f.read(8)
    ...     await f.seek(-8, io.SEEK_END)
    ...     footer = await f.read()
    """

    def __init__(self, file: _AsyncFile) -> None:
        self.file = file

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.file!r}>"

    @property
    def path(self) -> str:
        return self.file.path

    @property
    def closed(self) -> bool:
        return self.file.closed

    def _on_loop(self) -> _AsyncFile:
        """The file, once checked that this runs on its loop."""
        if asyncio.get_running_loop() is not self.file.loop:
            raise RuntimeError(f"Awaited outside the loop of {self.path}")
        return self.file

    def readable(self) -> bool:
        return self.file.readable()

    def writable(self) -> bool:
        return self.file.writable()

    def seekable(self) -> bool:
        return self.file.seekable()

    async def read(self, size: int | None = -1) -> bytes:
        return await self._on_loop()._read(-1 if size is None else size)

    async def readall(self) -> bytes:
        return await self.read(-1)

    async def write(self, data: Any) -> int:
        return await self._on_loop()._write(data)

    async def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return await self._on_loop()._seek(offset, whence)

    async def tell(self) -> int:
        return await self._on_loop()._tell()

    async def truncate(self, size: int | None = None) -> int:
        return await self._on_loop()._truncate(size)

    async def flush(self) -> None:
        # The files' flush sends no request: parts are uploaded as they fill.
        self._on_loop().flush()

    async def close(self) -> None:
        await self._on_loop().aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._on_loop().__aexit__(*exc_info)
