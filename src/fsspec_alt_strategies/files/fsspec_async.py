# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""fsspec's ``open_async``, on top of this package's file objects.

``AsyncFileSystem.open_async`` returns a file whose methods are awaited, with
the API of fsspec's ``AbstractAsyncStreamedFile``. An `AsyncFileAdapter`
gives that API to the file objects of `files.adaptive` and `files.buffered`,
so that a filesystem implements its requests once, for both ``open`` and
``open_async``:

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
    """A file object of this package, as fsspec's ``open_async`` returns it.

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

    async def flush(self, force: bool = False) -> None:
        await self._on_loop()._flush(force)

    async def close(self) -> None:
        await self._on_loop().aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._on_loop().__aexit__(*exc_info)
