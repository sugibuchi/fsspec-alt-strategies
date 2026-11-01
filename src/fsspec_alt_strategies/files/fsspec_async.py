# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""fsspec's ``open_async``, on top of this package's file objects.

``AsyncFileSystem.open_async`` returns a file whose methods are awaited, with
the API of fsspec's ``AbstractAsyncStreamedFile``. An `AsyncFileAdapter`
gives that API to the file objects of `files.adaptive` and `files.buffered`,
so that a filesystem implements its requests once, for both ``open`` and
``open_async``:

>>> async def open_async(self, path, mode="rb", **kwargs):
...     return await AsyncFileAdapter.open(MyFile(self, path, mode))

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

    As in fsspec's ``AbstractAsyncStreamedFile``:

    - ``read``, ``write``, ``flush`` and ``close`` are awaited. Each awaits the
      file's async counterpart of the sync method, e.g. ``_read``.
    - ``seek``, ``tell``, ``readable``, ``writable`` and ``seekable`` are not:
      they send no request.
    - ``path``, ``mode``, ``size``, ``loc``, ``closed`` and ``info`` describe
      the file.

    Create it with `open`, which awaits what ``seek`` and ``tell`` need first,
    e.g. a locally buffered file's download.

    Unlike fsspec's, leaving an ``async with`` block by an exception discards a
    locally buffered file's changes, as leaving a ``with`` block does. Close
    it: an open file closed by the garbage collector waits on the loop, which
    may be the running one.

    Parameters
    ----------
    file : AdaptiveReadableFile, AdaptiveWritableFile or LocallyBufferedFile
        The file. Its requests run on its ``loop``: the awaited methods raise
        RuntimeError on another one.

    Examples
    --------
    >>> async def open_async(self, path, mode="rb", **kwargs):
    ...     return await AsyncFileAdapter.open(MyFile(self, path, mode))
    >>> async with await fs.open_async(path) as f:
    ...     header = await f.read(8)
    ...     f.seek(-8, io.SEEK_END)
    ...     footer = await f.read()
    """

    def __init__(self, file: _AsyncFile) -> None:
        self.file = file

    @classmethod
    async def open(cls, file: _AsyncFile) -> Self:
        """Adapt a file, once it has sent what ``seek`` and ``tell`` need.

        A locally buffered file downloads here, as fsspec's async files learn
        their size on opening. So a missing file raises here, and the file is
        closed.
        """
        adapter = cls(file)
        try:
            await adapter._on_loop()._prepare()
        except BaseException:
            await file.aclose()
            raise
        return adapter

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.file!r}>"

    @property
    def path(self) -> str:
        return self.file.path

    @property
    def mode(self) -> str:
        return self.file.mode

    @property
    def size(self) -> int | None:
        """The file's size: None if not known, e.g. while writing it."""
        return getattr(self.file, "size", None)

    @property
    def closed(self) -> bool:
        return self.file.closed

    @property
    def loc(self) -> int:
        """The position, as fsspec's files keep it."""
        return self.file.tell()

    def info(self) -> dict[str, Any]:
        """The file's details, as fsspec's ``info`` gives them, when reading."""
        if not self.file.readable():
            raise ValueError("Info not available while writing")
        return {"name": self.path, "size": self.size, "type": "file"}

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

    def seek(self, loc: int, whence: int = io.SEEK_SET) -> int:
        return self.file.seek(loc, whence)

    def tell(self) -> int:
        return self.file.tell()

    async def read(self, length: int | None = -1) -> bytes:
        return await self._on_loop()._read(-1 if length is None else length)

    async def write(self, data: Any) -> int:
        return await self._on_loop()._write(data)

    async def flush(self, force: bool = False) -> None:
        await self._on_loop()._flush(force)

    async def close(self) -> None:
        await self._on_loop().aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._on_loop().__aexit__(*exc_info)
