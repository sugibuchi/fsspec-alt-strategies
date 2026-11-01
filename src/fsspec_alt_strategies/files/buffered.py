# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A file object for storages transferring only whole files, worked on locally.

This module depends on `_async_file`.
"""

from __future__ import annotations

import errno
import io
import tempfile
from abc import ABC, abstractmethod
from typing import IO, TYPE_CHECKING, Any

from ._async_file import _AsyncFile, _MiB

if TYPE_CHECKING:
    import asyncio

__all__ = ["LocallyBufferedFile"]

_MODES = {"r", "w", "a", "x"}


class LocallyBufferedFile(_AsyncFile, ABC):
    """A file of a storage transferring only whole files, worked on locally.

    Some storages read or write a file only whole, in one request: no byte
    ranges, no parts. Such a file is downloaded once, read and written in a
    local buffer, and uploaded once, if it changed. A subclass implements the
    storage's requests: `_download`, `_upload` and `_exists`.

    - Nothing is downloaded until the first read, write, seek or tell, or
      `load`. A filesystem's ``_open`` may call `load`, so that a missing file
      raises there, as fsspec's files do.
    - The buffer is a `tempfile.SpooledTemporaryFile`: in memory up to
      ``spool_size`` bytes, then on disk. It is deleted on closing.
    - Closing uploads the buffer, only if it was written or truncated, or the
      file is new. Leaving a ``with`` block by an exception, or `discard`,
      uploads nothing: the file is written whole, or not at all.
    - ``max_size`` caps the file: downloading, writing or truncating beyond it
      raises OSError with ``errno.EFBIG``.
    - The download and the upload each hold the concurrency limit. Reads and
      writes after the download are local: they never wait on the loop.

    Modes are binary: "rb", "wb", "ab" and "xb", each with "+" too. Appending
    writes at the end, wherever the position is. Exclusive creation, "x", asks
    `_exists` first, and raises FileExistsError: another may still create the
    file before the upload.

    Parameters
    ----------
    path : str
        The file's path, for messages.
    loop : asyncio.AbstractEventLoop
        The loop the requests run on, e.g. the filesystem's ``loop``.
    mode : str
        The mode, as above.
    concurrency_limit : asyncio.Semaphore, optional
        Held by the download and the upload. Share one between files to limit
        their requests together. The file's own, of
        ``default_concurrency_limit``, if None.
    spool_size : int, optional
        The most bytes buffered in memory, beyond which the buffer moves to
        disk. The class attribute if None.
    max_size : int, optional
        The most bytes the file may have. The class attribute if None.

    Raises
    ------
    ValueError
        If ``mode`` is not one of those above.

    See Also
    --------
    AdaptiveReadableFile, AdaptiveWritableFile : For storages reading byte
        ranges and writing in parts.

    Examples
    --------
    >>> class MyFile(LocallyBufferedFile):
    ...     def __init__(self, fs, path, mode):
    ...         super().__init__(path, fs.loop, mode)
    ...         self.fs = fs
    ...
    ...     async def _download(self, target):
    ...         target.write(await self.fs._get_whole_file(self.path))
    ...
    ...     async def _upload(self, source):
    ...         await self.fs._put_whole_file(self.path, source.read())
    ...
    ...     async def _exists(self):
    ...         return await self.fs._exists(self.path)
    """

    spool_size: int = 8 * _MiB
    """The most bytes buffered in memory, beyond which the buffer moves to disk."""

    max_size: int | None = None
    """The most bytes a file may have. None for no limit."""

    def __init__(
        self,
        path: str,
        loop: asyncio.AbstractEventLoop,
        mode: str = "rb",
        *,
        concurrency_limit: asyncio.Semaphore | None = None,
        spool_size: int | None = None,
        max_size: int | None = None,
    ) -> None:
        super().__init__(path, loop, concurrency_limit=concurrency_limit)
        self._buffer: tempfile.SpooledTemporaryFile[bytes] | None = None
        kinds = [c for c in mode if c in _MODES]
        if (
            len(set(mode)) != len(mode)
            or not set(mode) <= {*_MODES, "b", "+"}
            or "b" not in mode
            or len(kinds) != 1
        ):
            # Closed as a raw stream: nothing was sent, nor is to upload.
            io.RawIOBase.close(self)
            raise ValueError(f"Unsupported mode, not a binary one: {mode!r}")
        self.mode = mode
        self._kind = kinds[0]
        self._plus = "+" in mode
        self.spool_size = spool_size or self.spool_size
        self.max_size = self.max_size if max_size is None else max_size
        # Whether closing uploads: a new file is uploaded even if empty.
        self._changed = self._kind in "wx"
        self._discarded = False

    @abstractmethod
    async def _download(self, target: IO[bytes]) -> None:
        """Write the whole file into ``target``, in one request.

        Raises FileNotFoundError if the file does not exist.
        """

    @abstractmethod
    async def _upload(self, source: IO[bytes]) -> None:
        """Write the whole file from ``source``, read from its start, in one request."""

    @abstractmethod
    async def _exists(self) -> bool:
        """Whether the file exists, for exclusive creation."""

    def readable(self) -> bool:
        self._check_open()
        return self._kind == "r" or self._plus

    def writable(self) -> bool:
        self._check_open()
        return self._kind != "r" or self._plus

    def seekable(self) -> bool:
        self._check_open()
        return True

    def load(self) -> None:
        """Download the file now, if not yet, so that a missing file raises now."""
        self._check_open()
        if self._buffer is None:
            self._sync(self._load)

    async def _load(self) -> IO[bytes]:
        """The buffer, downloaded first if not yet: `load`'s async counterpart."""
        self._check_open()
        if self._buffer is not None:
            return self._buffer
        buffer = tempfile.SpooledTemporaryFile(max_size=self.spool_size)  # noqa: SIM115
        try:
            if self._kind == "x":
                async with self._requests.concurrency_limit:
                    if await self._exists():
                        raise FileExistsError(self.path)
            elif self._kind in "ra":
                try:
                    async with self._requests.concurrency_limit:
                        await self._download(_Capped(buffer, self.max_size, self.path))
                except FileNotFoundError:
                    if self._kind == "r":
                        raise
                    buffer.seek(0)
                    buffer.truncate()
                    self._changed = True  # Appending creates the file.
                buffer.seek(0, io.SEEK_END if self._kind == "a" else io.SEEK_SET)
        except BaseException:
            buffer.close()
            raise
        self._buffer = buffer
        return buffer

    @property
    def size(self) -> int | None:
        """The file's size so far, once downloaded or created: None before."""
        if self._buffer is None:
            return None
        pos = self._buffer.tell()
        size = self._buffer.seek(0, io.SEEK_END)
        self._buffer.seek(pos)
        return size

    async def _prepare(self) -> None:
        # Downloaded, the position is known: seek and tell stay local.
        await self._load()

    def _local(self) -> IO[bytes]:
        self.load()
        assert self._buffer is not None
        return self._buffer

    def _check_readable(self) -> None:
        if not self.readable():
            raise io.UnsupportedOperation(f"File not open for reading: {self.path}")

    def _check_writable(self) -> None:
        if not self.writable():
            raise io.UnsupportedOperation(f"File not open for writing: {self.path}")

    def read(self, size: int | None = -1) -> bytes:
        self._check_readable()
        return self._local().read(-1 if size is None else size)

    def readall(self) -> bytes:
        return self.read(-1)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(memoryview(buffer)))
        memoryview(buffer).cast("B")[: len(data)] = data
        return len(data)

    async def _read(self, size: int = -1) -> bytes:
        """Read as ``read`` does."""
        self._check_readable()
        return (await self._load()).read(size)

    def write(self, data: Any) -> int:
        self._check_writable()
        return self._write_to(self._local(), data)

    async def _write(self, data: Any) -> int:
        """Write as ``write`` does."""
        self._check_writable()
        return self._write_to(await self._load(), data)

    def _write_to(self, buffer: IO[bytes], data: Any) -> int:
        if self._kind == "a":
            buffer.seek(0, io.SEEK_END)
        size = len(memoryview(data).cast("B"))
        _check_size(buffer.tell() + size, self.max_size, self.path)
        buffer.write(data)
        self._changed = True
        return size

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._check_open()
        return self._local().seek(offset, whence)

    async def _seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move as ``seek`` does."""
        self._check_open()
        return (await self._load()).seek(offset, whence)

    def tell(self) -> int:
        self._check_open()
        return self._local().tell()

    def truncate(self, size: int | None = None) -> int:
        self._check_writable()
        buffer = self._local()
        size = buffer.tell() if size is None else size
        _check_size(size, self.max_size, self.path)
        self._changed = True
        return buffer.truncate(size)

    def flush(self, force: bool = False) -> None:
        # The file is uploaded whole, on closing: even forced, nothing is
        # shown before.
        self._check_open()

    def discard(self) -> None:
        """Upload nothing on closing: the changes are lost."""
        self._discarded = True

    async def _close(self) -> None:
        try:
            if self._changed and not self._discarded:
                buffer = await self._load()
                buffer.seek(0)
                async with self._requests.concurrency_limit:
                    await self._upload(buffer)
        finally:
            buffer, self._buffer = self._buffer, None
            if buffer is not None:
                buffer.close()
            await super()._close()

    def __exit__(self, exc_type: object, *exc_info: object) -> None:
        if exc_type is not None:
            self.discard()
        self.close()

    async def __aexit__(self, exc_type: object, *exc_info: object) -> None:
        if exc_type is not None:
            self.discard()
        await self.aclose()


def _check_size(size: int, max_size: int | None, path: str) -> None:
    if max_size is not None and size > max_size:
        raise OSError(errno.EFBIG, f"File too large, over {max_size} bytes", path)


class _Capped:
    """A buffer's writing end that refuses to grow past a size, for downloads."""

    def __init__(self, buffer: IO[bytes], max_size: int | None, path: str) -> None:
        self.buffer = buffer
        self.max_size = max_size
        self.path = path

    def write(self, data: Any) -> int:
        _check_size(
            self.buffer.tell() + len(memoryview(data).cast("B")),
            self.max_size,
            self.path,
        )
        return self.buffer.write(data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.buffer, name)
