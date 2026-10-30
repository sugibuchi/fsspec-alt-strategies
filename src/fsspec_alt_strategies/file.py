# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""File objects whose buffering adapts to how a file is read or written.

fsspec's buffered file reads through a cache whose type and block size the
developer picks, such as "readahead", "bytes" or "block". No one choice suits
both small random reads, such as a Parquet footer's, and bulk sequential reads.
And a file is opened long before its reads show which kind it gets.

The classes here adapt instead, from the reads and writes so far:

- `AbstractAsyncReadableFile` fetches small blocks for scattered reads. The
  longer a run of sequential reads, the bigger its blocks, and the further
  ahead it fetches them, several at a time.
- `AbstractAsyncWritableFile` uploads small files in one request, and larger
  ones in parts that grow with the bytes written, several at a time.

Both run their requests on an asyncio event loop, such as a filesystem's
``loop``. A filesystem subclasses them, implements their requests, and returns
them from ``_open``.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
import io
import math
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsspec.asyn import sync

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Self

__all__ = [
    "AbstractAsyncFile",
    "AbstractAsyncReadableFile",
    "AbstractAsyncWritableFile",
]

_MiB = 1024 * 1024


class AbstractAsyncFile(io.RawIOBase):
    """A file whose requests run on an asyncio event loop, several at a time.

    Each sync method, such as ``read``, runs its async counterpart, such as
    ``_read``, on the loop and waits for it, as fsspec's filesystems do. So the
    sync methods cannot be called on the loop itself: code there awaits the
    async ones instead.

    Parameters
    ----------
    path : str
        The file's path, for messages.
    loop : asyncio.AbstractEventLoop
        The loop the requests run on, e.g. the filesystem's ``loop``.
    max_concurrency : int, optional
        The most requests at a time. ``default_max_concurrency`` if None.

    Raises
    ------
    ValueError
        If ``max_concurrency`` is less than 1.
    """

    default_max_concurrency: int = 10
    """The most requests at a time, unless the constructor is given another."""

    def __init__(
        self,
        path: str,
        loop: asyncio.AbstractEventLoop,
        *,
        max_concurrency: int | None = None,
    ) -> None:
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError(f"max_concurrency must be at least 1: {max_concurrency}")
        super().__init__()
        self.path = path
        self.loop = loop
        self.max_concurrency = max_concurrency or self.default_max_concurrency
        self._requests = _Requests(self.max_concurrency)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.path!r}>"

    def _sync(self, func: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        return sync(self.loop, func, *args)

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError(f"I/O operation on closed file: {self.path}")

    async def _close(self) -> None:
        """Finish the file's requests: what a subclass adds to closing."""
        await self._requests.cancel_all()

    async def aclose(self) -> None:
        """Close the file, finishing or cancelling its requests."""
        if self.closed:
            return
        try:
            await self._close()
        finally:
            super().close()

    def close(self) -> None:
        if self.closed:
            return
        if getattr(self, "loop", None) is None:
            # The constructor failed before the loop was set: no request was sent.
            super().close()
            return
        self._sync(self.aclose)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


@dataclass(eq=False)
class _Range:
    """A request for the byte range [start, end) of a file."""

    start: int
    end: int
    task: asyncio.Task[Any]

    def succeeded(self) -> bool:
        return (
            self.task.done()
            and not self.task.cancelled()
            and self.task.exception() is None
        )


class _Requests:
    """A file's requests, in the order sent, at most ``limit`` running at a time."""

    def __init__(self, limit: int) -> None:
        self.slots = asyncio.Semaphore(limit)
        self.queue: deque[_Range] = deque()
        # The first request that failed, so that writes stop at once.
        self.error: BaseException | None = None

    async def send(
        self, start: int, end: int, func: Callable[..., Awaitable[Any]], *args: Any
    ) -> None:
        """Send a request once a slot is free: the caller waits for it."""
        await self.slots.acquire()
        try:
            task = asyncio.ensure_future(func(*args))
        except BaseException:
            self.slots.release()
            raise
        task.add_done_callback(self._done)
        self.queue.append(_Range(start, end, task))

    def _done(self, task: asyncio.Task[Any]) -> None:
        self.slots.release()
        if not task.cancelled() and self.error is None:
            self.error = task.exception()

    async def cancel(self, ranges: list[_Range]) -> None:
        for r in ranges:
            r.task.cancel()
        await asyncio.gather(*(r.task for r in ranges), return_exceptions=True)

    async def cancel_all(self) -> None:
        ranges = list(self.queue)
        self.queue.clear()
        await self.cancel(ranges)


def _block_size(target: int, smallest: int, largest: int, concurrency: int) -> int:
    """The block size for ``target`` bytes, in ``concurrency`` requests at most.

    It is ``smallest`` times a power of 2, so that blocks stay aligned, and at
    most ``largest``.
    """
    needed = math.ceil(target / concurrency)
    if needed <= smallest:
        return smallest
    doublings = (math.ceil(needed / smallest) - 1).bit_length()
    return min(smallest << doublings, largest)


class AbstractAsyncReadableFile(AbstractAsyncFile, ABC):
    """A file read through blocks fetched as its reads show how it is read.

    It assumes a storage where a request's latency outweighs the bytes it
    returns, up to a block, as with object stores. Each fetch is one
    ``_fetch_range`` request, which a subclass implements. Requests cost too,
    so it sends as few as the reads allow:

    - A read fetches what it asks for, in one request up to ``max_block_size``,
      and in even parts fetched together beyond.
    - A read smaller than ``min_block_size`` fetches the block of that size
      holding it, aligned to a multiple of it. So reads near each other share
      it, e.g. a Parquet file's footer and its metadata.
    - A read that starts where the last one ended continues a run. A run that
      has read n bytes is assumed to read n more: they are fetched ahead,
      topped up once half of them are read. Their blocks double in size up to
      ``max_block_size``, and up to ``max_concurrency`` are fetched at a time.
    - A seek within the blocks fetched keeps them and the run. A seek elsewhere
      cancels the blocks in flight and ends the run, so that scattered reads,
      such as a Parquet file's column chunks, fetch nothing ahead.
    - Fetched blocks are kept, up to ``kept_size`` bytes, for later reads that
      come back to them.

    It wins over fsspec's caches when the reads are not known in advance, or
    mix both kinds. It loses on storages whose requests are cheap, where
    fetching a block for a few bytes reads more than needed.

    Parameters
    ----------
    path : str
        The file's path, for messages.
    loop : asyncio.AbstractEventLoop
        The loop the fetches run on, e.g. the filesystem's ``loop``.
    size : int
        The file's size, e.g. from the filesystem's ``_info``.
    max_concurrency : int, optional
        The most fetches at a time. ``default_max_concurrency`` if None.
    min_block_size, max_block_size : int, optional
        The smallest and largest request. The class attributes if None.
    kept_size : int, optional
        The most bytes of fetched blocks kept. The class attribute if None.

    See Also
    --------
    AbstractAsyncWritableFile : Its counterpart for writing.

    Examples
    --------
    >>> class MyFile(AbstractAsyncReadableFile):
    ...     def __init__(self, fs, path, size):
    ...         super().__init__(path, fs.loop, size)
    ...         self.fs = fs
    ...
    ...     async def _fetch_range(self, start, end):
    ...         return await self.fs._cat_file(self.path, start=start, end=end)
    """

    min_block_size: int = _MiB // 2
    """The smallest request, which smaller reads fetch whole."""

    max_block_size: int = 8 * _MiB
    """The largest request. Larger reads are split, to fetch the parts together."""

    kept_size: int = 8 * _MiB
    """The most bytes of fetched blocks kept for reads that come back to them."""

    def __init__(
        self,
        path: str,
        loop: asyncio.AbstractEventLoop,
        size: int,
        *,
        max_concurrency: int | None = None,
        min_block_size: int | None = None,
        max_block_size: int | None = None,
        kept_size: int | None = None,
    ) -> None:
        super().__init__(path, loop, max_concurrency=max_concurrency)
        self.size = size
        self.min_block_size = min_block_size or self.min_block_size
        self.max_block_size = max(
            max_block_size or self.max_block_size, self.min_block_size
        )
        self.kept_size = self.kept_size if kept_size is None else kept_size
        self._pos = 0
        # The bytes read in sequence since the last seek away.
        self._run = 0
        # Blocks no longer ahead of the position, by start, the least recently used first.
        self._kept: OrderedDict[int, tuple[int, bytes]] = OrderedDict()
        self._kept_bytes = 0

    @abstractmethod
    async def _fetch_range(self, start: int, end: int) -> bytes:
        """Fetch the bytes [start, end) of the file in one request."""

    def readable(self) -> bool:
        self._check_open()
        return True

    def seekable(self) -> bool:
        self._check_open()
        return True

    def tell(self) -> int:
        self._check_open()
        return self._pos

    def read(self, size: int | None = -1) -> bytes:
        self._check_open()
        return self._sync(self._read, -1 if size is None else size)

    def readall(self) -> bytes:
        return self.read(-1)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(memoryview(buffer)))
        memoryview(buffer).cast("B")[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._check_open()
        return self._sync(self._seek, offset, whence)

    async def _read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes, or to the end if negative, as ``read`` does."""
        self._check_open()
        pos = self._pos
        if size == 0 or pos >= self.size:
            return b""
        end = self.size if size < 0 else min(pos + size, self.size)
        if not self._requests.queue and (data := self._from_kept(pos, end)) is not None:
            self._advance(end)
            return data
        await self._fetch(pos, end)
        try:
            data = await self._collect(pos, end)
        except Exception:
            # The next read fetches again, rather than meeting the same failure.
            await self._drop_all()
            raise
        self._advance(end)
        return data

    async def _seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move to a position, as ``seek`` does."""
        self._check_open()
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if pos < 0:
            raise ValueError(f"Negative seek position: {pos}")
        queue = self._requests.queue
        if pos == self._pos:
            return pos
        if queue and queue[0].start <= pos <= queue[-1].end:
            # Many readers seek before each read, even in sequence: within the
            # blocks fetched, or right after them, the run goes on.
            behind = []
            while queue[0].end <= pos:
                behind.append(queue.popleft())
            await self._drop(behind)
        else:
            await self._drop_all()
        self._pos = pos
        return pos

    def _advance(self, end: int) -> None:
        self._run += end - self._pos
        self._pos = end

    async def _fetch(self, pos: int, end: int) -> None:
        """Send the fetches for [pos, end), and those ahead of it a run calls for."""
        queue = self._requests.queue
        continuing = self._run > 0
        if not queue and end - pos < self.min_block_size:
            # Aligned, so that reads near each other share the block, and the
            # kept ones serve later reads.
            start = pos - pos % self.min_block_size
            await self._send(start, max(end, start + self.min_block_size))
        elif (start := queue[-1].end if queue else pos) < end:
            await self._send(start, end)
        if not continuing:
            return
        # A run that has read n bytes is assumed to read n more. Topping them up
        # only once half are read keeps the requests few and large.
        ahead = min(self._run + end - pos, self.max_block_size * self.max_concurrency)
        start = queue[-1].end
        if start - end < ahead // 2:
            block = _block_size(
                ahead, self.min_block_size, self.max_block_size, self.max_concurrency
            )
            await self._send(start, end + ahead, block)

    async def _send(self, start: int, stop: int, largest: int | None = None) -> None:
        """Fetch [start, stop), within the file, in even parts of at most ``largest``."""
        stop = min(stop, self.size)
        if stop <= start:
            return
        parts = math.ceil((stop - start) / (largest or self.max_block_size))
        part = math.ceil((stop - start) / parts)
        for s in range(start, stop, part):
            e = min(s + part, stop)
            await self._requests.send(s, e, self._fetch_range, s, e)

    async def _collect(self, pos: int, end: int) -> bytes:
        """The bytes [pos, end) from the blocks fetched, which must cover them."""
        queue = self._requests.queue
        data = bytearray()
        used = 0
        # The queue changes only once all is collected, so that a cancelled read
        # leaves it as it was.
        for r in list(queue):
            if pos >= end:
                break
            chunk = await r.task
            data += memoryview(chunk)[pos - r.start : end - r.start]
            pos = min(r.end, end)
            used += r.end <= end
        for _ in range(used):
            self._keep(queue.popleft())
        return bytes(data)

    async def _drop_all(self) -> None:
        """Drop every block ahead, and end the run."""
        ranges = list(self._requests.queue)
        self._requests.queue.clear()
        self._run = 0
        await self._drop(ranges)

    async def _drop(self, ranges: list[_Range]) -> None:
        """Keep the blocks fetched among these, and cancel the others."""
        for r in ranges:
            self._keep(r)
        await self._requests.cancel([r for r in ranges if not r.task.done()])

    def _keep(self, r: _Range) -> None:
        if not r.succeeded() or r.end - r.start > self.kept_size:
            return
        if (old := self._kept.pop(r.start, None)) is not None:
            self._kept_bytes -= len(old[1])
        data = r.task.result()
        self._kept[r.start] = r.end, data
        self._kept_bytes += len(data)
        while self._kept_bytes > self.kept_size:
            _, (_, evicted) = self._kept.popitem(last=False)
            self._kept_bytes -= len(evicted)

    def _from_kept(self, pos: int, end: int) -> bytes | None:
        """The bytes [pos, end) from one kept block, if one holds them all."""
        for start, (stop, data) in self._kept.items():
            if start <= pos and end <= stop:
                self._kept.move_to_end(start)
                return data[pos - start : end - start]
        return None


class AbstractAsyncWritableFile(AbstractAsyncFile, ABC):
    """A file written in one upload, or in parts sized as its writes show.

    A subclass implements the storage's requests: one upload for small files,
    and a multipart upload's start, parts, completion and abort.

    - Until ``min_multipart_size`` bytes are written, they are buffered. A file
      closed by then is uploaded in one request.
    - Beyond, parts are uploaded as soon as they fill, up to
      ``max_concurrency`` at a time, while writing goes on. A file that has had
      n bytes written is assumed to get n more: its parts grow, so as to use the
      slots, from ``min_block_size`` up to ``max_block_size``.
    - Every part but the last has at least ``min_block_size`` bytes, as storages
      such as S3 require. So ``flush`` uploads nothing.
    - A part that fails stops the writes, and closing aborts the upload. So does
      any failure or cancellation while closing: the file is written whole, or
      not at all.

    Parameters
    ----------
    path : str
        The file's path, for messages.
    loop : asyncio.AbstractEventLoop
        The loop the uploads run on, e.g. the filesystem's ``loop``.
    max_concurrency : int, optional
        The most part uploads at a time. ``default_max_concurrency`` if None.
    min_block_size, max_block_size : int, optional
        The smallest and largest part. The class attributes if None.
    min_multipart_size : int, optional
        The size from which a multipart upload is used. The class attribute if
        None.

    See Also
    --------
    AbstractAsyncReadableFile : Its counterpart for reading.
    """

    min_block_size: int = 5 * _MiB
    """The smallest part but the last, the least S3 accepts."""

    max_block_size: int = 16 * _MiB
    """The largest part, but the last may have up to ``min_block_size`` more."""

    min_multipart_size: int = 5 * _MiB
    """The size from which a file is uploaded in parts."""

    def __init__(
        self,
        path: str,
        loop: asyncio.AbstractEventLoop,
        *,
        max_concurrency: int | None = None,
        min_block_size: int | None = None,
        max_block_size: int | None = None,
        min_multipart_size: int | None = None,
    ) -> None:
        super().__init__(path, loop, max_concurrency=max_concurrency)
        self.min_block_size = min_block_size or self.min_block_size
        self.max_block_size = max(
            max_block_size or self.max_block_size, self.min_block_size
        )
        self.min_multipart_size = min_multipart_size or self.min_multipart_size
        self._pos = 0
        self._buffer = bytearray()
        self._multipart = False

    @abstractmethod
    async def _upload_all(self, data: bytes) -> None:
        """Upload the whole file in one request."""

    @abstractmethod
    async def _start_multipart_upload(self) -> None:
        """Start a multipart upload of the file."""

    @abstractmethod
    async def _upload_part(
        self, data: bytes, start: int, end: int, index: int, last: bool
    ) -> Any:
        """Upload the part [start, end) of the file, the ``index``th from 0.

        Parts may be uploaded out of order. What this returns, such as an ETag,
        is passed on to ``_complete_multipart_upload``.
        """

    @abstractmethod
    async def _complete_multipart_upload(self, parts: list[Any]) -> None:
        """Commit the parts uploaded, given what each upload returned, in order."""

    @abstractmethod
    async def _abort_multipart_upload(self) -> None:
        """Discard the parts uploaded, leaving no file."""

    def writable(self) -> bool:
        self._check_open()
        return True

    def tell(self) -> int:
        self._check_open()
        return self._pos

    def write(self, data: Any) -> int:
        self._check_open()
        return self._sync(self._write, data)

    def flush(self) -> None:
        # A part smaller than min_block_size can only be the last: parts are
        # uploaded as they fill, and the rest on closing.
        self._check_open()

    async def _write(self, data: Any) -> int:
        """Write bytes, as ``write`` does."""
        self._check_open()
        self._raise_failed_upload()
        size = len(memoryview(data).cast("B"))
        self._buffer += data
        self._pos += size
        await self._upload_buffer(closing=False)
        return size

    async def _close(self) -> None:
        try:
            self._raise_failed_upload()
            await self._upload_buffer(closing=True)
            if self._multipart:
                parts = await asyncio.gather(*(r.task for r in self._requests.queue))
                await self._complete_multipart_upload(list(parts))
        except BaseException:
            if self._multipart:
                await self._requests.cancel_all()
                await self._abort_multipart_upload()
            raise
        finally:
            self._buffer = bytearray()
            await self._requests.cancel_all()

    def _raise_failed_upload(self) -> None:
        if (error := self._requests.error) is not None:
            raise OSError(f"Failed to upload a part of {self.path}") from error

    async def _upload_buffer(self, closing: bool) -> None:
        """Upload the parts the buffer fills, or all of it when ``closing``."""
        buffer = self._buffer
        if not self._multipart and len(buffer) < self.min_multipart_size:
            if closing:
                # Even an empty file is created.
                await self._upload_all(bytes(buffer))
                buffer.clear()
            return
        block = _block_size(
            len(buffer) if closing else self._pos,
            self.min_block_size,
            self.max_block_size,
            self.max_concurrency,
        )
        offset = self._pos - len(buffer)
        taken = 0
        while taken < len(buffer):
            remaining = len(buffer) - taken
            if remaining - block >= self.min_block_size:
                size = block
            elif closing:
                # The last part takes the rest, which would be too small alone.
                size = remaining
            else:
                break
            if not self._multipart:
                await self._start_multipart_upload()
                self._multipart = True
            start = offset + taken
            await self._requests.send(
                start,
                start + size,
                self._upload_part,
                bytes(buffer[taken : taken + size]),
                start,
                start + size,
                len(self._requests.queue),
                closing and size == remaining,
            )
            taken += size
        del buffer[:taken]
