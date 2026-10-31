# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""File objects written in a transaction, visible only once committed.

fsspec opens a file with ``autocommit=False`` inside ``with fs.transaction:``.
When the block ends, it calls the file's ``commit()``, or its ``discard()`` if
the block raised. How a file stays invisible until then depends on the
storage. So the file objects stay plain files, and each class here wraps one
in a strategy for it:

- `TempPathTransaction` writes to a temporary path beside the file, renamed
  on commit. It suits any storage, best one renaming in one request. Other
  clients may see the temporary file.
- `LocalBufferTransaction` writes to a local buffer, written to the file on
  commit. It suits any storage, holding the file's bytes locally until then.
- `StagedTransaction` has the file upload all but what makes it visible,
  e.g. the parts of a multipart upload but not its completion. It costs
  nothing more, but needs a file that stages, as the file objects of
  `files.adaptive` and `files.buffered` do.

A filesystem's ``_open`` returns the wrapper when ``autocommit`` is False:

>>> def _open(self, path, mode="rb", autocommit=True, **kwargs):
...     f = MyFile(self, path, mode)
...     return f if autocommit else StagedTransaction(f)

Each wrapper is a file: text and compression wrappers work on it. Its methods
are sync, as fsspec's transactions are: call them from a thread other than the
filesystem's loop, as fsspec's sync API. fsspec commits a transaction's files
one after another: a transaction is not atomic across files.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from typing import IO, TYPE_CHECKING, Any, Protocol

from fsspec.asyn import sync

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable

    from fsspec.asyn import AsyncFileSystem

__all__ = [
    "LocalBufferTransaction",
    "StagedFile",
    "StagedTransaction",
    "TempPathTransaction",
]

_MiB = 1024 * 1024


class _TransactionFile(io.RawIOBase, ABC):
    """A file whose writes become visible on `commit`, or are lost on `discard`.

    Reads and writes go to the file `_target` returns. Closing finishes the
    writing, but makes nothing visible: `commit` does, closing first if
    needed. Leaving a ``with`` block by an exception discards.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        # None while undecided, then "committed" or "discarded".
        self._outcome: str | None = None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.path!r}>"

    @abstractmethod
    def _target(self) -> IO[bytes]:
        """The file the reads and writes go to."""

    @abstractmethod
    def _finish(self) -> None:
        """Finish the writing, making nothing visible."""

    @abstractmethod
    def _publish(self) -> None:
        """Make the finished writing visible."""

    @abstractmethod
    def _drop(self) -> None:
        """Lose the writing, finished or not."""

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError(f"I/O operation on closed file: {self.path}")

    def readable(self) -> bool:
        self._check_open()
        return self._target().readable()

    def writable(self) -> bool:
        self._check_open()
        return self._target().writable()

    def seekable(self) -> bool:
        self._check_open()
        return self._target().seekable()

    def read(self, size: int | None = -1) -> bytes:
        self._check_open()
        return self._target().read(-1 if size is None else size)

    def readall(self) -> bytes:
        return self.read(-1)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(memoryview(buffer)))
        memoryview(buffer).cast("B")[: len(data)] = data
        return len(data)

    def write(self, data: Any) -> int:
        self._check_open()
        return self._target().write(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._check_open()
        return self._target().seek(offset, whence)

    def tell(self) -> int:
        self._check_open()
        return self._target().tell()

    def truncate(self, size: int | None = None) -> int:
        self._check_open()
        return self._target().truncate(size)

    def flush(self) -> None:
        # io.IOBase.close flushes after _finish or _drop closed the target.
        if not self.closed and not (target := self._target()).closed:
            target.flush()

    def close(self) -> None:
        """Finish the writing. It stays invisible until `commit`."""
        if self.closed:
            return
        try:
            if self._outcome is None:
                self._finish()
        finally:
            super().close()

    def commit(self) -> None:
        """Make the writing visible, closing the file first if needed."""
        if self._outcome == "committed":
            return
        if self._outcome == "discarded":
            raise ValueError(f"Committing a discarded file: {self.path}")
        self.close()
        self._outcome = "committed"
        self._publish()

    def discard(self) -> None:
        """Lose the writing, whether the file is closed or not."""
        if self._outcome == "discarded":
            return
        if self._outcome == "committed":
            raise ValueError(f"Discarding a committed file: {self.path}")
        self._outcome = "discarded"
        try:
            self._drop()
        finally:
            super().close()

    def __exit__(self, exc_type: object, *exc_info: object) -> None:
        if exc_type is not None:
            self.discard()
        else:
            self.close()


def _temp_path(path: str) -> str:
    """A hidden path beside ``path``, unique to one transaction's file."""
    parent, _, name = path.rpartition("/")
    temp = f".{name}.{uuid.uuid4().hex[:16]}.tmp"
    return f"{parent}/{temp}" if parent else temp


class TempPathTransaction(_TransactionFile):
    """A file written to a temporary path, renamed to its own on commit.

    The temporary path is in the file's folder, hidden by a leading ".", so
    that a rename stays within one folder, container or bucket. Closing closes
    the file opened there, which writes it in full. Commit renames it, and
    discard deletes it.

    It suits any writable file, and any storage: best one renaming in one
    request, where commit is atomic. Without a rename, the default copies then
    deletes, on the server. Other clients may see the temporary file, and it
    stays if the process dies before commit or discard.

    The file writes from its start: the temporary path is new. So appending, or
    exclusive creation, do not apply.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem, whose ``loop`` runs the rename and the deletion.
    path : str
        The file's path.
    open_file : callable
        Opens a plain file for writing at the path given: the temporary one.
    rename : callable, optional
        An async function moving a file from its first path to its second,
        overwriting it, e.g. the storage's own rename. By default, the
        filesystem's ``_cp_file`` then ``_rm_file``.
    temp_path : str, optional
        The temporary path. A hidden one beside ``path`` if None.

    Examples
    --------
    >>> f = TempPathTransaction(fs, path, lambda p: MyFile(fs, p, "wb"))
    """

    def __init__(
        self,
        fs: AsyncFileSystem,
        path: str,
        open_file: Callable[[str], IO[bytes]],
        *,
        rename: Callable[[str, str], Awaitable[Any]] | None = None,
        temp_path: str | None = None,
    ) -> None:
        super().__init__(path)
        self.fs = fs
        self.temp_path = temp_path or _temp_path(path)
        self.rename = rename or self._copy_then_delete
        self._file = open_file(self.temp_path)

    async def _copy_then_delete(self, source: str, destination: str) -> None:
        await self.fs._cp_file(source, destination)
        await self.fs._rm_file(source)

    async def _delete_temp(self) -> None:
        # Never written, e.g. if discarded before the inner file uploaded.
        with contextlib.suppress(FileNotFoundError):
            await self.fs._rm_file(self.temp_path)

    def _target(self) -> IO[bytes]:
        return self._file

    def _finish(self) -> None:
        self._file.close()

    def _publish(self) -> None:
        sync(self.fs.loop, self.rename, self.temp_path, self.path)

    def _drop(self) -> None:
        try:
            self._file.close()
        finally:
            sync(self.fs.loop, self._delete_temp)


class LocalBufferTransaction(_TransactionFile):
    """A file written to a local buffer, written to its own on commit.

    The buffer is a `tempfile.SpooledTemporaryFile`: in memory up to
    ``spool_size`` bytes, then on disk. Nothing is sent before commit, which
    opens the file and writes the buffer into it, from start to end. Discard
    drops the buffer.

    It suits any writable file, and any storage, and other clients see nothing
    until commit. But the file's bytes are held locally until then, and commit
    writes them all. Reads read the buffer, not the file as it was.

    Parameters
    ----------
    path : str
        The file's path.
    open_file : callable
        Opens a plain file for writing at the path given, in the mode wanted:
        e.g. "ab" appends the buffer on commit.
    spool_size : int
        The most bytes buffered in memory, beyond which the buffer moves to
        disk.

    Examples
    --------
    >>> f = LocalBufferTransaction(path, lambda p: MyFile(fs, p, "wb"))
    """

    def __init__(
        self,
        path: str,
        open_file: Callable[[str], IO[bytes]],
        *,
        spool_size: int = 8 * _MiB,
    ) -> None:
        super().__init__(path)
        self.open_file = open_file
        self._buffer: IO[bytes] = tempfile.SpooledTemporaryFile(  # noqa: SIM115
            max_size=spool_size
        )

    def _target(self) -> IO[bytes]:
        return self._buffer

    def _finish(self) -> None:
        pass  # The buffer is kept for commit.

    def _publish(self) -> None:
        try:
            self._buffer.seek(0)
            with self.open_file(self.path) as f:
                shutil.copyfileobj(self._buffer, f, _MiB)
        finally:
            self._buffer.close()

    def _drop(self) -> None:
        self._buffer.close()


class StagedFile(Protocol):
    """A file that can upload all but what makes it visible, for a transaction.

    The file objects of `files.adaptive` and `files.buffered` are such files.
    Each step is a coroutine on the file's ``loop``.
    """

    loop: asyncio.AbstractEventLoop

    async def _stage(self) -> None:
        """Upload all but what makes the file visible, e.g. a multipart upload's parts."""

    async def _publish(self) -> None:
        """Make the staged file visible, e.g. complete the multipart upload."""

    async def _unstage(self) -> None:
        """Drop all written, e.g. abort the multipart upload."""

    def close(self) -> None:
        """Close the file, which has been published or dropped."""


class StagedTransaction(_TransactionFile):
    """A file staged on closing, and published on commit.

    The file uploads all but what makes it visible when closed: e.g. the parts
    of a multipart upload, but not its completion. Commit publishes it, and
    discard drops it: e.g. completes, or aborts, the multipart upload.

    It costs nothing more than writing the file without a transaction, and
    other clients see nothing until commit. Staged uploads take storage until
    then: if the process dies first, they stay until the storage expires them.

    Parameters
    ----------
    file : StagedFile
        The file, open for writing, e.g. an
        `~fsspec_alt_strategies.files.adaptive.AdaptiveWritableFile`.

    Examples
    --------
    >>> f = StagedTransaction(MyFile(fs, path, "wb"))
    """

    def __init__(self, file: StagedFile) -> None:
        super().__init__(getattr(file, "path", repr(file)))
        self._file = file

    def _target(self) -> IO[bytes]:
        return self._file  # type: ignore[return-value]

    def _finish(self) -> None:
        sync(self._file.loop, self._file._stage)

    def _publish(self) -> None:
        sync(self._file.loop, self._file._publish)
        self._file.close()

    def _drop(self) -> None:
        sync(self._file.loop, self._file._unstage)
        self._file.close()
