# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""File objects written in a transaction, visible only once committed.

fsspec opens a file with ``autocommit=False`` inside ``with fs.transaction:``.
When the block ends, it calls the file's ``commit()``, or its ``discard()`` if
the block raised. How a file stays invisible until then depends on the
storage. So the file objects stay plain files, and a `FileTransaction` holds
the strategy, for one file:

- `TempPathTransaction` writes to a temporary path beside the file, renamed
  on commit. It suits any storage, best one renaming in one request. Other
  clients may see the temporary file.
- `LocalBufferTransaction` writes to a local buffer, written to the file on
  commit. It suits any storage, holding the file's bytes locally until then.

A storage with a way of its own gets a subclass of `FileTransaction`: e.g. one
uploading the parts of a multipart upload, but completing it only on commit.
`TransactionFile` is the file fsspec gets, for any of them. A filesystem's
``_open`` returns it when ``autocommit`` is False:

>>> def _open(self, path, mode="rb", autocommit=True, **kwargs):
...     if autocommit:
...         return MyFile(self, path, mode)
...     opener = lambda p: MyFile(self, p, mode)
...     return TransactionFile(LocalBufferTransaction(path, opener))

A `TransactionFile` is a file: text and compression wrappers work on it. Its
methods are sync, as fsspec's transactions are: call them from a thread other
than the filesystem's loop, as fsspec's sync API. fsspec commits a
transaction's files one after another: a transaction is not atomic across
files.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from fsspec.asyn import sync

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fsspec.asyn import AsyncFileSystem

__all__ = [
    "FileTransaction",
    "LocalBufferTransaction",
    "TempPathTransaction",
    "TransactionFile",
]

_MiB = 1024 * 1024


class FileTransaction(ABC):
    """How one file's writes stay invisible until commit.

    A `TransactionFile` calls its steps, each once and in order: `stage` when
    opened, then `commit` once the staged file is closed, or `discard` at any
    time instead. Subclass it for a storage with a way of its own.

    Parameters
    ----------
    path : str
        The file's path.

    Examples
    --------
    A storage whose uploads stay invisible until a commit request:

    >>> class NativeTransaction(FileTransaction):
    ...     def __init__(self, fs, path):
    ...         super().__init__(path)
    ...         self.fs = fs
    ...
    ...     def stage(self):
    ...         self.upload = self.fs.start_upload(self.path)
    ...         return self.upload.file
    ...
    ...     def commit(self):
    ...         self.upload.commit()
    ...
    ...     def discard(self):
    ...         self.upload.abort()
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.path!r}>"

    @abstractmethod
    def stage(self) -> io.IOBase:
        """Open the file the reads and writes go to.

        Closing it finishes the writing, but must make nothing visible.
        """

    @abstractmethod
    def commit(self) -> None:
        """Make the writing visible, once the staged file is closed."""

    @abstractmethod
    def discard(self) -> None:
        """Lose the writing, whether the staged file is closed or not."""


class _Proxy(io.RawIOBase):
    """A file passing reads and writes on to another, which closing keeps open."""

    def __init__(self, file: io.IOBase, path: str) -> None:
        super().__init__()
        self._file: Any = file
        self.path = path

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.path!r}>"

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError(f"I/O operation on closed file: {self.path}")

    def readable(self) -> bool:
        self._check_open()
        return self._file.readable()

    def writable(self) -> bool:
        self._check_open()
        return self._file.writable()

    def seekable(self) -> bool:
        self._check_open()
        return self._file.seekable()

    def read(self, size: int | None = -1) -> bytes:
        self._check_open()
        return self._file.read(-1 if size is None else size)

    def readall(self) -> bytes:
        return self.read(-1)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(memoryview(buffer)))
        memoryview(buffer).cast("B")[: len(data)] = data
        return len(data)

    def write(self, data: Any) -> int:
        self._check_open()
        return self._file.write(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._check_open()
        return self._file.seek(offset, whence)

    def tell(self) -> int:
        self._check_open()
        return self._file.tell()

    def truncate(self, size: int | None = None) -> int:
        self._check_open()
        return self._file.truncate(size)

    def flush(self) -> None:
        # io.IOBase.close flushes after a subclass's close closed the file.
        if not self.closed and not self._file.closed:
            self._file.flush()


class TransactionFile(_Proxy):
    """A file whose writes become visible on `commit`, or are lost on `discard`.

    Reads and writes go to the file its transaction staged. Closing closes
    that, which finishes the writing but makes nothing visible: `commit` does,
    closing first if needed. Leaving a ``with`` block by an exception
    discards.

    Parameters
    ----------
    transaction : FileTransaction
        The strategy keeping the writes invisible. Its ``stage`` is called at
        once.

    Examples
    --------
    >>> f = TransactionFile(LocalBufferTransaction(path, open_file))
    """

    def __init__(self, transaction: FileTransaction) -> None:
        self.transaction = transaction
        super().__init__(transaction.stage(), transaction.path)
        # None while undecided, then "committed" or "discarded".
        self._outcome: str | None = None
        # Whether closing finished the writing: a failed one is never committed.
        self._finished = False

    def close(self) -> None:
        """Finish the writing. It stays invisible until `commit`."""
        if self.closed:
            return
        try:
            if self._outcome is None:
                self._file.close()
                self._finished = True
        finally:
            super().close()

    def commit(self) -> None:
        """Make the writing visible, closing the file first if needed."""
        if self._outcome == "committed":
            return
        if self._outcome == "discarded":
            raise ValueError(f"Committing a discarded file: {self.path}")
        self.close()
        if not self._finished:
            raise ValueError(f"Committing a file whose writing failed: {self.path}")
        self._outcome = "committed"
        self.transaction.commit()

    def discard(self) -> None:
        """Lose the writing, whether the file is closed or not."""
        if self._outcome == "discarded":
            return
        if self._outcome == "committed":
            raise ValueError(f"Discarding a committed file: {self.path}")
        self._outcome = "discarded"
        try:
            self.transaction.discard()
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


class TempPathTransaction(FileTransaction):
    """A file written to a temporary path, renamed to its own on commit.

    The temporary path is in the file's folder, hidden by a leading ".", so
    that a rename stays within one folder, container or bucket. Staging opens
    a file there, and closing it writes it in full. Commit renames it, and
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
    >>> t = TempPathTransaction(fs, path, lambda p: MyFile(fs, p, "wb"))
    >>> f = TransactionFile(t)
    """

    def __init__(
        self,
        fs: AsyncFileSystem,
        path: str,
        open_file: Callable[[str], io.IOBase],
        *,
        rename: Callable[[str, str], Awaitable[Any]] | None = None,
        temp_path: str | None = None,
    ) -> None:
        super().__init__(path)
        self.fs = fs
        self.open_file = open_file
        self.temp_path = temp_path or _temp_path(path)
        self.rename = rename or self._copy_then_delete
        self._file: io.IOBase | None = None

    async def _copy_then_delete(self, source: str, destination: str) -> None:
        await self.fs._cp_file(source, destination)
        await self.fs._rm_file(source)

    async def _delete_temp(self) -> None:
        # Never written, e.g. if discarded before the file uploaded.
        with contextlib.suppress(FileNotFoundError):
            await self.fs._rm_file(self.temp_path)

    def stage(self) -> io.IOBase:
        self._file = self.open_file(self.temp_path)
        return self._file

    def commit(self) -> None:
        sync(self.fs.loop, self.rename, self.temp_path, self.path)

    def discard(self) -> None:
        try:
            if self._file is not None:
                self._file.close()
        finally:
            sync(self.fs.loop, self._delete_temp)


class LocalBufferTransaction(FileTransaction):
    """A file written to a local buffer, written to its own on commit.

    The buffer is a `tempfile.SpooledTemporaryFile`: in memory up to
    ``spool_size`` bytes, then on disk. Closing the staged file keeps it.
    Nothing is sent before commit, which opens the file and writes the buffer
    into it, from start to end. Discard drops the buffer.

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
    >>> t = LocalBufferTransaction(path, lambda p: MyFile(fs, p, "wb"))
    >>> f = TransactionFile(t)
    """

    def __init__(
        self,
        path: str,
        open_file: Callable[[str], io.IOBase],
        *,
        spool_size: int = 8 * _MiB,
    ) -> None:
        super().__init__(path)
        self.open_file = open_file
        self.spool_size = spool_size
        self._buffer: Any = None

    def stage(self) -> io.IOBase:
        self._buffer = tempfile.SpooledTemporaryFile(max_size=self.spool_size)  # noqa: SIM115
        # Closing the proxy keeps the buffer for commit.
        return _Proxy(self._buffer, self.path)

    def commit(self) -> None:
        try:
            self._buffer.seek(0)
            with self.open_file(self.path) as f:
                shutil.copyfileobj(self._buffer, f, _MiB)
        finally:
            self._buffer.close()

    def discard(self) -> None:
        if self._buffer is not None:
            self._buffer.close()
