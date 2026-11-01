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
- `AdaptiveWritableTransaction` has an
  `~fsspec_alt_strategies.files.adaptive.AdaptiveWritableFile` upload its
  parts, but complete the upload only on commit. It costs nothing more.
- `LocallyBufferedTransaction` has a
  `~fsspec_alt_strategies.files.buffered.LocallyBufferedFile` upload its
  buffer only on commit. It costs nothing more.

A storage with a way of its own gets a subclass of `FileTransaction`.
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

This module depends on nothing but the standard library: it imports the file
classes only for type checking.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .adaptive import AdaptiveWritableFile
    from .buffered import LocallyBufferedFile

__all__ = [
    "AdaptiveWritableTransaction",
    "FileTransaction",
    "LocalBufferTransaction",
    "LocallyBufferedTransaction",
    "TempPathTransaction",
    "TransactionFile",
]

_MiB = 1024 * 1024


class FileTransaction(ABC):
    """How one file's writes stay invisible until commit.

    A `TransactionFile` calls its steps, each once and in order:
    `start_staging` when opened, `end_staging` when closed, then `commit`. Or
    `discard`, at any time instead. Subclass it for a storage with a way of its
    own.

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
    ...     def start_staging(self):
    ...         self.upload = self.fs.start_upload(self.path)
    ...         return self.upload.file
    ...
    ...     def end_staging(self):
    ...         self.upload.file.close()
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
    def start_staging(self) -> io.IOBase:
        """Open the file the reads and writes go to."""

    @abstractmethod
    def end_staging(self) -> None:
        """Finish the writing, making nothing visible, e.g. close the staged file."""

    @abstractmethod
    def commit(self) -> None:
        """Make the finished writing visible."""

    @abstractmethod
    def discard(self) -> None:
        """Lose the writing, finished or not."""


class TransactionFile(io.RawIOBase):
    """A file whose writes become visible on `commit`, or are lost on `discard`.

    Reads and writes go to the file its transaction staged. Closing finishes
    the writing, as the transaction does, but makes nothing visible: `commit`
    does, closing first if needed. Leaving a ``with`` block by an exception
    discards.

    Parameters
    ----------
    transaction : FileTransaction
        The strategy keeping the writes invisible. Its ``start_staging`` is
        called at once.

    Examples
    --------
    >>> f = TransactionFile(LocalBufferTransaction(path, open_file))
    """

    def __init__(self, transaction: FileTransaction) -> None:
        super().__init__()
        self.transaction = transaction
        self.path = transaction.path
        self._file: Any = transaction.start_staging()
        # None while undecided, then "committed" or "discarded".
        self._outcome: str | None = None
        # Whether closing finished the writing: a failed one is never committed.
        self._finished = False

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
        # io.IOBase.close flushes after end_staging or discard closed the file.
        if not self.closed and not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        """Finish the writing. It stays invisible until `commit`."""
        if self.closed:
            return
        try:
            if self._outcome is None:
                self.transaction.end_staging()
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
    that a rename stays within one folder, container or bucket. Starting the
    staging opens a file there, and ending it closes it, which writes it in
    full. Commit renames it, and discard deletes it.

    It suits any writable file, and any storage: best one renaming in one
    request, where commit is atomic. A rename copying then deleting works too,
    at the cost of copying the file. Other clients may see the temporary file,
    and it stays if the process dies before commit or discard.

    The file writes from its start: the temporary path is new. So appending, or
    exclusive creation, do not apply.

    Parameters
    ----------
    path : str
        The file's path.
    open_file : callable
        Opens a plain file for writing at the path given: the temporary one.
    rename : callable
        Moves a file from its first path to its second, overwriting it, e.g.
        the storage's own rename.
    delete : callable
        Deletes the file at the path given, raising FileNotFoundError if there
        is none.
    temp_path : str, optional
        The temporary path. A hidden one beside ``path`` if None.

    Examples
    --------
    >>> t = TempPathTransaction(
    ...     path, lambda p: MyFile(fs, p, "wb"), fs.mv, fs.rm_file
    ... )
    >>> f = TransactionFile(t)
    """

    def __init__(
        self,
        path: str,
        open_file: Callable[[str], io.IOBase],
        rename: Callable[[str, str], object],
        delete: Callable[[str], object],
        *,
        temp_path: str | None = None,
    ) -> None:
        super().__init__(path)
        self.open_file = open_file
        self.rename = rename
        self.delete = delete
        self.temp_path = temp_path or _temp_path(path)
        self._file: io.IOBase | None = None

    def start_staging(self) -> io.IOBase:
        self._file = self.open_file(self.temp_path)
        return self._file

    def end_staging(self) -> None:
        if self._file is not None:
            self._file.close()

    def commit(self) -> None:
        self.rename(self.temp_path, self.path)

    def discard(self) -> None:
        try:
            self.end_staging()
        finally:
            # Never written, e.g. if discarded before the file uploaded.
            with contextlib.suppress(FileNotFoundError):
                self.delete(self.temp_path)


class LocalBufferTransaction(FileTransaction):
    """A file written to a local buffer, written to its own on commit.

    The buffer is a `tempfile.SpooledTemporaryFile`: in memory up to
    ``spool_size`` bytes, then on disk. Ending the staging keeps it as it is.
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

    def start_staging(self) -> io.IOBase:
        self._buffer = tempfile.SpooledTemporaryFile(max_size=self.spool_size)  # noqa: SIM115
        return self._buffer

    def end_staging(self) -> None:
        pass  # The buffer is kept for commit.

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


class _Discarded(Exception):
    """Raised in a file's closing, so that the file aborts its upload."""


class AdaptiveWritableTransaction(FileTransaction):
    """An adaptive file uploading its parts on closing, completed on commit.

    The file's last request, the one making it visible, waits for commit: the
    multipart upload's completion, or a small file's upload. The transaction
    stands in for the file's ``_complete_multipart_upload`` and
    ``_upload_all``, on that file only, so as to hold that request back.
    Discard aborts the multipart upload, or drops the small file.

    It costs nothing more than writing the file without a transaction, and
    other clients see nothing until commit. A small file is held in memory
    until then. Uploaded parts take storage until commit: if the process dies
    first, they stay until the storage expires them.

    Parameters
    ----------
    file : AdaptiveWritableFile
        The file, open for writing.

    Examples
    --------
    >>> f = TransactionFile(AdaptiveWritableTransaction(MyFile(fs, path)))
    """

    def __init__(self, file: AdaptiveWritableFile) -> None:
        super().__init__(file.path)
        self.file = file
        # The request making the file visible, once closing held it back.
        self._last: Callable[[], Awaitable[None]] | None = None
        self._multipart = False
        self._discarding = False

    def start_staging(self) -> io.IOBase:
        file = self.file
        complete, upload_all = file._complete_multipart_upload, file._upload_all

        async def hold_completion(parts: list[Any]) -> None:
            if self._discarding:
                raise _Discarded  # The file aborts the upload.
            self._multipart = True
            self._last = lambda: complete(parts)

        async def hold_upload(data: bytes) -> None:
            self._last = lambda: upload_all(data)

        file._complete_multipart_upload = hold_completion  # type: ignore[method-assign]
        file._upload_all = hold_upload  # type: ignore[method-assign]
        return file

    def end_staging(self) -> None:
        self.file.close()

    def commit(self) -> None:
        try:
            if self._last is not None:
                self.file._sync(self._last)
        except BaseException:
            self._abort()
            raise

    def discard(self) -> None:
        self._discarding = True
        if not self.file.closed:
            with contextlib.suppress(_Discarded):
                self.file.close()
        else:
            self._abort()
        self._last = None

    def _abort(self) -> None:
        if self._multipart:
            self._multipart = False
            self.file._sync(self.file._abort_multipart_upload)


class LocallyBufferedTransaction(FileTransaction):
    """A locally buffered file, uploaded on commit.

    The file holds its bytes locally until closed: so closing it waits for
    commit. Discard closes it after its own ``discard``, uploading nothing.

    It costs nothing more than writing the file without a transaction, and
    other clients see nothing until commit. Exclusive creation, "x", is checked
    on the first write, or on commit. If neither commit nor discard comes, the
    file uploads when garbage collected, as any open file of its class does.

    Parameters
    ----------
    file : LocallyBufferedFile
        The file, open for writing.

    Examples
    --------
    >>> f = TransactionFile(LocallyBufferedTransaction(MyFile(fs, path, "wb")))
    """

    def __init__(self, file: LocallyBufferedFile) -> None:
        super().__init__(file.path)
        self.file = file

    def start_staging(self) -> io.IOBase:
        return self.file

    def end_staging(self) -> None:
        pass  # The file keeps its buffer for commit.

    def commit(self) -> None:
        self.file.close()

    def discard(self) -> None:
        self.file.discard()
        self.file.close()
