# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What a storage handler holds, whatever its family: the base of `TreeFileOperationHandler` and `FlatFileOperationHandler`.

A storage handler holds what is particular to a storage: its primitives, its
profile, its limits, and its clients, opened and closed. The strategies take
handlers, never filesystems. A family's handler class adds the primitives its
strategies call, e.g. a tree's ``rm_tree``.

`FileOperationHandler` is a `StorageHandler` with the methods of `Strategy`,
which run the strategies on it: what a filesystem takes. A family's subclass
of it, e.g. `TreeFileOperationHandler`, runs its family's strategies.

The primitives name an item that exists by its entry, and one to be made by
its path. An entry is a dict whose ``"name"`` alone is sure: a path given by a
caller comes as `StorageHandler.entry`'s ``{"name": path}``, and a listing's
entries come whole.

This module depends on `strategy`.
"""

from __future__ import annotations

import asyncio
import io
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NewType, TypeAlias

from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.utils import stringify_path

from fsspec_alt_strategies.profile import UNKNOWN, StorageProfile

from .strategy import Strategy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

    from fsspec.callbacks import Callback

__all__ = ["Entry", "FileOperationHandler", "StorageHandler", "StoragePath"]

_MiB = 1024 * 1024


StoragePath = NewType("StoragePath", str)
"""A path of a storage, without its protocol or a trailing "/", as its handler checked it.

Never make one directly, as ``StoragePath(path)``: that skips the check. Get
one from the handler: `StorageHandler.verified` for a caller's path, or its
``join`` and ``parent`` for one built from another.
"""


Entry: TypeAlias = dict[str, Any]
"""An item of a storage, as its handler describes it, as fsspec's ``ls`` with ``detail`` gives it.

Only ``"name"`` is sure: its path, a `StoragePath` where `StorageHandler.entry`
made it, else the name the storage listed, which is never verified again. The
other keys, e.g. ``"type"``, ``"size"``, or the storage's own, e.g. an
``"id"``, are there when the storage told them. A key missing is unknown, never
absent on the storage: a missing ``"size"`` is not 0. A key may be stale,
listed before a change.
"""


class _PathHandler(ABC):
    """A handler's paths: what it verifies, and how it builds paths, with or without primitives.

    The base of `StorageHandler`, and of a router's handler, which routes its
    paths to storages but has no primitives of its own. The strategies build
    paths only by these methods, so each path they build is verified.
    """

    protocol: str | tuple[str, ...] = "abstract"
    """The storage's protocol, or several, as an fsspec filesystem's: the first is the one a URL is given."""

    root: str = ""
    """The path of the storage's root, e.g. "" for an object store or "/" for a local disk."""

    @classmethod
    def strip_protocol(cls, path: Any) -> Any:
        """A URL or path as the storage's path, as fsspec's ``_strip_protocol`` gives it.

        Any of ``protocol`` is removed, with "://" or "::", and a trailing "/":
        an empty path is the ``root``. Where the root is "/", the paths are
        absolute: one given without it gains it, so that ``proto://a/b`` is
        ``/a/b``. A list gives a list. It is a class method, so that a
        filesystem's class strips URLs before it is made, as fsspec does. A
        handler serving another filesystem's paths, as a wrapper does,
        overrides it with the other's, as does one whose URLs name a host.
        """
        if isinstance(path, list):
            return [cls.strip_protocol(p) for p in path]
        path = stringify_path(path)
        protocols = (cls.protocol,) if isinstance(cls.protocol, str) else cls.protocol
        for protocol in protocols:
            if path.startswith(f"{protocol}://"):
                path = path[len(protocol) + 3 :]
            elif path.startswith(f"{protocol}::"):
                path = path[len(protocol) + 2 :]
        path = path.rstrip("/") or cls.root
        if cls.root == "/" and not path.startswith("/"):
            return "/" + path
        else:
            return path

    @classmethod
    def options_from_url(cls, url: str) -> dict[str, Any]:
        """The storage options a URL names beyond its path, for fsspec's ``_get_kwargs_from_urls``: here, none.

        fsspec asks a filesystem's class for them as it makes one from a URL,
        e.g. by ``fsspec.open("abfss://container@account.dfs.core.windows.net/x")``,
        before any handler is made: so it is a class method. A handler whose
        URLs name an account, a host or a user returns them, under the names
        of its storage class's options, which it is made from. A storage's
        class names it as its ``_get_kwargs_from_urls``, as it names
        `strip_protocol` as its ``_strip_protocol``.
        """
        return {}

    def unstrip_protocol(self, name: str) -> str:
        """A path of the storage as a URL, with the first ``protocol``, as fsspec's ``unstrip_protocol``."""
        protocols = (
            (self.protocol,) if isinstance(self.protocol, str) else self.protocol
        )
        if any(name.startswith(f"{protocol}://") for protocol in protocols):
            return name
        return f"{protocols[0]}://{name}"

    @abstractmethod
    def verify_path(self, path: str) -> None:
        """Raise ValueError if ``path`` is no path of this storage.

        The path is as its caller gave it: a URL, or a path already without
        its protocol. A URL may name more than the path, e.g.
        ``abfs://container@account.dfs.core.windows.net/path``, whose account
        is checked here, as `strip_protocol` then drops it. A handler checking
        the path's names strips it first, by `strip_protocol`.

        It sees a glob pattern too, e.g. ``data/2026-*/x.csv``, given to
        ``glob``, or to a copy or deletion. A handler whose names refuse a
        pattern's characters checks its fixed prefix instead, by
        `paths.fixed_prefix`: the names a glob matches below are listed.

        A handler refuses what its storage would, or what it does not serve,
        e.g. a path of another storage account than its own, a bucket name the
        storage refuses, or a name too long. It sends no request: it checks
        the path alone. The handler's ``root`` is a path it accepts, as the
        parent of the paths at the root.

        The filesystem verifies each path a caller gives before it calls a
        strategy or a primitive. The paths `join` and `parent` build, and
        those `entry` wraps, are verified by them. So a strategy and
        a primitive take only verified paths, or entries a listing gave.
        """

    def verified(self, path: str, *, strip_only_protocol: bool = False) -> StoragePath:
        """A caller's path, a URL or not, as a `StoragePath`, once `verify_path` accepts it, else ValueError.

        `verify_path` sees the path as given, then `strip_protocol` drops the
        protocol and a trailing "/". So a filesystem passes its caller's paths
        on as they come: this is where they lose their protocol.

        With ``strip_only_protocol``, the protocol alone is stripped, for a
        path a strategy takes as given. A trailing "/" is kept, as it means a
        directory to a copy or a glob. The path is verified all the same, a
        glob pattern too: a handler refusing a pattern's characters refuses
        the glob.

        It and `entry` are the only ways a string becomes a path. A strategy
        builds the others by `join` and `parent`, which verify too.
        """
        self.verify_path(path)
        plain = self.strip_protocol(path)
        if strip_only_protocol and path.endswith("/") and not plain.endswith("/"):
            plain += "/"
        return StoragePath(plain)

    def entry(self, path: str, *, strip_only_protocol: bool = False) -> Entry:
        """The entry of a path alone, verified: for a primitive taking an item its caller knows only by its path.

        It raises ValueError for a path `verify_path` refuses. A name a listing
        gave is never wrapped so: the strategies pass on the entry the listing
        gave, as the storage holds what it lists, whatever its name.
        ``strip_only_protocol`` is `verified`'s.
        """
        return {"name": self.verified(path, strip_only_protocol=strip_only_protocol)}

    def join(self, directory: StoragePath, *paths: str) -> StoragePath:
        """The path of ``paths`` in ``directory``, each relative to the one before, as `os.path.join` joins them.

        Each path is split into segments by `_split`, e.g. "a/b" into "a" and
        "b", and each segment checked by `_verify_segment`, which refuses "..".
        The segments are joined to the directory by ``delimiter``, and the path
        built is verified once by `verify_path`, e.g. for a name the storage
        refuses. A segment or a path refused raises ValueError.

        So a path built never leaves the directory it was built from, whatever
        names another storage listed. Unlike `os.path.join`, a path starting
        with the delimiter is still relative: it never starts again from the
        root. No path, or only empty ones, is the directory itself.
        """
        path = str(directory)
        delimiter = self.delimiter
        joined = False
        for relative in paths:
            for segment in self._split(relative):
                self._verify_segment(segment)
                if path and not path.endswith(delimiter):
                    path += delimiter
                path += segment
                joined = True
        return self.verified(path) if joined else directory

    delimiter: str = "/"
    """The delimiter between a path's names, which `join` splits and joins paths by."""

    def _split(self, path: str) -> list[str]:
        """The segments of a relative path, for `join`: split by ``delimiter``.

        One leading delimiter, the one after the directory's own path, and the
        trailing ones are ignored, so an empty path has none. A storage whose
        names may hold the delimiter, escaped, overrides it.
        """
        delimiter = self.delimiter
        rest = path.removeprefix(delimiter).rstrip(delimiter)
        return rest.split(delimiter) if rest else []

    def _verify_segment(self, segment: str) -> None:
        """Raise ValueError for a segment that is no single name: empty, ".", "..", or holding the delimiter.

        `join` checks each segment by it before joining it. A storage refusing
        other names, or allowing ".", overrides it, calling this one unless
        it relaxes it. The whole path is checked afterwards by `verify_path`.
        """
        if not segment or segment in (".", "..") or self.delimiter in segment:
            raise ValueError(f"Not a name in a directory: {segment!r}")

    def parent(self, path: str) -> StoragePath:
        """The directory a path is in, verified: the root for one at the root.

        The path may be one a listing gave: its parent is verified all the same.

        The root is in no directory: it raises ValueError, as does a parent
        `verify_path` refuses, e.g. one too short for the storage's paths.
        """
        delimiter = self.delimiter
        trimmed = path.rstrip(delimiter) or path
        if trimmed == self.root:
            raise ValueError(f"The root has no parent: {path!r}")
        head = trimmed.rpartition(delimiter)[0]
        return self.verified(
            head or (self.root if trimmed.startswith(self.root) else "")
        )

    def name(self, path: str) -> str:
        """The last name of a path, after its last ``delimiter``."""
        return path.rstrip(self.delimiter).rpartition(self.delimiter)[2]


class StorageHandler(_PathHandler):
    """A storage's primitives, profile, limits and clients, for the strategies.

    It is made without a request. `open` loads its profile, and makes its
    clients, on the running loop: the strategies read ``profile`` once it is
    open. `aclose` releases what `open` made, and `close` all it holds, from a
    sync caller.

    Two limits bound its requests, each an `asyncio.Semaphore`:

    - ``operation_limit``, the strategies' requests, such as listings,
      deletions and copies;
    - ``io_limit``, the reads and writes of data.

    The callers hold the permits, never the primitives: the strategies hold
    one of ``operation_limit`` around each primitive they call. Data is the
    exception: a file's requests take ``io_limit``'s, and so does an override
    of `cat_file`, `pipe_file`, `get_file` or `put_file` around each request
    it sends.

    Parameters
    ----------
    operation_concurrency : int or asyncio.Semaphore
        The requests of the strategies at a time. A semaphore is used as it
        is, shared with what else holds it.
    io_concurrency : int or asyncio.Semaphore
        The reads and writes of data at a time. A semaphore is used as it is.

    Raises
    ------
    ValueError
        If a concurrency is less than 1.
    """

    profile: StorageProfile = UNKNOWN
    """What the storage does, for the strategies: unknown, right on any storage, by default.

    A subclass sets what it knows. One overriding `load_profile` has it
    replaced by what that loads, as `open` does.
    """

    def __init__(
        self,
        *,
        operation_concurrency: int | asyncio.Semaphore,
        io_concurrency: int | asyncio.Semaphore,
    ) -> None:
        self.operation_concurrency = operation_concurrency
        self.io_concurrency = io_concurrency
        self.operation_limit = _limit(operation_concurrency, "operation_concurrency")
        self.io_limit = _limit(io_concurrency, "io_concurrency")
        self._opened = False
        self._open_lock = asyncio.Lock()

    # Its lifecycle.

    async def open(self) -> None:
        """Load the profile, and make the clients, once: a call after the first does nothing.

        A first call opens it while the others wait, then find it open. One
        that raises leaves it closed: the next call tries again. So a profile
        that cannot be told yet is not kept.
        """
        if self._opened:
            return
        async with self._open_lock:
            if not self._opened:
                self.profile = await self.load_profile()
                self._opened = True

    async def load_profile(self) -> StorageProfile:
        """Find the profile, by a request where the storage tells it: here, ``profile`` as declared.

        It takes no path: the profile is one for the whole storage. A storage
        whose behaviour differs by path is several handlers under a router.
        """
        return self.profile

    async def aclose(self) -> None:
        """Release what `open` made, on the running loop: a later `open` opens it again.

        A subclass closes its clients, then calls this one. It never calls
        `close`, which a handler may take as closing it for good: a handler
        closed so is open again for the next call.
        """
        self._opened = False

    def close(self) -> None:
        """Release at once, from a sync caller, what the handler holds: here, nothing.

        A filesystem's sync ``close()`` calls it, after which a handler may
        refuse calls, as a closed filesystem does. A subclass holding threads
        shuts them down here, and one holding clients closes them as it can
        without awaiting. `aclose` never calls it.
        """

    # The primitives of single paths, which every handler has.

    @abstractmethod
    def ils(self, entry: Entry, **kwargs: Any) -> AsyncIterator[Entry]:
        """List a directory, as an async iterator of its entries, a page at a time.

        A file lists itself, as fsspec's ``_ls`` lists it. A missing directory
        raises FileNotFoundError, when the iteration starts. The entries come
        in any order: the listing is done when it ends.
        """

    def ils_recursive(self, entry: Entry, **kwargs: Any) -> AsyncIterator[Entry]:
        """List all below a directory, at any depth, as an async iterator of entries.

        Each family says its order and its options: a tree handler lists a
        tree depth first or by name, and a flat handler lists keys by name,
        narrowed by a ``prefix``. The filesystem's recursive ``_ils`` calls it.
        Here, it raises NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} lists nothing recursively")

    @abstractmethod
    async def info(self, entry: Entry, **kwargs: Any) -> Entry:
        """The whole entry of an item, raising FileNotFoundError if there is none."""

    @abstractmethod
    async def rm_file(self, entry: Entry) -> None:
        """Delete a file.

        The strategies call it only on items found to be files, by a listing
        or an `info`.
        """

    @abstractmethod
    async def rmdir(self, entry: Entry) -> None:
        """Delete an empty directory.

        One already gone raises FileNotFoundError, which the strategies count
        as deleted: on a storage whose directories go with their last file, it
        is.
        """

    # The primitives making an item take ``missing``: True where the caller
    # knows nothing is at the path, e.g. a copy into a directory it made
    # itself, unless another client put something there since. A storage whose
    # creation never conflicts, e.g. Google Drive's, which would make a second
    # item of the name, then skips asking what is there first. Any other
    # storage may ignore it: False is always right.

    @abstractmethod
    async def makedirs(
        self, path: StoragePath, exist_ok: bool = False, *, missing: bool = False
    ) -> None:
        """Make a directory and its parents, raising FileExistsError unless ``exist_ok``.

        ``missing`` says nothing is at ``path``, as the comment above says.
        """

    @abstractmethod
    async def cp_file(
        self, entry: Entry, path2: StoragePath, *, missing: bool = False
    ) -> None:
        """Copy a file within the storage, to ``path2``.

        The entry's keys, e.g. its size, may save a request. They may be
        stale: a copy must still succeed, or fail as the storage says, if the
        file changed since. A directory at ``path2`` is never replaced: it
        raises IsADirectoryError, or FileExistsError. ``missing`` says nothing
        is at ``path2``, as the comment above says.
        """

    @abstractmethod
    async def mv_file(
        self, entry: Entry, path2: StoragePath, *, missing: bool = False
    ) -> None:
        """Move a file within the storage, to ``path2``, as `cp_file` copies one.

        A storage renaming a file in one request renames it. One with no rename
        copies, then deletes the source. The strategies call it only where
        ``path2``'s parent exists. ``missing`` is `cp_file`'s.
        """

    # Files: to read, mode "rb", or to write, "wb" or "xb". Each takes an entry
    # whatever the mode: a file to write may not exist yet, its entry then
    # holding its name, and its final size as "size" where the caller knows
    # it, e.g. `pipe_file`'s default, so that a storage needing it up front,
    # e.g. for an upload session, need not hold the whole file first. To
    # read, a missing file raises FileNotFoundError, and a directory
    # IsADirectoryError. To write, "xb" raises FileExistsError where a file
    # is, as the storage tells it.

    def open_file(
        self,
        entry: Entry,
        mode: str = "rb",
        *,
        loop: asyncio.AbstractEventLoop | None,
        autocommit: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Open a file-like object with Python's sync API, for fsspec's sync ``open``.

        The filesystem returns the file as it is, e.g. an
        `AdaptiveReadableFile` or an `AdaptiveWritableFile`. ``loop`` is the
        filesystem's, which its sync methods run on: a file whose requests
        are awaited runs them there, and a request opening it is awaited
        there by fsspec's ``sync``. It is None for a filesystem made with
        ``asynchronous=True``. Its requests hold ``io_limit``.

        Within fsspec's transaction, ``autocommit`` is False: a handler
        supporting it returns a file committed with the transaction, e.g. a
        `TransactionFile`. One that does not writes as it would without.
        ``block_size`` and ``cache_options``, fsspec's, come in ``kwargs``.
        """
        raise NotImplementedError(f"{type(self).__name__} opens no file")

    async def open_async_file(
        self, entry: Entry, mode: str = "rb", **kwargs: Any
    ) -> Any:
        """Open a file with the API of fsspec's ``open_async`` files: its methods awaited.

        The filesystem's ``open_async``, the strategies' copies between
        storages, and the defaults of `cat_file`, `pipe_file`, `get_file`
        and `put_file` use it. A handler gives its own files that API, e.g.
        an adaptive file through `AsyncFileAdapter`, or an SDK's handle
        through `SyncFileAdapter`. Its requests hold ``io_limit``.
        """
        raise NotImplementedError(f"{type(self).__name__} opens no async file")

    # What a filesystem on the handler keeps.

    def invalidate_cache(self, path: str | None = None) -> None:
        """Drop what the handler keeps of the storage's listings, for a filesystem's ``invalidate_cache``: here, nothing.

        A handler keeping ids by path, or a wrapper whose filesystem keeps its
        own listings, drops them here.
        """

    async def sign(self, entry: Entry, expiration: int = 100, **kwargs: Any) -> str:
        """A URL of a file, signed to be read without the storage's credential, for ``expiration`` seconds.

        The filesystem's ``sign`` answers it, as fsspec's ``sign`` does. It is
        awaited, as signing may ask the storage first, e.g. for a key to sign
        with. A storage signing no URL raises NotImplementedError, as fsspec's
        ``sign`` does.
        """
        raise NotImplementedError(f"{type(self).__name__} signs no URL")

    async def created(self, entry: Entry) -> datetime:
        """When an item was created, as fsspec's ``created`` answers.

        A storage telling no creation time raises NotImplementedError, as
        fsspec's ``created`` does.
        """
        raise NotImplementedError(f"{type(self).__name__} tells no creation time")

    async def modified(self, entry: Entry) -> datetime:
        """When an item was last modified, as fsspec's ``modified`` answers.

        A storage telling no modification time raises NotImplementedError, as
        fsspec's ``modified`` does.
        """
        raise NotImplementedError(f"{type(self).__name__} tells no modification time")

    # Data moved at once: `cat_file`, `pipe_file`, `get_file` and `put_file`
    # move a file's data whole, or `cat_file` a range of it. Their defaults
    # stream it through an async file opened by `open_async_file`. A handler
    # may override one with requests of its own, which know the range to move
    # in advance: e.g. one ranged download in place of a seek and a read. Such
    # an override holds a permit of io_limit around each request it sends, as
    # the files do. The defaults hold none: the files take their own, and a
    # permit held around them would wait for itself.

    transfer_size: int = 8 * _MiB
    """The bytes read or written at a time by the defaults of `get_file` and `put_file`."""

    async def cat_file(
        self,
        entry: Entry,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Read a file, or a range of it, as fsspec's ``cat_file`` does: by default through `open_async_file`.

        A file that cannot seek, e.g. a stream of an SDK, is read up to
        ``start`` and the bytes dropped, a ``transfer_size`` at a time.
        """
        async with await self.open_async_file(entry, "rb", **kwargs) as f:
            position = 0
            if start is not None:
                position = start if start >= 0 else max(0, f.size + start)
                await _skip(f, position, self.transfer_size)

            if end is not None:
                if end < 0:
                    end = f.size + end
                return await f.read(max(0, end - position))  # type: ignore[no-any-return]
            else:
                return await f.read()  # type: ignore[no-any-return]

    async def pipe_file(
        self, path: StoragePath, value: bytes, mode: str = "overwrite", **kwargs: Any
    ) -> None:
        """Write a file whole, as fsspec's ``pipe_file`` does: by default through `open_async_file`.

        Mode "create" writes it only if none is there, by `open_async_file`'s "xb".
        """
        # The size goes with the name, as the writer may need it up front.
        entry = {"name": path, "size": len(value)}
        async with await self.open_async_file(entry, _write_mode(mode), **kwargs) as f:
            await f.write(value)

    async def get_file(
        self,
        entry: Entry,
        lpath: str,
        callback: Callback = DEFAULT_CALLBACK,
        **kwargs: Any,
    ) -> None:
        """Copy a file to a local path, as fsspec's ``get_file`` does: by default through `open_async_file`."""
        async with await self.open_async_file(entry, "rb", **kwargs) as f:
            # The bytes, as fsspec's own filesystems tell them to the callback.
            callback.set_size(f.size)

            # Local writes are short next to a storage's requests, as in fsspec's
            # own filesystems.
            with open(lpath, "wb") as local:  # noqa: ASYNC230
                while chunk := await f.read(self.transfer_size):
                    local.write(chunk)
                    callback.relative_update(len(chunk))

    async def put_file(
        self,
        lpath: str,
        path: StoragePath,
        mode: str = "overwrite",
        callback: Callback = DEFAULT_CALLBACK,
        **kwargs: Any,
    ) -> None:
        """Copy a local file to the storage, as fsspec's ``put_file`` does: by default through `open_async_file`."""
        with open(lpath, "rb") as local:  # noqa: ASYNC230
            size = os.fstat(local.fileno()).st_size
            callback.set_size(size)
            # The size goes with the name, as the writer may need it up front.
            entry = {"name": path, "size": size}
            async with await self.open_async_file(
                entry, _write_mode(mode), **kwargs
            ) as f:
                while chunk := local.read(self.transfer_size):
                    await f.write(chunk)
                    callback.relative_update(len(chunk))


class FileOperationHandler(StorageHandler, Strategy):
    """A storage's primitives, with the strategies its filesystem's recursive methods run.

    `BaseFileSystem` takes one. A family's subclass, `TreeFileOperationHandler`
    or `FlatFileOperationHandler`, runs its family's strategies. A storage's
    handler subclasses one, and overrides a strategy to choose another.

    Parameters
    ----------
    operation_concurrency, io_concurrency
        As `StorageHandler` takes them.

    See Also
    --------
    StorageHandler : The primitives, profile, limits and clients.
    Strategy : The strategies.
    """


def _write_mode(mode: str) -> str:
    """`open_async_file`'s mode for fsspec's ``pipe_file`` and ``put_file`` mode: "xb" to create, else "wb"."""
    return "xb" if mode == "create" else "wb"


async def _skip(f: Any, start: int, chunk_size: int) -> None:
    """Move a file just opened to ``start``: by seeking, or else by reading.

    A storage SDK's handle may stream its data, unable to seek. Its bytes up
    to ``start`` are then read and dropped, a chunk at a time, so that memory
    stays bounded.
    """
    try:
        if f.seekable():
            f.seek(start)
            return
    except (io.UnsupportedOperation, OSError):
        pass

    skipped = 0
    while skipped < start:
        chunk = await f.read(min(chunk_size, start - skipped))
        if not chunk:
            return  # Past the end: the read after answers nothing.
        skipped += len(chunk)


def _limit(concurrency: int | asyncio.Semaphore, name: str) -> asyncio.Semaphore:
    """The semaphore of a concurrency: the one given, shared, or a new one of the number given."""
    if isinstance(concurrency, asyncio.Semaphore):
        return concurrency
    if concurrency < 1:
        raise ValueError(f"{name} must be at least 1: {concurrency}")
    return asyncio.Semaphore(concurrency)
