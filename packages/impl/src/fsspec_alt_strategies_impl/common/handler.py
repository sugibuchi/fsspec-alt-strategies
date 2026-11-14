# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What a storage handler holds, whatever its family: the base of `TreeHandler` and `FlatHandler`.

A storage handler holds what is particular to a storage: its primitives, its
profile, its limits, and its clients, opened and closed. The strategies take
handlers, never filesystems. A family's handler class adds the primitives its
strategies call, e.g. a tree's ``rm_tree``.

The primitives name an item that exists by its entry, and one to be made by
its path. An entry is a dict whose ``"name"`` alone is sure: a path given by a
caller comes as `StorageHandler.entry`'s ``{"name": path}``, and a listing's
entries come whole.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NewType, TypeAlias

from fsspec.utils import stringify_path

from fsspec_alt_strategies.profile import UNKNOWN, StorageProfile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime

__all__ = ["Entry", "StorageHandler", "StoragePath"]


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

    def verified(self, path: str) -> StoragePath:
        """A caller's path, a URL or not, as a `StoragePath`, once `verify_path` accepts it, else ValueError.

        `verify_path` sees the path as given, then `strip_protocol` drops the
        protocol and a trailing "/". So a filesystem passes its caller's paths
        on as they come: this is where they lose their protocol.

        It and `entry` are the only ways a string becomes a path. A strategy
        builds the others by `join` and `parent`, which verify too.
        """
        self.verify_path(path)
        return StoragePath(self.strip_protocol(path))

    def entry(self, path: str) -> Entry:
        """The entry of a path alone, verified: for a primitive taking an item its caller knows only by its path.

        It raises ValueError for a path `verify_path` refuses. A name a listing
        gave is never wrapped so: the strategies pass on the entry the listing
        gave, as the storage holds what it lists, whatever its name.
        """
        return {"name": self.verified(path)}

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
    one of ``operation_limit`` around each primitive they call.

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

    @abstractmethod
    async def makedirs(self, path: StoragePath, exist_ok: bool = False) -> None:
        """Make a directory and its parents, raising FileExistsError unless ``exist_ok``."""

    @abstractmethod
    async def cp_file(self, entry: Entry, path2: StoragePath) -> None:
        """Copy a file within the storage, to ``path2``.

        The entry's keys, e.g. its size, may save a request. They may be
        stale: a copy must still succeed, or fail as the storage says, if the
        file changed since. A directory at ``path2`` is never replaced: it
        raises IsADirectoryError, or FileExistsError.
        """

    @abstractmethod
    async def mv_file(self, entry: Entry, path2: StoragePath) -> None:
        """Move a file within the storage, to ``path2``, as `cp_file` copies one.

        A storage renaming a file in one request renames it. One with no rename
        copies, then deletes the source. The strategies call it only where
        ``path2``'s parent exists.
        """

    # Files. The filesystem awaits them on its own loop, for its sync open too.

    async def open_file(self, entry: Entry, mode: str = "rb", **kwargs: Any) -> Any:
        """Open a file, one used sync or async: to read, mode "rb", or to write, "wb" or "xb".

        It returns e.g. an `AdaptiveReadableFile` or an `AdaptiveWritableFile`,
        whose requests hold ``io_limit``, as their concurrency limit.

        It takes an entry whatever the mode, as every primitive on a file
        does. A file to write may not exist yet: its entry may hold its name
        alone, verified, as `entry` makes it. One a listing found may hold
        more, e.g. an id to overwrite it by.

        To read, a missing file raises FileNotFoundError, and a directory
        IsADirectoryError. To write, "xb" raises FileExistsError where a file
        is, as the storage tells it.
        """
        raise NotImplementedError(f"{type(self).__name__} opens no file")

    # What fsspec leaves to each filesystem.

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


def _limit(concurrency: int | asyncio.Semaphore, name: str) -> asyncio.Semaphore:
    """The semaphore of a concurrency: the one given, shared, or a new one of the number given."""
    if isinstance(concurrency, asyncio.Semaphore):
        return concurrency
    if concurrency < 1:
        raise ValueError(f"{name} must be at least 1: {concurrency}")
    return asyncio.Semaphore(concurrency)
