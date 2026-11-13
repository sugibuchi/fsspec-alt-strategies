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

from fsspec_alt_strategies.profile import UNKNOWN, StorageProfile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

__all__: list[str] = []


StoragePath = NewType("StoragePath", str)
"""A path of a storage, without its protocol or a trailing "/", as its handler checked it."""


Entry: TypeAlias = dict[str, Any]
"""An item of a storage, as its handler describes it, as fsspec's ``ls`` with ``detail`` gives it.

Only ``"name"`` is sure: its path, a `StoragePath` where `StorageHandler.entry`
made it, else the name the storage listed, which is never verified again. The
other keys, e.g. ``"type"``, ``"size"``, or the storage's own, e.g. an
``"id"``, are there when the storage told them. A key missing is unknown, never
absent on the storage: a missing ``"size"`` is not 0. A key may be stale,
listed before a change.
"""


class StorageHandler(ABC):
    """A storage's primitives, profile, limits and clients, for the strategies.

    It is made without a request. `open` loads its profile, and makes its
    clients, on the running loop: the strategies read ``profile`` once it is
    open. `close` releases what `open` made.

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

    async def close(self) -> None:
        """Release what `open` made: here, nothing. A later `open` opens it again."""
        self._opened = False

    # Paths and entries: the strategies build paths only by these.

    root: str = ""
    """The path of the storage's root, e.g. "" for an object store or "/" for a local disk."""

    @abstractmethod
    def verify_path(self, path: str) -> None:
        """Raise ValueError if ``path`` is no path of this storage.

        The path has no protocol. A handler refuses what its storage would, or
        what it does not serve, e.g. a path of another storage account than its
        own, a bucket name the storage refuses, or a name too long. It sends no
        request: it checks the path alone. The handler's ``root`` is a path it
        accepts, as the parent of the paths at the root.

        The filesystem verifies each path a caller gives before it calls a
        strategy or a primitive. The paths `join`, `join_all` and `parent`
        build, and those `entry` wraps, are verified by them. So a strategy and
        a primitive take only verified paths, or entries a listing gave.
        """

    def verified(self, path: str) -> StoragePath:
        """A caller's path as a `StoragePath`, once `verify_path` accepts it, else ValueError.

        It and `entry` are the only ways a string becomes a path. A strategy
        builds the others by `join`, `join_all` and `parent`, which verify too.
        """
        self.verify_path(path)
        return StoragePath(path)

    def entry(self, path: str) -> Entry:
        """The entry of a path alone, verified: for a primitive taking an item its caller knows only by its path.

        It raises ValueError for a path `verify_path` refuses. A name a listing
        gave is never wrapped so: the strategies pass on the entry the listing
        gave, as the storage holds what it lists, whatever its name.
        """
        return {"name": self.verified(path)}

    def join(self, directory: StoragePath, name: str) -> StoragePath:
        """The path of ``name`` in ``directory``, verified, raising ValueError for one refused.

        A name is refused if empty, ".", "..", or holding a "/". So a path built
        by joining names never leaves the directory it was built from, whatever
        names another storage listed. The path built is then verified by
        `verify_path`, e.g. for a name the storage refuses.
        """
        return self.verified(_joined(directory, name))

    def join_all(self, directory: StoragePath, relative: str) -> StoragePath:
        """The path of ``relative``, names joined by "/", in ``directory``, verified as `join` verifies one.

        Each name is checked as `join` checks one, and the path built is
        verified once. One leading "/", the separator after the directory's own
        path, and a trailing one are ignored. An empty ``relative`` is the
        directory itself.
        """
        path = StoragePath(directory)
        rest = relative.removeprefix("/").rstrip("/")
        if not rest:
            return path
        for name in rest.split("/"):
            path = _joined(path, name)
        return self.verified(path)

    def parent(self, path: str) -> StoragePath:
        """The directory a path is in, verified: the root for one at the root.

        The path may be one a listing gave: its parent is verified all the same.

        The root is in no directory: it raises ValueError, as does a parent
        `verify_path` refuses, e.g. one too short for the storage's paths.
        """
        trimmed = path.rstrip("/") or path
        if trimmed == self.root:
            raise ValueError(f"The root has no parent: {path!r}")
        head = trimmed.rpartition("/")[0]
        return self.verified(
            head or (self.root if trimmed.startswith(self.root) else "")
        )

    def name(self, path: str) -> str:
        """The last name of a path."""
        return path.rstrip("/").rpartition("/")[2]

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


class _FileSystemHandler(StorageHandler):
    """An fsspec filesystem as a handler, its primitives by path, until the filesystems are made on handlers.

    It calls the filesystem's own primitives with each entry's name, and its
    ``_cp_file_entry`` and ``_mv_file_entry`` with the entry, where it has
    them. Its limits and profile are the filesystem's. A router's destination
    may be any async filesystem, with neither: its limits are then new, and
    its profile unknown.
    """

    # The filesystem's recursive _ils gives either order the strategies take.
    recursive_order = "depth first"

    def __init__(self, fs: Any) -> None:
        super().__init__(
            operation_concurrency=getattr(fs, "operation_limit", None) or 32,
            io_concurrency=getattr(fs, "io_limit", None) or 8,
        )
        self.fs = fs
        self.profile = getattr(fs, "storage_profile", UNKNOWN)
        self.root = getattr(fs, "root_marker", "")

    def __repr__(self) -> str:
        return f"<handler of {self.fs!r}>"

    @property
    def rm_batch_size(self) -> int:
        """The flat filesystem's own, for `rm_files`."""
        return self.fs.rm_batch_size  # type: ignore[no-any-return]

    async def load_profile(self) -> StorageProfile:
        get = getattr(self.fs, "_get_storage_profile", None)
        return await get() if get is not None else UNKNOWN

    def verify_path(self, path: str) -> None:
        """Nothing: an fsspec filesystem checks its paths in its own methods."""

    def ils(self, entry: Entry, **kwargs: Any) -> AsyncIterator[Entry]:
        return self.fs._ils(entry["name"], **kwargs)  # type: ignore[no-any-return]

    def ils_recursive(self, entry: Entry, **kwargs: Any) -> AsyncIterator[Entry]:
        return self.fs._ils(entry["name"], recursive=True, **kwargs)  # type: ignore[no-any-return]

    async def info(self, entry: Entry, **kwargs: Any) -> Entry:
        return await self.fs._info(entry["name"], **kwargs)  # type: ignore[no-any-return]

    async def rm_file(self, entry: Entry) -> None:
        await self.fs._rm_file(entry["name"])

    async def rmdir(self, entry: Entry) -> None:
        await self.fs._rmdir(entry["name"])

    async def makedirs(self, path: StoragePath, exist_ok: bool = False) -> None:
        await self.fs._makedirs(path, exist_ok=exist_ok)

    async def cp_file(self, entry: Entry, path2: StoragePath) -> None:
        if (cp := getattr(self.fs, "_cp_file_entry", None)) is not None:
            await cp(entry, path2)
        else:
            await self.fs._cp_file(entry["name"], path2)

    async def mv_file(self, entry: Entry, path2: StoragePath) -> None:
        if (mv := getattr(self.fs, "_mv_file_entry", None)) is not None:
            await mv(entry, path2)
        else:
            await self.fs._mv_file(entry["name"], path2)

    async def rm_files(self, entries: list[Entry]) -> Mapping[str, Exception] | None:
        return await self.fs._rm_files([e["name"] for e in entries])  # type: ignore[no-any-return]

    async def rm_tree(self, entry: Entry) -> None:
        await self.fs._rm_tree(entry["name"])

    async def mv_tree(self, entry: Entry, path2: StoragePath) -> None:
        await self.fs._mv_tree(entry["name"], path2)

    async def cp_tree(self, entry: Entry, path2: StoragePath) -> None:
        await self.fs._cp_tree(entry["name"], path2)


def _handler_of(fs: Any) -> StorageHandler:
    """``fs`` if a handler, else the filesystem's handler, made once and kept on it.

    One handler per filesystem, so that a copy within one storage is given the
    same handler twice, as its source and its destination.
    """
    if isinstance(fs, StorageHandler):
        return fs
    handler = fs.__dict__.get("_alt_strategies_handler")
    if handler is None:
        handler = _FileSystemHandler(fs)
        fs.__dict__["_alt_strategies_handler"] = handler
    return handler  # type: ignore[no-any-return]


def _joined(directory: str, name: str) -> StoragePath:
    """The path of a single name in a directory, raising ValueError for a name that is no single name."""
    if not name or name in (".", "..") or "/" in name:
        raise ValueError(f"Not a name in a directory: {name!r}")
    if not directory:
        return StoragePath(name)
    if directory.endswith("/"):
        return StoragePath(directory + name)
    else:
        return StoragePath(f"{directory}/{name}")


async def _profile(fs: Any) -> StorageProfile:
    """The profile of ``fs``'s handler, opened first."""
    handler = _handler_of(fs)
    await handler.open()
    return handler.profile


def _limit(concurrency: int | asyncio.Semaphore, name: str) -> asyncio.Semaphore:
    """The semaphore of a concurrency: the one given, shared, or a new one of the number given."""
    if isinstance(concurrency, asyncio.Semaphore):
        return concurrency
    if concurrency < 1:
        raise ValueError(f"{name} must be at least 1: {concurrency}")
    return asyncio.Semaphore(concurrency)
