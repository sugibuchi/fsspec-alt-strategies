# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A simulated async filesystem, to test and benchmark strategies.

Strategies win or lose by the requests they make, and by how many run at a time.
`SimulatedFileSystem` keeps its files in memory. It counts each request a real
storage would get, and delays each by a fixed latency, plus its bytes over a
bandwidth if one is set.

Its two modes model the two kinds of storage that strategies are chosen for:

- *flat*, an object store such as S3. Folders are only prefixes of keys, so a
  folder without files does not exist, and deleting a folder deletes each file.
- *hierarchical*, a storage with a hierarchical namespace such as Azure Data
  Lake Storage Gen2. Folders are real, and a folder is deleted in one request.

Whether a whole subtree can be listed in one paginated request is set apart, by
``recursive_listing``. Some hierarchical storages have it, others list one folder
per request.

It is a package of its own, apart from the strategies: copying them into
another package leaves it out. It depends on fsspec, and on the interface for
`StorageProfile`.
"""

from __future__ import annotations

import asyncio
import errno
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fsspec.asyn import AsyncFileSystem, _run_coros_in_chunks

from fsspec_alt_strategies.profile import StorageProfile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

__all__ = ["CallStats", "FlatView", "SimulatedFileSystem", "TreeView", "folder_ils"]


@dataclass
class CallStats:
    """The requests a `SimulatedFileSystem` received.

    Attributes
    ----------
    calls : collections.Counter
        The number of requests of each kind: "list", "info", "read", "write",
        "delete", "copy", "rename" and "mkdir". A listing counts a request per page.
    max_concurrency : int
        The most requests in flight at once.
    transferred : collections.Counter
        The bytes read or written by the requests of each kind: "read" and
        "write".
    """

    calls: Counter[str] = field(default_factory=Counter)
    max_concurrency: int = 0
    transferred: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        """The number of requests of all kinds."""
        return self.calls.total()


@dataclass
class _File:
    data: bytes
    mtime: float


def _parent_of(path: str) -> str:
    return path.rpartition("/")[0]


class SimulatedFileSystem(AsyncFileSystem):
    """An in-memory async filesystem that counts and delays its requests.

    Each instance has files of its own. Paths have no leading "/", and the root
    folder is "". The methods that a real storage would answer with requests,
    such as `_ls`, `_info` and `_rm_file`, count each request in `stats`.
    Setting up files with `add_file` and `add_dir` makes no request.

    Parameters
    ----------
    hierarchical : bool
        True for a storage with real folders, False for an object store.
        Only a hierarchical one keeps empty folders, and deletes a folder with
        everything in it in one request.
    recursive_listing : bool
        Whether `_find` lists a subtree in one paginated request, as object
        stores do. Otherwise, it walks the subtree a folder at a time.
    latency : float
        The seconds each request takes. Requests sent together overlap.
    bandwidth : float, optional
        The bytes per second each request reads or writes, as one connection
        does. A request then takes ``latency`` plus its bytes over this. Requests
        sent together each get it in full. No limit by default.
    page_size : int
        The most entries a listing request returns. Listing more takes a
        request per page, one after another.
    parents_required : bool
        Whether writing a file into a missing folder raises FileNotFoundError,
        rather than creating its parents. Only on a hierarchical storage.
    overwrite : bool
        Whether writing onto an existing file replaces it, rather than raising
        FileExistsError.
    **kwargs
        Passed to `fsspec.asyn.AsyncFileSystem`, e.g. ``asynchronous=True``.

    Attributes
    ----------
    stats : CallStats
        The requests received since the filesystem was made, or since
        `reset_stats`.

    Examples
    --------
    >>> fs = SimulatedFileSystem(hierarchical=True, latency=0.02)
    >>> fs.add_file("data/2024/a.parquet", b"...")
    >>> fs.rm("data", recursive=True)
    >>> fs.stats.calls["delete"]
    1
    """

    protocol = "simulated"
    root_marker = ""
    # Each instance has files of its own: never share one between callers.
    cachable = False

    def __init__(
        self,
        *,
        hierarchical: bool = False,
        recursive_listing: bool = True,
        latency: float = 0.0,
        bandwidth: float | None = None,
        page_size: int = 1000,
        parents_required: bool = False,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> None:
        if page_size < 1:
            raise ValueError(f"page_size must be positive: {page_size}")
        if bandwidth is not None and bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive: {bandwidth}")

        super().__init__(**kwargs)

        self.hierarchical = hierarchical
        self.recursive_listing = recursive_listing
        self.latency = latency
        self.bandwidth = bandwidth
        self.page_size = page_size
        self.parents_required = parents_required and hierarchical
        self.overwrite = overwrite

        self.stats = CallStats()
        self._in_flight = 0
        self._file_data: dict[str, _File] = {}
        # Each folder's direct children. In flat mode, a folder is here only
        # while it holds a file, as object stores have no folders of their own.
        self._folders: dict[str, set[str]] = {"": set()}

    @classmethod
    def _strip_protocol(cls, path: Any) -> Any:
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        return super()._strip_protocol(path).lstrip("/")

    # --- Setting up, without requests ---------------------------------------

    def add_file(
        self, path: str, data: bytes = b"", mtime: float | None = None
    ) -> None:
        """Add a file, with its parent folders, without a request.

        Parameters
        ----------
        path : str
            The file's path.
        data : bytes
            The file's contents.
        mtime : float, optional
            The file's modification time, as a POSIX timestamp. Defaults to now.
        """
        self._write(self._strip_protocol(path), data, mtime)

    def add_dir(self, path: str) -> None:
        """Add an empty folder, with its parent folders, without a request.

        Raises
        ------
        ValueError
            On a flat filesystem, which has no empty folders.
        """
        if not self.hierarchical:
            raise ValueError("A flat filesystem has no empty folders.")
        self._make_dirs(self._strip_protocol(path))

    def reset_stats(self) -> None:
        """Start counting requests again from zero."""
        self.stats = CallStats()

    # --- Requests -----------------------------------------------------------

    async def _request(self, kind: str, count: int = 1, size: int = 0) -> None:
        """Wait as a storage answering ``count`` requests one after another would.

        Each request reads or writes ``size`` bytes.
        """
        seconds = self.latency
        if self.bandwidth is not None:
            seconds += size / self.bandwidth

        for _ in range(count):
            self.stats.calls[kind] += 1
            self.stats.transferred[kind] += size
            self._in_flight += 1
            self.stats.max_concurrency = max(
                self.stats.max_concurrency, self._in_flight
            )
            try:
                # Even without latency, yield to the loop, so that requests sent
                # together overlap as they would on a real storage.
                await asyncio.sleep(seconds)
            finally:
                self._in_flight -= 1

    def _pages(self, entries: int) -> int:
        return max(1, math.ceil(entries / self.page_size))

    # Each method below sends its requests first, then reads or changes the
    # files without awaiting. So what it does is atomic, as a request is.

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        path = self._strip_protocol(path)
        await self._request("info")
        if path in self._file_data or path in self._folders:
            return self._entry(path)
        raise FileNotFoundError(path)

    async def _ls(
        self, path: str, detail: bool = True, **kwargs: Any
    ) -> list[dict[str, Any]] | list[str]:
        path = self._strip_protocol(path)
        children = self._folders.get(path)
        await self._request("list", self._pages(len(children)) if children else 1)

        if path in self._file_data:
            entries = [self._entry(path)]
        elif path in self._folders:
            entries = [self._entry(c) for c in sorted(self._folders[path])]
        else:
            raise FileNotFoundError(path)

        return entries if detail else [e["name"] for e in entries]

    async def _find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        **kwargs: Any,
    ) -> list[str] | dict[str, dict[str, Any]]:
        if not self.recursive_listing or maxdepth is not None:
            # fsspec's own walk, a folder per request.
            return await super()._find(
                path, maxdepth=maxdepth, withdirs=withdirs, **kwargs
            )

        path = self._strip_protocol(path)
        detail = kwargs.pop("detail", False)
        await self._request("list", self._pages(len(self._listing(path))))

        names = self._listing(path)
        if withdirs and path in self._folders:
            # Object stores list only files, but their folders follow from the
            # files' keys. fsspec's find includes the folder itself.
            names = [*self._descendants(path), *([path] if path else [])]
        elif not withdirs:
            names = [n for n in names if n in self._file_data]
        names.sort()

        if detail:
            return {n: self._entry(n) for n in names}
        return names

    async def _iexpand(self, prefix: str) -> AsyncIterator[dict[str, Any]]:
        """List the files whose paths start with ``prefix``, as object stores do.

        It is no fsspec method: the flat strategies list with it.
        ``prefix`` is a raw string, e.g. "data/2024-" lists "data/2024-01.csv".
        The files come in path order, a page per request: a page is asked for
        only once the one before it is all taken.
        """
        after: str | None = None

        while True:
            await self._request("list")
            # Asked for after the last path taken, as with S3's StartAfter, so
            # that files written meanwhile show as a real listing shows them.
            names = sorted(
                p
                for p in self._file_data
                if p.startswith(prefix) and (after is None or p > after)
            )
            page = names[: self.page_size]
            for name in page:
                yield self._entry(name)

            if len(names) <= self.page_size:
                return
            after = page[-1]

    async def _cat_file(
        self,
        path: str,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        path = self._strip_protocol(path)
        found = self._file_data.get(path)
        size = len(found.data[start:end]) if found is not None else 0
        await self._request("read", size=size)
        return self._file(path).data[start:end]

    async def _pipe_file(self, path: str, value: bytes, **kwargs: Any) -> None:
        path = self._strip_protocol(path)
        await self._request("write", size=len(value))
        self._check_writable(path)
        self._write(path, value)

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        path1, path2 = self._strip_protocol(path1), self._strip_protocol(path2)
        # A copy on the server: one request, whatever the file's size.
        await self._request("copy")
        data = self._file(path1).data
        self._check_writable(path2)
        self._write(path2, data)

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        path = self._strip_protocol(path)
        await self._request("delete")
        self._file(path)
        self._unlink(path)

    async def _rm_files(self, paths: list[str]) -> None:
        """Delete files in one request, as S3's DeleteObjects does.

        It is no fsspec method: the flat strategies delete with it. A path
        that is no file is left out, as object stores leave out a key that
        does not exist.
        """
        await self._request("delete")
        for path in map(self._strip_protocol, paths):
            if path in self._file_data:
                self._unlink(path)

    async def _rm(
        self,
        path: str | list[str],
        recursive: bool = False,
        maxdepth: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        batch_size = batch_size or self.batch_size

        if self.hierarchical:
            if maxdepth is not None:
                raise NotImplementedError(
                    "maxdepth: a folder is deleted in one request, all of it."
                )
            paths = await self._expand_path(path)
            coros = [self._rm_path(p, recursive) for p in paths]
        else:
            # As fsspec does: list everything, then delete each file. Folders
            # are only prefixes, so they go with their last file.
            paths = await self._expand_path(
                path, recursive=recursive, maxdepth=maxdepth
            )
            coros = [self._rm_file(p) for p in paths if p in self._file_data]

        await _run_coros_in_chunks(coros, batch_size=batch_size, nofiles=True)

    async def _rm_path(self, path: str, recursive: bool) -> None:
        if path in self._file_data:
            await self._rm_file(path)
        elif not recursive:
            await self._rmdir(path)
        else:
            await self._request("delete")
            self._folder(path)
            if not path:
                raise PermissionError("The root folder cannot be deleted.")
            self._drop(path)
            self._detach(path)

    async def _rmdir(self, path: str) -> None:
        path = self._strip_protocol(path)
        if not self.hierarchical:
            return  # No request: there is no folder to delete.

        await self._request("delete")
        if self._folder(path):
            raise OSError(errno.ENOTEMPTY, "Folder not empty", path)
        if not path:
            raise PermissionError("The root folder cannot be deleted.")

        del self._folders[path]
        self._detach(path)

    async def _rename(self, path1: str, path2: str) -> None:
        """Move a file, or a folder with all in it, in one request.

        It is no fsspec method: it is a hierarchical storage's own, as Azure Data
        Lake Storage Gen2 renames a path. `_tree_mv` takes it as its
        ``mv_file``, with a profile's ``renames_folders=True``. It refuses a
        ``path2`` that exists, and an object store has none.
        """
        path1, path2 = self._strip_protocol(path1), self._strip_protocol(path2)
        if not self.hierarchical:
            raise NotImplementedError("An object store renames nothing at once.")

        await self._request("rename")
        if path1 not in self._file_data and path1 not in self._folders:
            raise FileNotFoundError(path1)
        if path2 in self._file_data or path2 in self._folders:
            raise FileExistsError(path2)
        if not path1 or path2 == path1 or path2.startswith(path1 + "/"):
            raise OSError(errno.EINVAL, "Cannot move a folder into itself", path2)
        if self.parents_required and _parent_of(path2) not in self._folders:
            raise FileNotFoundError(_parent_of(path2))

        if path1 in self._file_data:
            file = self._file_data[path1]
            self._unlink(path1)
            self._write(path2, file.data, file.mtime)
            return

        # Parents first, so that each folder is made before what is in it.
        paths = [path1, *self._descendants(path1)]
        files = {p: self._file_data[p] for p in paths if p in self._file_data}
        folders = [p for p in paths if p in self._folders]

        self._drop(path1)
        self._detach(path1)

        for folder in folders:
            self._make_dirs(path2 + folder[len(path1) :])
        for path, file in files.items():
            self._write(path2 + path[len(path1) :], file.data, file.mtime)

    async def _mkdir(
        self, path: str, create_parents: bool = True, **kwargs: Any
    ) -> None:
        path = self._strip_protocol(path)
        if not self.hierarchical:
            return  # No request: a folder appears with its first file.

        await self._request("mkdir")
        if path in self._folders or path in self._file_data:
            raise FileExistsError(path)
        if not create_parents and _parent_of(path) not in self._folders:
            raise FileNotFoundError(_parent_of(path))

        self._make_dirs(path)

    async def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        path = self._strip_protocol(path)
        if not self.hierarchical:
            return  # No request: a folder appears with its first file.

        await self._request("mkdir")
        if path in self._folders and not exist_ok:
            raise FileExistsError(path)
        self._make_dirs(path)

    # --- The files, without requests ----------------------------------------

    def _entry(self, path: str) -> dict[str, Any]:
        file = self._file_data.get(path)
        if file is None:
            return {"name": path, "size": 0, "type": "directory"}
        return {
            "name": path,
            "size": len(file.data),
            "type": "file",
            "mtime": file.mtime,
        }

    def _file(self, path: str) -> _File:
        if path in self._file_data:
            return self._file_data[path]
        if path in self._folders:
            raise IsADirectoryError(path)
        raise FileNotFoundError(path)

    def _folder(self, path: str) -> set[str]:
        if path in self._folders:
            return self._folders[path]
        if path in self._file_data:
            raise NotADirectoryError(path)
        raise FileNotFoundError(path)

    def _descendants(self, path: str) -> Iterator[str]:
        for child in self._folders[path]:
            yield child
            if child in self._folders:
                yield from self._descendants(child)

    def _listing(self, path: str) -> list[str]:
        """What a recursive listing of ``path`` returns, before filtering."""
        if path in self._file_data:
            return [path]
        if path not in self._folders:
            return []

        # A hierarchical storage lists its folders too; an object store only keys.
        return [
            p
            for p in self._descendants(path)
            if self.hierarchical or p in self._file_data
        ]

    def _check_writable(self, path: str) -> None:
        """Raise as a storage would, if its settings forbid writing ``path``."""
        if self.parents_required and _parent_of(path) not in self._folders:
            raise FileNotFoundError(_parent_of(path))
        if not self.overwrite and path in self._file_data:
            raise FileExistsError(path)

    def _make_dirs(self, path: str) -> None:
        if path in self._folders:
            return
        if path in self._file_data:
            raise NotADirectoryError(path)

        parent = _parent_of(path)
        self._make_dirs(parent)
        self._folders[path] = set()
        self._folders[parent].add(path)

    def _write(self, path: str, data: bytes, mtime: float | None = None) -> None:
        if path in self._folders:
            raise IsADirectoryError(path)

        parent = _parent_of(path)
        self._make_dirs(parent)
        self._file_data[path] = _File(
            bytes(data), time.time() if mtime is None else mtime
        )
        self._folders[parent].add(path)

    def _unlink(self, path: str) -> None:
        del self._file_data[path]
        self._detach(path)

    def _drop(self, path: str) -> None:
        """Forget a folder and everything in it, but not its entry in its parent."""
        for child in self._folders.pop(path):
            if child in self._file_data:
                del self._file_data[child]
            else:
                self._drop(child)

    def _detach(self, path: str) -> None:
        parent = _parent_of(path)
        self._folders[parent].discard(path)
        # Without folders of their own, a folder is gone with its last file.
        if not self.hierarchical and parent and not self._folders[parent]:
            del self._folders[parent]
            self._detach(parent)


def folder_ils(fs: Any) -> Callable[[str], AsyncIterator[dict[str, Any]]]:
    """A tree strategy's ``ils``, listing a folder by ``fs._ls``, or by ``fs`` itself.

    A `SimulatedFileSystem`'s own ``_iexpand`` lists by prefix, as the flat
    strategies take it. This one lists one folder, as the tree strategies take
    it. ``fs._ls`` is looked up at each call, so that a test may replace it.
    """

    async def ils(path: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        ls = getattr(fs, "_ls", fs)
        for entry in await ls(path, detail=True, **kwargs):
            yield entry

    return ils


class TreeView:
    """A filesystem as the tree strategies take it, its primitives replaceable.

    The tree strategies take a filesystem, and call its ``_ils``, ``_info``,
    ``_rm_file``, ``_rmdir``, ``_cp_file``, ``_mv_file``, ``_makedirs`` and
    ``_rm_tree``. A `SimulatedFileSystem` has no ``_ils``: this view lists
    with `folder_ils`. Each other primitive is the filesystem's own, unless
    given, so that a test may replace one, e.g. with one that fails.

    ``operation_limit`` and ``storage_profile`` are a filesystem's: 32
    requests at a time, and a storage unknown.
    """

    def __init__(
        self,
        fs: Any,
        *,
        ils: Any = None,
        info: Any = None,
        rm_file: Any = None,
        rm_folder: Any = None,
        rm_tree: Any = None,
        cp_file: Any = None,
        mv_file: Any = None,
        makedirs: Any = None,
        storage_profile: StorageProfile | None = None,
        operation_limit: asyncio.Semaphore | None = None,
    ) -> None:
        self.fs = fs
        self._ils = folder_ils(fs) if ils is None else ils
        self._info = fs._info if info is None else info
        self._rm_file = fs._rm_file if rm_file is None else rm_file
        self._rmdir = fs._rmdir if rm_folder is None else rm_folder
        self._rm_tree = rm_tree
        self._cp_file = fs._cp_file if cp_file is None else cp_file
        self._mv_file = fs._mv_file if mv_file is None else mv_file
        self._makedirs = fs._makedirs if makedirs is None else makedirs
        self.storage_profile = storage_profile or StorageProfile()
        self.operation_limit = operation_limit or asyncio.Semaphore(32)


class FlatView:
    """A filesystem as the flat strategies take it, its primitives replaceable.

    The flat strategies take a filesystem, and call its ``_iexpand``,
    ``_info``, ``_rm_files``, ``_cp_file``, ``_mv_file``, ``_makedirs`` and
    ``_rm_file``. Each is the filesystem's own, unless given, so that a test
    may replace one, e.g. with one that fails. ``fs`` may be None when all
    that a test calls is given.

    ``operation_limit``, ``storage_profile`` and ``rm_batch_size`` are a
    filesystem's: 32 requests at a time, a storage unknown, and 1000 keys
    deleted in one request.
    """

    def __init__(
        self,
        fs: Any,
        *,
        iexpand: Any = None,
        info: Any = None,
        rm_files: Any = None,
        cp_file: Any = None,
        mv_file: Any = None,
        makedirs: Any = None,
        rm_file: Any = None,
        storage_profile: StorageProfile | None = None,
        operation_limit: asyncio.Semaphore | None = None,
        rm_batch_size: int = 1000,
    ) -> None:
        def own(name: str, given: Any) -> Any:
            return getattr(fs, name, None) if given is None else given

        self.fs = fs
        self._iexpand = own("_iexpand", iexpand)
        self._info = own("_info", info)
        self._rm_files = own("_rm_files", rm_files)
        self._cp_file = own("_cp_file", cp_file)
        self._mv_file = own("_mv_file", mv_file)
        self._makedirs = own("_makedirs", makedirs)
        self._rm_file = own("_rm_file", rm_file)
        self.storage_profile = storage_profile or StorageProfile()
        self.operation_limit = operation_limit or asyncio.Semaphore(32)
        self.rm_batch_size = rm_batch_size
