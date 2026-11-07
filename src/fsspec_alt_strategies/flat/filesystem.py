# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""An abstract async filesystem whose recursive methods are the flat strategies.

`FlatFileSystem` is for flat storages, such as object stores, which list every
key under a prefix in one paginated request. A subclass implements the
storage's primitives, that listing among them, and gets fsspec's recursive
methods from the flat strategies.

This module depends on `control`, `walk`, `find`, `du`, `glob`, `rm`, `cp`,
`mv`, and `filesystem` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from fsspec.asyn import sync_wrapper

from ..common.filesystem import _failed_folder, _FileSystem, _raise
from ..common.monitors import _collected
from ..common.paths import _triples
from ..control import Call
from .cp import flat_cp
from .du import flat_du
from .find import flat_find
from .glob import flat_glob
from .mv import flat_mv
from .rm import flat_rm
from .walk import flat_walk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    from ..profile import StorageProfile

__all__ = ["FlatFileSystem"]


class FlatFileSystem(_FileSystem):
    """An abstract async filesystem, its recursive methods the flat strategies.

    It suits flat storages, such as object stores, which list every key under a
    prefix in one paginated request. It overrides fsspec's ``_walk``,
    ``_find``, ``_du``, ``_glob``, and ``_rm``, ``_copy`` and ``mv`` when
    recursive, with `flat_walk`, `flat_find`, `flat_du`, `flat_glob`, `flat_rm`,
    `flat_cp` and `flat_mv`. Their answers are fsspec's, and their failures are
    raised, a single one as it is, several in an ExceptionGroup. ``_walk``
    holds the whole listing, to answer folder by folder, as fsspec's does.

    A subclass implements the storage's primitives: ``_ils``, the listing by
    prefix, which fsspec has not, and ``_rm_files``, a deletion of several
    files in one request, with ``rm_batch_size``. It implements those fsspec
    leaves abstract too: ``_ls``, for fsspec's own methods, ``_info``,
    ``_rm_file`` and ``_cp_file``. It implements ``open_async``, giving its
    files ``io_limit`` as their concurrency limit, and ``_open``. ``_cat_file``, ``_pipe_file``,
    ``_get_file`` and ``_put_file`` read and write through ``open_async``'s
    files.

    Two limits bound its requests, each an `asyncio.Semaphore` of its own:

    - ``operation_limit``, the strategies' requests, such as deletions and
      copies, but not their listings, which come one page after another;
    - ``io_limit``, the reads and writes of data, by its files.

    ``storage_profile``, a `StorageProfile`, tells the strategies what the
    storage does. With ``renames_files=True``, a recursive ``mv`` moves each
    key by ``_mv_file``, which a subclass then overrides with its rename.
    Otherwise it copies each key, and deletes them in batches.

    Parameters
    ----------
    operation_concurrency : int
        The requests of the strategies at a time.
    io_concurrency : int
        The reads and writes of data at a time.
    **kwargs
        Passed to `fsspec.asyn.AsyncFileSystem`, e.g. ``asynchronous=True``.

    Raises
    ------
    ValueError
        If a concurrency is less than 1, or ``rm_batch_size`` is not set.

    See Also
    --------
    TreeFileSystem : The same, for storages with a hierarchical namespace.

    Examples
    --------
    >>> class MyFileSystem(FlatFileSystem):
    ...     protocol = "my"
    ...     rm_batch_size = 1000  # As S3's DeleteObjects takes.
    ...
    ...     async def _ils(self, prefix): ...
    ...     async def _rm_files(self, paths): ...
    ...     async def _ls(self, path, detail=True, **kwargs): ...
    ...     async def _info(self, path, **kwargs): ...
    ...     async def _rm_file(self, path, **kwargs): ...
    ...     async def _cp_file(self, path1, path2, **kwargs): ...
    ...
    ...     async def open_async(self, path, mode="rb", **kwargs):
    ...         file = MyFile(self, path, mode, concurrency_limit=self.io_limit)
    ...         return AsyncFileAdapter(file)
    ...
    ...     def _open(self, path, mode="rb", **kwargs):
    ...         return MyFile(self, path, mode, concurrency_limit=self.io_limit)
    >>> fs = MyFileSystem(operation_concurrency=32, io_concurrency=8)
    """

    rm_batch_size: int
    """The most keys ``_rm_files`` deletes in one request, e.g. 1000 on S3."""

    storage_profile: StorageProfile | None = None
    """What the storage does, for the strategies. Unknown if None."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        size = getattr(self, "rm_batch_size", None)
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"rm_batch_size must be set, to at least 1: {size}")

    # What a subclass implements: the storage's primitives, the flat ones first.

    @abstractmethod
    def _ils(self, prefix: str) -> AsyncIterator[dict[str, Any]]:
        """List every key starting with ``prefix``, as an async iterator of entries.

        ``prefix`` is raw text, e.g. "data/2024-". The entries are as ``_ls``
        gives them, a key ending with "/" being a folder's marker, in any order.
        """

    @abstractmethod
    async def _rm_files(self, paths: list[str]) -> None:
        """Delete several files in one request, e.g. with S3's DeleteObjects.

        A key missing is no error. It raises if any is not deleted.
        """

    @abstractmethod
    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        """List a folder, as fsspec's own methods do: its entries, or their names."""

    @abstractmethod
    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Describe a path, raising FileNotFoundError if there is none."""

    @abstractmethod
    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        """Delete a file, as fsspec's ``_rm`` does when not recursive."""

    @abstractmethod
    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        """Copy a file within the storage."""

    @abstractmethod
    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        """Open a file whose methods are awaited, as fsspec's ``open_async`` does.

        The file holds ``io_limit`` as its concurrency limit, e.g. an
        `AsyncFileAdapter` of one of this package's file classes.
        """

    @abstractmethod
    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        """Open a file, as fsspec's ``_open`` does, for the sync ``open``."""

    async def _walk(
        self,
        path: str,
        maxdepth: int | None = None,
        topdown: bool = True,
        on_error: str | Callable[[Exception], Any] = "omit",
        **kwargs: Any,
    ) -> AsyncGenerator[tuple[str, Any, Any]]:
        detail = kwargs.pop("detail", False)
        root = self._strip_protocol(path)
        try:
            entries, _ = await _collected(
                flat_walk(
                    root,
                    ils=self._ils,
                    info=self._info,
                    maxdepth=maxdepth,
                    concurrency_limit=self.operation_limit,
                ),
                Call(self, "walk", path),
            )
        except NotADirectoryError:
            # fsspec's walk of a file yields the file, under no name.
            entry = await self._info(root)
            yield (root, {}, {"": entry}) if detail else (root, [], [""])
            return
        except OSError as e:
            yield _failed_folder(root, e, on_error, detail)
            return
        # Each folder's entries, the folders as deep as fsspec's walk lists.
        start = len(root) + 1 if root else 0
        folders: dict[str, list[dict[str, Any]]] = {root: []}
        for entry in entries:
            name = entry["name"].rstrip("/")
            depth = name[start:].count("/") + 1
            if entry["type"] == "directory" and (maxdepth is None or depth < maxdepth):
                folders.setdefault(name, [])
        for entry in entries:
            parent = entry["name"].rstrip("/").rpartition("/")[0]
            if parent in folders:
                folders[parent].append(entry)
        order = sorted(folders, key=lambda f: f.count("/"), reverse=not topdown)
        for folder in order:
            yield _triples(folder, folders[folder], detail)

    async def _find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        **kwargs: Any,
    ) -> Any:
        detail = kwargs.pop("detail", False)
        entries, _ = await _collected(
            flat_find(
                self._strip_protocol(path),
                ils=self._ils,
                info=self._info,
                maxdepth=maxdepth,
                withdirs=withdirs,
                concurrency_limit=self.operation_limit,
            ),
            Call(self, "find", path),
        )
        found = {e["name"].rstrip("/"): e for e in entries}
        names = sorted(found)
        return {n: found[n] for n in names} if detail else names

    async def _du(
        self, path: str, total: bool = True, maxdepth: int | None = None, **kwargs: Any
    ) -> Any:
        if kwargs.get("withdirs"):
            return await super()._du(path, total=total, maxdepth=maxdepth, **kwargs)
        op = flat_du(
            self._strip_protocol(path),
            ils=self._ils,
            info=self._info,
            maxdepth=maxdepth,
            concurrency_limit=self.operation_limit,
        )
        entries, _ = await _collected(op, Call(self, "du", path))
        if total:
            return op.state.size
        return {e["name"]: e.get("size") or 0 for e in entries}

    async def _glob(self, path: str, maxdepth: int | None = None, **kwargs: Any) -> Any:
        detail = kwargs.pop("detail", False)
        entries, _ = await _collected(
            flat_glob(
                self._plain(path),
                ils=self._ils,
                info=self._info,
                maxdepth=maxdepth,
                withdirs=kwargs.pop("withdirs", True),
                concurrency_limit=self.operation_limit,
            ),
            Call(self, "glob", path),
        )
        found = {e["name"].rstrip("/"): e for e in entries}
        names = sorted(found)
        return {n: found[n] for n in names} if detail else names

    async def _rm(
        self,
        path: str | list[str],
        recursive: bool = False,
        batch_size: int | None = None,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not recursive or maxdepth is not None:
            await super()._rm(path, recursive, batch_size, maxdepth, **kwargs)
            return
        _, errors = await _collected(
            flat_rm(
                self._plain(path),
                ils=self._ils,
                info=self._info,
                rm_files=self._rm_files,
                batch_size=self.rm_batch_size,
                concurrency_limit=self.operation_limit,
            ),
            Call(self, "rm", path),
            kwargs.get("callback"),
        )
        _raise(errors, "Deleting")

    async def _copy(
        self,
        path1: str | list[str],
        path2: str | list[str],
        recursive: bool = False,
        on_error: str | None = None,
        maxdepth: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not recursive or maxdepth is not None or not isinstance(path2, str):
            await super()._copy(
                path1, path2, recursive, on_error, maxdepth, batch_size, **kwargs
            )
            return
        _, errors = await _collected(
            flat_cp(
                self._plain(path1),
                self._plain(path2),
                ils=self._ils,
                info=self._info,
                cp_file=self._cp_file,
                concurrency_limit=self.operation_limit,
            ),
            Call(self, "copy", path1, path2),
            kwargs.get("callback"),
        )
        if on_error != "raise":
            # As fsspec's recursive copy: a source gone meanwhile is no error.
            errors = [e for e in errors if not isinstance(e, FileNotFoundError)]
        _raise(errors, "Copying")

    async def _mv(
        self,
        path1: str | list[str],
        path2: str,
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Move files, as fsspec's ``mv`` does: with `flat_mv` when recursive."""
        if path1 == path2:
            return
        if not recursive or maxdepth is not None:
            # fsspec's own: a copy, then the deletion of its sources.
            await self._copy(path1, path2, recursive, "raise", maxdepth, **kwargs)
            await self._rm(path1, recursive=recursive)
            return
        _, errors = await _collected(
            flat_mv(
                self._plain(path1),
                self._plain(path2),
                ils=self._ils,
                info=self._info,
                mv_file=self._mv_file,
                cp_file=self._cp_file,
                rm_files=self._rm_files,
                batch_size=self.rm_batch_size,
                profile=self.storage_profile,
                concurrency_limit=self.operation_limit,
            ),
            Call(self, "mv", path1, path2),
            kwargs.get("callback"),
        )
        _raise(errors, "Moving")

    # fsspec makes sync methods only from the async ones it knows: _mv is not.
    mv = sync_wrapper(_mv)
