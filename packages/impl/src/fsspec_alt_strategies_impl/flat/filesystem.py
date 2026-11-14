# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""An abstract async filesystem whose recursive methods are the flat strategies.

`FlatFileSystem` is for flat storages, such as object stores, which list every
key under a prefix in one paginated request. A subclass implements the
storage's primitives, that listing among them, and gets fsspec's recursive
methods from the flat strategies.

This module depends on `cp`, `du`, `find`, `glob`, `mv`, `rm`, `walk`, and
`filesystem` and `operation` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.monitor import Call

from ..common.filesystem import (
    _answer,
    _BaseFileSystem,
    _du_answer,
    _failed_dir,
    _raise,
)
from ..common.operation import _collect_with_monitors, _with_monitors
from .cp import _flat_cp
from .du import _flat_du
from .find import _flat_find
from .glob import _flat_glob
from .mv import _flat_mv
from .rm import _flat_rm
from .walk import _flat_walk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping

    from fsspec_alt_strategies.states import CpState, MvState

    from ..common.handler import Entry
    from ..common.operation import Operation


__all__ = ["FlatFileSystem"]


class FlatFileSystem(_BaseFileSystem):
    """An async filesystem on a flat handler, its recursive methods the flat strategies.

    It suits flat storages, such as object stores, which list every key under a
    prefix in one paginated request. It implements fsspec's API on its
    handler, a `FlatHandler`: each primitive, e.g. ``_info`` or ``_rm_file``,
    calls the handler's, on the caller's path as the handler verifies it. It
    overrides fsspec's ``_walk``, ``_find``, ``_du``, ``_glob``, and ``_rm``,
    ``_copy`` and ``_mv`` when recursive, with the flat strategies, which take
    the handler. These list every key under a prefix with its recursive
    listing, and act on each key as it arrives. Their answers are fsspec's,
    and their failures are raised, a single one as it is, several in an
    ExceptionGroup.

    What the storage does is the handler's: its primitives, its profile, its
    limits, its batch of deletions and its files. A storage's class below it
    makes its handler from its storage options, and names its paths, as one
    below `TreeFileSystem` does. With the profile's ``renames_files=True``, a recursive ``mv``, or one
    of a single file, moves each key by the handler's rename. Otherwise it
    copies each key, and deletes them in batches.

    Parameters
    ----------
    handler : FlatHandler
        The storage's handler.
    asynchronous, loop, use_listings_cache, listings_expiry_time, max_paths
        fsspec's own options, as `TreeFileSystem` takes them.

    Raises
    ------
    TypeError
        If given an option it does not name, e.g. one misspelled.

    See Also
    --------
    TreeFileSystem : The same, for storages with a hierarchical namespace.
    FlatHandler : What a storage implements.

    Examples
    --------
    >>> class MyFileSystem(FlatFileSystem):
    ...     protocol = MyHandler.protocol
    ...     _strip_protocol = MyHandler.strip_protocol
    ...
    ...     def __init__(self, bucket, **kwargs):
    ...         super().__init__(MyHandler(bucket), **kwargs)
    >>> fs = MyFileSystem("mine")
    >>> s3 = FlatFileSystemWrapper(S3fsHandler(s3fs.S3FileSystem(), ...))
    """

    # The storage's primitives: the handler's, each on a caller's path, verified by it.

    def _ils(
        self, path: str, recursive: bool = False, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """List a directory, or every key below it, as an async iterator of entries.

        Without ``recursive``, the handler's ``ils`` lists ``path``'s files
        and subdirectories. With it, its ``ils_recursive`` lists every key
        below ``path`` whose name below it starts with ``prefix``, in name
        order.
        """
        handler = self.handler
        entry = handler.entry(path)
        if recursive:
            return handler.ils_recursive(entry, prefix=prefix, **kwargs)  # type: ignore[attr-defined, no-any-return]
        else:
            return handler.ils(entry, **kwargs)

    async def _rm_files(self, paths: list[str]) -> Mapping[str, Exception] | None:
        """Delete several files in one request, by the handler's ``rm_files``."""
        try:
            handler = self.handler
            entries = [handler.entry(p) for p in paths]
            return await handler.rm_files(entries)  # type: ignore[attr-defined, no-any-return]
        finally:
            self._drop_listings()

    async def _rmdir(self, path: str) -> None:
        """Delete an empty directory: on an object store, a bucket."""
        try:
            handler = self.handler
            await handler.rmdir(handler.entry(path))
        finally:
            self._drop_listings()

    async def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Create a directory and its parents: on an object store, a bucket, if missing."""
        try:
            handler = self.handler
            await handler.makedirs(handler.verified(path), exist_ok)
        finally:
            self._drop_listings()

    async def _mkdir(
        self, path: str, create_parents: bool = True, **kwargs: Any
    ) -> None:
        """Create a directory, as fsspec's local ``mkdir`` does, by the handler's ``makedirs``.

        What exists is the handler's to tell: an existing bucket raises
        FileExistsError, as s3fs's does, while a prefix within one needs
        nothing made. Without ``create_parents``, a missing parent raises
        FileNotFoundError, and one that is a file NotADirectoryError, as
        `TreeFileSystem`'s ``_mkdir`` does: no bucket is made for a prefix.
        """
        try:
            handler = self.handler
            target = handler.verified(path)
            if not create_parents and target != handler.root:
                parent = handler.parent(target)
                if parent != handler.root:
                    try:
                        kind = (await handler.info(handler.entry(parent)))["type"]
                    except FileNotFoundError:
                        raise FileNotFoundError(parent) from None
                    if kind != "directory":
                        raise NotADirectoryError(parent)
            await handler.makedirs(target, exist_ok=False)
        finally:
            self._drop_listings()

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Describe a path, raising FileNotFoundError if there is none."""
        handler = self.handler
        return await handler.info(handler.entry(path), **kwargs)

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        """Delete a file, as fsspec's ``_rm`` does when not recursive."""
        try:
            handler = self.handler
            await handler.rm_file(handler.entry(path))
        finally:
            self._drop_listings()

    async def _cp_file_entry(
        self, entry: dict[str, Any], path2: str, **kwargs: Any
    ) -> None:
        """Copy a key within the storage, by its entry, to ``path2``."""
        try:
            handler = self.handler
            await handler.cp_file(entry, handler.verified(path2))
        finally:
            self._drop_listings()

    async def _mv_file_entry(
        self, entry: dict[str, Any], path2: str, **kwargs: Any
    ) -> None:
        """Move a key within the storage, by its entry, to ``path2``."""
        try:
            handler = self.handler
            await handler.mv_file(entry, handler.verified(path2))
        finally:
            self._drop_listings()

    # fsspec's recursive methods, answered by the flat strategies.

    async def _walk(
        self,
        path: str,
        maxdepth: int | None = None,
        topdown: bool = True,
        on_error: str | Callable[[Exception], Any] = "omit",
        **kwargs: Any,
    ) -> AsyncGenerator[tuple[str, Any, Any]]:
        detail = kwargs.pop("detail", False)
        root = self.handler.verified(path)
        walk = _flat_walk(
            self.handler,
            root,
            maxdepth=maxdepth,
            topdown=topdown,
            # The listing's error, which fsspec's on_error then handles as a
            # failure to list the root: its "omit" still yields the root empty.
            on_error="raise" if on_error == "raise" else None,
            ils_kwargs=kwargs,
        )

        async with aclosing(
            _with_monitors(walk, Call(self._unwrapped(), "walk", path))
        ) as directories:
            try:
                async for item in directories:
                    if isinstance(item, Exception):
                        yield _failed_dir(item.directory, item, on_error, detail)  # type: ignore[attr-defined]
                        continue

                    directory, dirs, files = item
                    if not detail:
                        dirs, files = list(dirs), list(files)
                    yield directory, dirs, files
            except NotADirectoryError:
                # fsspec's walk of a file yields the file, under no name.
                entry = await self._info(root)
                yield (root, {}, {"": entry}) if detail else (root, [], [""])
            except OSError as e:
                # A missing root, or the listing's error raised: fsspec's
                # on_error handles it as a failure to list the root.
                yield _failed_dir(root, e, on_error, detail)

    async def _find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        **kwargs: Any,
    ) -> Any:
        detail = kwargs.pop("detail", False)
        entries, _ = await _collect_with_monitors(
            _flat_find(
                self.handler,
                self.handler.verified(path),
                maxdepth=maxdepth,
                withdirs=withdirs,
                ils_kwargs=kwargs,
            ),
            Call(self._unwrapped(), "find", path),
        )
        return _answer(entries, detail)

    async def _du(
        self, path: str, total: bool = True, maxdepth: int | None = None, **kwargs: Any
    ) -> Any:
        if kwargs.pop("withdirs", False):
            return await super()._du(
                path, total=total, maxdepth=maxdepth, withdirs=True, **kwargs
            )

        op = _flat_du(
            self.handler,
            self.handler.verified(path),
            maxdepth=maxdepth,
            ils_kwargs=kwargs,
        )

        entries, _ = await _collect_with_monitors(
            op, Call(self._unwrapped(), "du", path)
        )
        return _du_answer(op.state.size, entries, total)

    async def _glob(self, path: str, maxdepth: int | None = None, **kwargs: Any) -> Any:
        detail = kwargs.pop("detail", False)
        withdirs = kwargs.pop("withdirs", True)
        entries, _ = await _collect_with_monitors(
            _flat_glob(
                self.handler,
                self.handler.verified(path, strip_only_protocol=True),
                maxdepth=maxdepth,
                withdirs=withdirs,
                ils_kwargs=kwargs,
            ),
            Call(self._unwrapped(), "glob", path),
        )
        return _answer(entries, detail)

    async def _rm(
        self,
        path: str | list[str],
        recursive: bool = False,
        batch_size: int | None = None,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            if not recursive:
                await super()._rm(path, recursive, batch_size, maxdepth, **kwargs)
                return

            _, errors = await _collect_with_monitors(
                _flat_rm(self.handler, self._as_given(path), maxdepth=maxdepth),
                Call(self._unwrapped(), "rm", path),
                kwargs.get("callback"),
            )
            _raise(errors, "Deleting")
        finally:
            self._drop_listings()

    # The copy and move that _BaseFileSystem's _copy and _mv start, as fsspec's methods.

    def _cp_operation(
        self,
        sources: str | list[str],
        target: str,
        *,
        maxdepth: int | None,
        root_entry: Entry | None,
    ) -> Operation[tuple[str, str], CpState]:
        """The flat copy within the storage: `_flat_cp`."""
        handler = self.handler
        return _flat_cp(
            handler, handler, sources, target, maxdepth=maxdepth, root_entry=root_entry
        )

    def _mv_operation(
        self,
        sources: str | list[str],
        target: str,
        *,
        maxdepth: int | None,
        root_entry: Entry | None,
    ) -> Operation[tuple[str, str], MvState]:
        """The flat move within the storage: `_flat_mv`."""
        handler = self.handler
        return _flat_mv(
            handler, handler, sources, target, maxdepth=maxdepth, root_entry=root_entry
        )
