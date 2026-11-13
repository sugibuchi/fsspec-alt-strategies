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
    _strip_only_protocol,
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
    limits, its batch of deletions and its files. A storage's class names its
    protocol and its handler class, and nothing else, as `TreeFileSystem`'s
    does. With the profile's ``renames_files=True``, a recursive ``mv``, or one
    of a single file, moves each key by the handler's rename. Otherwise it
    copies each key, and deletes them in batches.

    Parameters
    ----------
    *args, **storage_options
        Passed to the class's ``handler_class``, but fsspec's own options,
        e.g. ``asynchronous=True``, passed to `fsspec.asyn.AsyncFileSystem`.

    Raises
    ------
    TypeError
        As `TreeFileSystem` raises.
    ValueError
        If the handler's ``rm_batch_size`` is less than 1.

    See Also
    --------
    TreeFileSystem : The same, for storages with a hierarchical namespace.
    FlatHandler : What a storage implements.

    Examples
    --------
    >>> class MyFileSystem(FlatFileSystem):
    ...     protocol = "my"
    ...     handler_class = MyHandler
    >>> fs = MyFileSystem(bucket="mine", operation_concurrency=32, io_concurrency=8)
    """

    rm_batch_size: int
    """The most keys the handler's ``rm_files`` deletes in one request, e.g. 1000 on S3."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        size = self.handler.rm_batch_size  # type: ignore[attr-defined]
        if not isinstance(size, int) or size < 1:
            raise ValueError(f"rm_batch_size must be set, to at least 1: {size}")
        self.rm_batch_size = size

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
        handler = self._required_handler()
        entry = handler.entry(self._strip_protocol(path))
        if recursive:
            return handler.ils_recursive(entry, prefix=prefix, **kwargs)  # type: ignore[attr-defined, no-any-return]
        else:
            return handler.ils(entry, **kwargs)

    async def _rm_files(self, paths: list[str]) -> Mapping[str, Exception] | None:
        """Delete several files in one request, by the handler's ``rm_files``."""
        handler = self._required_handler()
        entries = [handler.entry(self._strip_protocol(p)) for p in paths]
        return await handler.rm_files(entries)  # type: ignore[attr-defined, no-any-return]

    async def _rmdir(self, path: str) -> None:
        """Delete an empty directory: on an object store, a bucket."""
        handler = self._required_handler()
        await handler.rmdir(handler.entry(self._strip_protocol(path)))

    async def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Create a directory and its parents: on an object store, a bucket, if missing."""
        handler = self._required_handler()
        await handler.makedirs(handler.verified(self._strip_protocol(path)), exist_ok)

    async def _mkdir(
        self, path: str, create_parents: bool = True, **kwargs: Any
    ) -> None:
        """Create a directory, as fsspec's ``_mkdir`` does: by ``_makedirs``."""
        await self._makedirs(path, exist_ok=False)

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Describe a path, raising FileNotFoundError if there is none."""
        handler = self._required_handler()
        return await handler.info(handler.entry(self._strip_protocol(path)), **kwargs)

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        """Delete a file, as fsspec's ``_rm`` does when not recursive."""
        handler = self._required_handler()
        await handler.rm_file(handler.entry(self._strip_protocol(path)))

    async def _cp_file_entry(
        self, entry: dict[str, Any], path2: str, **kwargs: Any
    ) -> None:
        """Copy a key within the storage, by its entry, to ``path2``."""
        handler = self._required_handler()
        await handler.cp_file(entry, handler.verified(self._strip_protocol(path2)))

    async def _mv_file_entry(
        self, entry: dict[str, Any], path2: str, **kwargs: Any
    ) -> None:
        """Move a key within the storage, by its entry, to ``path2``."""
        handler = self._required_handler()
        await handler.mv_file(entry, handler.verified(self._strip_protocol(path2)))

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
        root = self._strip_protocol(path)
        walk = _flat_walk(self, root, maxdepth=maxdepth, topdown=topdown)

        async with aclosing(
            _with_monitors(walk, Call(self._unwrapped(), "walk", path))
        ) as directories:
            try:
                async for item in directories:
                    directory, dirs, files = item
                    if not detail:
                        dirs, files = list(dirs), list(files)
                    yield directory, dirs, files
            except NotADirectoryError:
                # fsspec's walk of a file yields the file, under no name.
                entry = await self._info(root)
                yield (root, {}, {"": entry}) if detail else (root, [], [""])
            except OSError as e:
                # The listing's error ends the walk: fsspec's on_error handles
                # it as a failure to list the root.
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
                self,
                self._strip_protocol(path),
                maxdepth=maxdepth,
                withdirs=withdirs,
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

        op = _flat_du(self, self._strip_protocol(path), maxdepth=maxdepth)

        entries, _ = await _collect_with_monitors(
            op, Call(self._unwrapped(), "du", path)
        )
        return _du_answer(op.state.size, entries, total)

    async def _glob(self, path: str, maxdepth: int | None = None, **kwargs: Any) -> Any:
        detail = kwargs.pop("detail", False)
        entries, _ = await _collect_with_monitors(
            _flat_glob(
                self,
                _strip_only_protocol(self, path),
                maxdepth=maxdepth,
                withdirs=kwargs.pop("withdirs", True),
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
        if not recursive:
            await super()._rm(path, recursive, batch_size, maxdepth, **kwargs)
            return

        _, errors = await _collect_with_monitors(
            _flat_rm(self, _strip_only_protocol(self, path), maxdepth=maxdepth),
            Call(self._unwrapped(), "rm", path),
            kwargs.get("callback"),
        )
        _raise(errors, "Deleting")

    # The copy and move that _BaseFileSystem's _copy and _mv start, as fsspec's methods.
    _cp_strategy = staticmethod(_flat_cp)
    _mv_strategy = staticmethod(_flat_mv)


FlatFileSystem._family = FlatFileSystem
