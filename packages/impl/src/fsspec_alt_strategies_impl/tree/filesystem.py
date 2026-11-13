# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""An abstract async filesystem whose recursive methods are the tree strategies.

`TreeFileSystem` is for storages with a hierarchical namespace, which list by
directory. A subclass implements the storage's primitives, and gets fsspec's
recursive methods from the tree strategies.

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
from .cp import _tree_cp
from .du import _tree_du
from .find import _tree_find
from .glob import _tree_glob
from .listing import _recursive
from .mv import _tree_mv
from .rm import _tree_rm
from .walk import _tree_walk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable


__all__ = ["TreeFileSystem"]


class TreeFileSystem(_BaseFileSystem):
    """An async filesystem on a tree handler, its recursive methods the tree strategies.

    It suits storages with a hierarchical namespace, such as Azure Data Lake
    Storage Gen2. It implements fsspec's API on its handler, a `TreeHandler`:
    each primitive, e.g. ``_info`` or ``_rm_file``, calls the handler's, on the
    caller's path as the handler verifies it. It overrides fsspec's
    ``_walk``, ``_find``, ``_du``, ``_glob``, and ``_rm``, ``_copy`` and
    ``_mv`` when recursive, with the tree strategies, which take the handler.
    These list a whole tree by one recursive listing, and a tree they prune a
    directory per request, several at a time. They act on each part of a
    listing as it arrives. Their answers are fsspec's, and their failures are
    raised, a single one as it is, several in an ExceptionGroup.

    A directory that fails to list is yielded empty by ``_walk``, as fsspec's
    ``walk`` yields it. ``on_error`` handles every directory's failure, where
    fsspec's applies it to the root only, as its walk recurses without it.

    What the storage does is the handler's: its primitives, its profile, its
    limits, its files, and its paths: its ``protocol``, ``root`` and
    ``strip_protocol``. It takes a handler, e.g. one wrapping another
    filesystem, and is never cached. A storage's class below it takes no
    handler: it names its handler class, and nothing else but ``cachable``.
    It takes its paths from that class, and makes its handler from its
    storage options, fsspec's own aside. So fsspec makes it from a URL or a
    registry, and caches it.

    Parameters
    ----------
    handler : TreeHandler
        The storage's handler. A class below it takes the handler class's
        storage options instead.
    **kwargs
        Passed to `fsspec.asyn.AsyncFileSystem`, e.g. ``asynchronous=True``.

    Raises
    ------
    TypeError
        If a class below it names no ``handler_class``, or anything but it
        and ``cachable``, or is given a handler.

    See Also
    --------
    FlatFileSystem : The same, for flat storages, such as object stores.
    TreeHandler : What a storage implements.

    Examples
    --------
    >>> class MyFileSystem(TreeFileSystem):
    ...     handler_class = MyHandler  # Its protocol is "my".
    >>> fs = MyFileSystem(account="mine", operation_concurrency=32, io_concurrency=8)
    >>> local = TreeFileSystem(FsspecTreeHandler(fsspec.filesystem("file"), ...))
    """

    # The storage's primitives: the handler's, each on a caller's path, verified by it.

    def _ils(
        self, path: str, recursive: bool = False, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """List a directory, or all below it, as an async iterator of entries.

        The entries are as fsspec's ``_ls`` gives them with ``detail``. With
        ``recursive=True``, all below ``path`` come, depth first or in name
        order: by the handler's recursive listing, or else a directory per
        request.
        """
        handler = self.handler
        entry = handler.entry(self._strip_protocol(path))
        if recursive:
            return _recursive(handler, entry, **kwargs)
        else:
            return handler.ils(entry, **kwargs)

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Describe a path, raising FileNotFoundError if there is none."""
        handler = self.handler
        return await handler.info(handler.entry(self._strip_protocol(path)), **kwargs)

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        """Delete a file."""
        handler = self.handler
        await handler.rm_file(handler.entry(self._strip_protocol(path)))

    async def _rmdir(self, path: str) -> None:
        """Delete an empty directory."""
        handler = self.handler
        await handler.rmdir(handler.entry(self._strip_protocol(path)))

    async def _rm_tree(self, path: str) -> None:
        """Delete a directory with everything in it, in one request, by the handler's ``rm_tree``."""
        handler = self.handler
        await handler.rm_tree(handler.entry(self._strip_protocol(path)))  # type: ignore[attr-defined]

    async def _cp_file_entry(
        self, entry: dict[str, Any], path2: str, **kwargs: Any
    ) -> None:
        """Copy a file within the storage, by its entry, to ``path2``."""
        handler = self.handler
        await handler.cp_file(entry, handler.verified(self._strip_protocol(path2)))

    async def _mv_file_entry(
        self, entry: dict[str, Any], path2: str, **kwargs: Any
    ) -> None:
        """Move a file within the storage, by its entry, to ``path2``."""
        handler = self.handler
        await handler.mv_file(entry, handler.verified(self._strip_protocol(path2)))

    async def _mv_tree(self, path1: str, path2: str) -> None:
        """Move a directory with everything in it, in one request, by the handler's ``mv_tree``."""
        handler = self.handler
        entry = handler.entry(self._strip_protocol(path1))
        await handler.mv_tree(entry, handler.verified(self._strip_protocol(path2)))  # type: ignore[attr-defined]

    async def _cp_tree(self, path1: str, path2: str) -> None:
        """Copy a directory with everything in it, in one request, by the handler's ``cp_tree``."""
        handler = self.handler
        entry = handler.entry(self._strip_protocol(path1))
        await handler.cp_tree(entry, handler.verified(self._strip_protocol(path2)))  # type: ignore[attr-defined]

    async def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        """Create a directory and its parents, raising FileExistsError unless ``exist_ok``."""
        handler = self.handler
        await handler.makedirs(handler.verified(self._strip_protocol(path)), exist_ok)

    async def _mkdir(
        self, path: str, create_parents: bool = True, **kwargs: Any
    ) -> None:
        """Create a directory, as fsspec's local ``mkdir`` does, by ``_makedirs``.

        fsspec's own does nothing. An existing path raises FileExistsError.
        Without ``create_parents``, a missing parent raises FileNotFoundError,
        and one that is a file NotADirectoryError. With it, ``_makedirs``
        makes the parents, and answers for a file among them.
        """
        path = self._strip_protocol(path)
        if await self._exists(path):
            raise FileExistsError(path)
        parent = self._parent(path)
        if not create_parents and parent:
            try:
                kind = (await self._info(parent))["type"]
            except FileNotFoundError:
                raise FileNotFoundError(parent) from None
            if kind != "directory":
                raise NotADirectoryError(parent)
        await self._makedirs(path, exist_ok=True)

    # fsspec's recursive methods, answered by the tree strategies.

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
        walk = _tree_walk(
            self.handler,
            root,
            maxdepth=maxdepth,
            topdown=topdown,
            # The failed directories' errors, which fsspec's on_error then handles:
            # its "omit" still yields them empty.
            on_error="raise" if on_error == "raise" else None,
            ils_kwargs=kwargs,
        )

        async with aclosing(
            _with_monitors(walk, Call(self._unwrapped(), "walk", path))
        ) as items:
            try:
                first = await anext(items)
            except StopAsyncIteration:
                return
            except NotADirectoryError:
                # fsspec's walk of a file yields the file, under no name.
                entry = await self._info(root)
                yield (root, {}, {"": entry}) if detail else (root, [], [""])
                return
            except OSError as e:
                yield _failed_dir(root, e, on_error, detail)
                return

            item = first
            while True:
                if isinstance(item, Exception):
                    yield _failed_dir(item.directory, item, on_error, detail)  # type: ignore[attr-defined]
                else:
                    directory, dirs, files = item
                    if not detail:
                        dirs, files = list(dirs), list(files)
                    yield directory, dirs, files

                try:
                    item = await anext(items)
                except StopAsyncIteration:
                    return

    async def _find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        **kwargs: Any,
    ) -> Any:
        detail = kwargs.pop("detail", False)
        entries, _ = await _collect_with_monitors(
            _tree_find(
                self.handler,
                self._strip_protocol(path),
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

        op = _tree_du(
            self.handler,
            self._strip_protocol(path),
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
            _tree_glob(
                self.handler,
                _strip_only_protocol(self, path),
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
        if not recursive:
            await super()._rm(path, recursive, batch_size, maxdepth, **kwargs)
            return

        _, errors = await _collect_with_monitors(
            _tree_rm(self.handler, _strip_only_protocol(self, path), maxdepth=maxdepth),
            Call(self._unwrapped(), "rm", path),
            kwargs.get("callback"),
        )

        _raise(errors, "Deleting")

    # The copy and move that _BaseFileSystem's _copy and _mv start, as fsspec's methods.
    _cp_strategy = staticmethod(_tree_cp)
    _mv_strategy = staticmethod(_tree_mv)


TreeFileSystem._family = TreeFileSystem
