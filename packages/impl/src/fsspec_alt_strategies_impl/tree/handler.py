# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The handler of a storage with a hierarchical namespace: its primitives, and the tree strategies on them.

`TreeFileOperationHandler` adds to `FileOperationHandler` a recursive
listing, and the requests on a directory whole, each optional and declared by
the profile. The tree strategies take it, and it runs them as the methods of
`Strategy`: what a filesystem takes.

This module depends on `cp`, `du`, `find`, `glob`, `mv`, `rm` and `walk`, and
`cp_files`, `handler`, `paths` and `rm_files` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal

from ..common.handler import FileOperationHandler
from ..common.paths import _trim
from .cp import _tree_cp
from .du import _tree_du
from .find import _tree_find
from .glob import _tree_glob
from .mv import _tree_mv
from .rm import _tree_rm
from .walk import _tree_walk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence

    from fsspec_alt_strategies.states import (
        CpState,
        DuState,
        FindState,
        GlobState,
        MvState,
        RmState,
        WalkState,
    )

    from ..common.handler import Entry, StorageHandler, StoragePath
    from ..common.jobs import _Transfer
    from ..common.operation import Operation

__all__ = ["TreeFileOperationHandler"]


class TreeFileOperationHandler(FileOperationHandler):
    """A storage's primitives, as the tree strategies call them, and the tree strategies on them.

    It suits storages with a hierarchical namespace, such as Azure Data Lake
    Storage Gen2, a local disk or SFTP. Its strategies list a directory per
    request, several at a time, or a whole tree by one ``ils(...,
    recursive=True)`` where the profile declares its order. They act on each
    part of a listing as it arrives.

    A subclass implements the storage's primitives, `StorageHandler`'s. Its
    ``ils`` with ``recursive`` lists all below a directory: by default, a
    directory per request, which it reaches by ``super().ils``. It may
    implement too:

    - ``rm_tree``, a storage's recursive delete, declared by the profile's
      ``deletes_directories=True``;
    - ``mv_tree``, a storage's rename of a directory, declared by
      ``renames_directories=True``;
    - ``cp_tree``, a storage's copy of a directory, declared by
      ``copies_directories=True``.

    Each takes a file too, as a root may be either, and a missing root raises
    FileNotFoundError. A subclass may override a strategy, e.g. with one of
    its own.

    Parameters
    ----------
    operation_concurrency, io_concurrency
        As `StorageHandler` takes them.

    See Also
    --------
    FlatFileOperationHandler : The same, for flat storages, such as object stores.
    BaseFileSystem : The filesystem on it.
    """

    @abstractmethod
    def ils(
        self, entry: Entry, recursive: bool = False, **kwargs: Any
    ) -> AsyncIterator[Entry]:
        """List a directory, or all below it, at any depth: depth first where the profile's ``recursive_ls_depth_first_preorder`` says so.

        Without ``recursive``, it lists the directory's items, as
        `StorageHandler.ils` says. With ``recursive``:

        - Depth first: a directory's entry, then all below it, before any
          other entry. Its entries among themselves may come in any order.
        - By name: sorted by the whole name, as object stores list keys. A
          directory's entry may then come apart from what is below it, as
          "a/b-x" comes between "a/b" and "a/b/c". It counts as depth first.
        - Any order, unless the profile declares one of the above.

        A directory may be left out, as object stores list only files: the
        strategies tell it from the paths below it. A file lists itself. A
        missing directory raises FileNotFoundError, when the iteration starts.

        Here, with ``recursive``, it lists depth first, a directory per request
        by ``ils``, one at a time: a subclass listing a directory per request
        reaches it by ``super().ils(entry, recursive=True)``. The strategies
        never ask it of a handler declaring no
        ``recursive_ls_depth_first_preorder``: they list a directory per
        request themselves, many at once. ``kwargs`` go to each ``ils``.

        A storage listing a tree in one request answers ``recursive`` itself,
        and declares its order: the strategies then stream a subtree they need
        whole by it, but a walk top down. One whose storage may refuse that
        listing, as only asking tells, e.g. a WebDAV server refusing ``Depth:
        infinity``, or one whose recursive listing costs more than a directory
        per request, e.g. billed apart, falls back on ``super().ils`` itself.
        """
        if recursive:
            return _tree_ils(self, entry, **kwargs)
        else:
            raise NotImplementedError(f"{type(self).__name__} lists no directory")

    async def rm_tree(self, entry: Entry) -> None:
        """Delete a directory with everything in it, in one request."""
        raise NotImplementedError(f"{type(self).__name__} deletes no directory whole")

    async def mv_tree(
        self, entry: Entry, path2: StoragePath, *, parent: Entry | None = None
    ) -> None:
        """Move a directory with everything in it to ``path2``, in one request.

        The strategies call it only with ``path2`` missing and its parent
        existing, having made it. ``parent`` is that parent's entry, where the
        caller holds it, as the primitives making an item take it.
        """
        raise NotImplementedError(f"{type(self).__name__} moves no directory whole")

    async def cp_tree(
        self, entry: Entry, path2: StoragePath, *, parent: Entry | None = None
    ) -> None:
        """Copy a directory with everything in it to ``path2``, in one request, as `mv_tree` moves one."""
        raise NotImplementedError(f"{type(self).__name__} copies no directory whole")

    # The tree strategies, as the methods of Strategy.

    def walk(
        self,
        path: StoragePath,
        *,
        maxdepth: int | None = None,
        topdown: bool = True,
        on_error: Literal["omit", "raise"]
        | Callable[[Exception], object]
        | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState]:
        """`_tree_walk` of this handler."""
        return _tree_walk(
            self,
            path,
            maxdepth=maxdepth,
            topdown=topdown,
            on_error=on_error,
            ils_kwargs=ils_kwargs,
        )

    def find(
        self,
        path: StoragePath,
        *,
        maxdepth: int | None = None,
        withdirs: bool = False,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], FindState]:
        """`_tree_find` of this handler."""
        return _tree_find(
            self, path, maxdepth=maxdepth, withdirs=withdirs, ils_kwargs=ils_kwargs
        )

    def glob(
        self,
        path: StoragePath | Sequence[StoragePath],
        *,
        maxdepth: int | None = None,
        withdirs: bool = True,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], GlobState]:
        """`_tree_glob` of this handler."""
        return _tree_glob(
            self, path, maxdepth=maxdepth, withdirs=withdirs, ils_kwargs=ils_kwargs
        )

    def du(
        self,
        path: StoragePath,
        *,
        maxdepth: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], DuState]:
        """`_tree_du` of this handler."""
        return _tree_du(self, path, maxdepth=maxdepth, ils_kwargs=ils_kwargs)

    def rm(
        self,
        path: StoragePath | Sequence[StoragePath],
        *,
        maxdepth: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[str, RmState]:
        """`_tree_rm` of this handler."""
        return _tree_rm(self, path, maxdepth=maxdepth, ils_kwargs=ils_kwargs)

    def cp(
        self,
        dst: StorageHandler,
        path1: StoragePath | Sequence[StoragePath],
        path2: StoragePath,
        *,
        cp_entry: _Transfer | None = None,
        same_storage: bool | None = None,
        maxdepth: int | None = None,
        path1_entry: dict[str, Any] | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[tuple[str, str], CpState]:
        """`_tree_cp` from this handler."""
        return _tree_cp(
            self,
            dst,
            path1,
            path2,
            cp_entry=cp_entry,
            same_storage=same_storage,
            maxdepth=maxdepth,
            path1_entry=path1_entry,
            ils_kwargs=ils_kwargs,
        )

    def mv(
        self,
        dst: StorageHandler,
        path1: StoragePath | Sequence[StoragePath],
        path2: StoragePath,
        *,
        cp_entry: _Transfer | None = None,
        mv_entry: _Transfer | None = None,
        same_storage: bool | None = None,
        maxdepth: int | None = None,
        path1_entry: dict[str, Any] | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[tuple[str, str], MvState]:
        """`_tree_mv` from this handler."""
        return _tree_mv(
            self,
            dst,
            path1,
            path2,
            cp_entry=cp_entry,
            mv_entry=mv_entry,
            same_storage=same_storage,
            maxdepth=maxdepth,
            path1_entry=path1_entry,
            ils_kwargs=ils_kwargs,
        )


async def _tree_ils(
    handler: TreeFileOperationHandler, entry: Entry, **ils_kwargs: Any
) -> AsyncGenerator[dict[str, Any]]:
    """List all below a directory depth first, by one ``ils`` per directory, in turn: the default recursive ``ils``.

    It yields what a recursive ``ils`` yields: each entry below ``entry``, at
    any depth, depth first in preorder. A directory's entries come in the
    order its ``ils`` gives them, each subdirectory's followed by all below
    it. A file lists itself, as ``ils`` lists it. A directory listed twice,
    e.g. as a prefix and as its marker object, is yielded twice, but listed
    once.

    The strategies never ask it of a handler declaring no
    ``recursive_ls_depth_first_preorder``, as they list a directory per
    request themselves, many at once: it serves the other callers, e.g. a
    filesystem's recursive listing.

    A directory below ``entry`` that fails to list raises its error, as the
    storage raised it, and ends the listing: a recursive listing cannot go on
    past a failure. A missing ``entry`` raises FileNotFoundError, as its own
    ``ils`` raises it.

    ``entry`` is the directory's entry, as its parent's listing gave it, or as
    the handler's ``entry`` makes it from a path: it is listed by it whole, so
    that a storage addressing items by id asks nothing again. ``ils_kwargs`` go
    to each ``ils``. The caller holds the permits, as for any ``ils``.
    """
    directory = _trim(entry["name"])
    listed: set[str] = set()
    async for found in handler.ils(entry, **ils_kwargs):
        name = found["name"].rstrip("/")
        if name == directory:
            # A file lists itself. A directory may list its own marker: it is
            # no entry below it.
            if found["type"] != "directory":
                yield found
            continue
        yield found
        if found["type"] == "directory" and name not in listed:
            listed.add(name)
            subtree = _tree_ils(handler, {**found, "name": name}, **ils_kwargs)
            async with aclosing(subtree):  # Closed with this one.
                async for below in subtree:
                    yield below
