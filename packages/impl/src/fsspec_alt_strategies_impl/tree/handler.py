# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The handler of a storage with a hierarchical namespace: its primitives, and the tree strategies on them.

`TreeFileOperationHandler` adds to `FileOperationHandler` a recursive
listing, and the requests on a directory whole, each optional and declared by
the profile. The tree strategies take it, and it runs them as the methods of
`Strategy`: what a filesystem takes.

This module depends on `cp`, `du`, `find`, `glob`, `listing`, `mv`, `rm` and
`walk`, and `handler` and `rm_files` of `common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ..common.cp_files import _cp_each_file, _mv_each_file
from ..common.handler import FileOperationHandler
from ..common.rm_files import _rm_each_file
from .cp import _tree_cp
from .du import _tree_du
from .find import _tree_find
from .glob import _tree_glob
from .listing import _tree_ils
from .mv import _tree_mv
from .rm import _tree_rm
from .walk import _tree_walk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from fsspec_alt_strategies.states import (
        CpState,
        DuState,
        FindState,
        GlobState,
        MvState,
        RmState,
        WalkState,
    )

    from ..common.handler import Entry, FileTransfer, StorageHandler, StoragePath
    from ..common.jobs import _Transfer
    from ..common.operation import Operation

__all__ = ["TreeFileOperationHandler"]


class TreeFileOperationHandler(FileOperationHandler):
    """A storage's primitives, as the tree strategies call them, and the tree strategies on them.

    It suits storages with a hierarchical namespace, such as Azure Data Lake
    Storage Gen2, a local disk or SFTP. Its strategies list a whole tree by
    one ``ils_recursive``, and a tree they prune a directory per request,
    several at a time. They act on each part of a listing as it arrives.

    A subclass implements the storage's primitives, `StorageHandler`'s. It may
    implement too:

    - ``ils_recursive``, all below a directory in one listing, depth first or
      by name. By default, a directory per request, several at a time;
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

    def ils_recursive(self, entry: Entry, **kwargs: Any) -> AsyncIterator[Entry]:
        """List all below a directory, at any depth, depth first or by name.

        - Depth first: a directory's entry, then all below it, before any
          other entry. Its entries among themselves may come in any order.
        - By name: sorted by the whole name, as object stores list keys. A
          directory's entry may then come apart from what is below it, as
          "a/b-x" comes between "a/b" and "a/b/c".

        A directory may be left out, as object stores list only files: the
        strategies tell it from the paths below it. A file lists itself. A
        missing directory raises FileNotFoundError, when the iteration starts.

        Here, `listing._tree_ils` lists a directory per request by ``ils``,
        several at a time, depth first. A storage listing a tree in one
        request overrides it. One whose storage may refuse that listing, as
        only asking tells, e.g. a WebDAV server refusing ``Depth: infinity``,
        falls back on `listing._tree_ils` itself.
        """
        return _tree_ils(self, entry["name"], entry=entry, **kwargs)

    async def rm_tree(self, entry: Entry) -> None:
        """Delete a directory with everything in it, in one request."""
        raise NotImplementedError(f"{type(self).__name__} deletes no directory whole")

    async def mv_tree(self, entry: Entry, path2: StoragePath) -> None:
        """Move a directory with everything in it to ``path2``, in one request.

        The strategies call it only with ``path2`` missing and its parent
        existing, having made it.
        """
        raise NotImplementedError(f"{type(self).__name__} moves no directory whole")

    async def cp_tree(self, entry: Entry, path2: StoragePath) -> None:
        """Copy a directory with everything in it to ``path2``, in one request, as `mv_tree` moves one."""
        raise NotImplementedError(f"{type(self).__name__} copies no directory whole")

    # The tree strategies, as the methods of Strategy.

    def walk(
        self,
        path: StoragePath,
        *,
        maxdepth: int | None = None,
        topdown: bool = True,
        descend: Callable[[dict[str, Any]], bool] | None = None,
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
            descend=descend,
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

    def rm_files(self, entries: list[Entry]) -> Operation[str, RmState]:
        """`rm_files._rm_each_file` of this handler: each by ``rm_file``, as many at once as the limit allows."""
        return _rm_each_file(self, entries)

    def cp_files(
        self, transfers: list[FileTransfer]
    ) -> Operation[tuple[str, str], CpState]:
        """`cp_files._cp_each_file` of this handler: each by ``cp_file``, as many at once as the limit allows."""
        return _cp_each_file(self, transfers)

    def mv_files(
        self, transfers: list[FileTransfer]
    ) -> Operation[tuple[str, str], MvState]:
        """`cp_files._mv_each_file` of this handler: each by ``mv_file``, as many at once as the limit allows."""
        return _mv_each_file(self, transfers)

    def cp(
        self,
        dst: StorageHandler,
        path1: StoragePath | Sequence[StoragePath],
        path2: StoragePath,
        *,
        cp_entry: _Transfer | None = None,
        same_storage: bool | None = None,
        maxdepth: int | None = None,
        root_entry: dict[str, Any] | None = None,
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
            root_entry=root_entry,
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
        root_entry: dict[str, Any] | None = None,
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
            root_entry=root_entry,
            ils_kwargs=ils_kwargs,
        )
