# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The handler of a storage with a hierarchical namespace: its primitives, and the tree strategies on them.

`TreeFileOperationHandler` adds to `FileOperationHandler` a recursive
listing, and the requests on a directory whole, each optional and declared by
the profile. The tree strategies take it, and it runs them as the methods of
`Strategy`: what a filesystem takes.

This module depends on `cp`, `du`, `find`, `glob`, `listing`, `mv`, `rm` and
`walk`, and `cp_files`, `handler` and `rm_files` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from ..common.handler import FileOperationHandler
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

    from ..common.handler import Entry, StorageHandler, StoragePath
    from ..common.jobs import _Transfer
    from ..common.operation import Operation

__all__ = ["TreeFileOperationHandler"]


class TreeFileOperationHandler(FileOperationHandler):
    """A storage's primitives, as the tree strategies call them, and the tree strategies on them.

    It suits storages with a hierarchical namespace, such as Azure Data Lake
    Storage Gen2, a local disk or SFTP. Its strategies list a whole tree by
    one ``ils(..., recursive=True)``, and a tree they prune a directory per
    request, several at a time. They act on each part of a listing as it
    arrives.

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
        - Parents first, unless the profile declares one of the above: a
          directory's entry before what is below it, in any order otherwise.

        A directory may be left out, as object stores list only files: the
        strategies tell it from the paths below it. A file lists itself. A
        missing directory raises FileNotFoundError, when the iteration starts.

        Here, with ``recursive``, it lists depth first, a directory per request
        by ``ils``, several at a time: a subclass listing a directory per
        request reaches it by ``super().ils(entry, recursive=True)``. The walk
        recognises it, and reads such a subtree a directory at a time itself,
        so a handler keeping it declares no
        ``recursive_ls_depth_first_preorder``. ``kwargs`` go to each ``ils``.

        A storage listing a tree in one request answers ``recursive`` itself.
        One whose storage may refuse that listing, as only asking tells, e.g.
        a WebDAV server refusing ``Depth: infinity``, or one whose recursive
        listing costs more than a directory per request, e.g. billed apart,
        falls back on ``super().ils`` itself. The strategies always list a
        subtree they need whole by it, but a walk top down, which needs each
        directory before those below it.
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
