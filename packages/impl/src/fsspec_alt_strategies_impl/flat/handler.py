# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The handler of a flat storage, such as an object store: its primitives, and the flat strategies on them.

`FlatFileOperationHandler` adds to `FileOperationHandler` the recursive
listing by prefix, which the flat strategies list with, and the deletion of
keys in batches. The flat strategies take it, and it runs them as the
methods of `Strategy`: what a filesystem takes.

This module depends on `cp`, `du`, `find`, `glob`, `mv`, `rm` and `walk`, and
`handler` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from ..common.handler import FileOperationHandler
from .cp import _flat_cp
from .du import _flat_du
from .find import _flat_find
from .glob import _flat_glob
from .mv import _flat_mv
from .rm import _flat_rm
from .walk import _flat_walk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence

    from fsspec_alt_strategies.states import (
        CpState,
        DuState,
        FindState,
        GlobState,
        MvState,
        RmState,
        WalkState,
    )

    from ..common.handler import Entry
    from ..common.jobs import _Transfer
    from ..common.operation import Operation

__all__ = ["FlatFileOperationHandler"]


class FlatFileOperationHandler(FileOperationHandler):
    """A storage's primitives, as the flat strategies call them, and the flat strategies on them.

    It suits flat storages, such as object stores, which list every key under
    a prefix in one paginated request. Its strategies list with that listing,
    and act on each key as it arrives. With the profile's
    ``renames_files=True``, a move renames each key. Otherwise it copies each
    key, and deletes them in batches.

    A subclass implements the storage's primitives, `StorageHandler`'s, and
    ``ils_recursive``, and ``rm_files``, which deletes keys in batches of
    ``rm_batch_size``. It may override a strategy, e.g. with one of its own.

    Parameters
    ----------
    operation_concurrency, io_concurrency
        As `StorageHandler` takes them.

    See Also
    --------
    TreeFileOperationHandler : The same, for storages with a hierarchical namespace.
    BaseFileSystem : The filesystem on it.
    """

    rm_batch_size: int = 1000
    """The most keys `rm_files` takes at once, e.g. 1000 as S3's DeleteObjects takes."""

    @abstractmethod
    async def rm_files(self, entries: list[Entry]) -> Mapping[str, Exception] | None:
        """Delete keys, by one request, at most ``rm_batch_size`` of them.

        It returns the keys it could not delete, each with its error, and
        raises only if the request failed whole. A batch may span the storage's
        partitions, e.g. buckets: a storage whose request takes one partition
        sends one per partition.
        """

    @abstractmethod
    def ils_recursive(
        self, entry: Entry, *, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[Entry]:
        """List the keys below a directory whose names below it start with ``prefix``, in name order.

        It yields the keys' entries, sorted by the whole name, a page at a
        time. A directory's own marker, if the storage keeps one, comes as an
        entry whose name ends with "/".
        """

    # The flat strategies, as the methods of Strategy.

    def walk(
        self,
        path: str,
        *,
        maxdepth: int | None = None,
        topdown: bool = True,
        descend: Callable[[dict[str, Any]], bool] | None = None,
        on_error: Literal["omit", "raise"]
        | Callable[[Exception], object]
        | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState]:
        """`_flat_walk` of this handler."""
        return _flat_walk(
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
        path: str,
        *,
        maxdepth: int | None = None,
        withdirs: bool = False,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], FindState]:
        """`_flat_find` of this handler."""
        return _flat_find(
            self, path, maxdepth=maxdepth, withdirs=withdirs, ils_kwargs=ils_kwargs
        )

    def glob(
        self,
        path: str | Sequence[str],
        *,
        maxdepth: int | None = None,
        withdirs: bool = True,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], GlobState]:
        """`_flat_glob` of this handler."""
        return _flat_glob(
            self, path, maxdepth=maxdepth, withdirs=withdirs, ils_kwargs=ils_kwargs
        )

    def du(
        self,
        path: str,
        *,
        maxdepth: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], DuState]:
        """`_flat_du` of this handler."""
        return _flat_du(self, path, maxdepth=maxdepth, ils_kwargs=ils_kwargs)

    def rm(
        self,
        path: str | Sequence[str],
        *,
        maxdepth: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[str, RmState]:
        """`_flat_rm` of this handler."""
        return _flat_rm(self, path, maxdepth=maxdepth, ils_kwargs=ils_kwargs)

    def cp(
        self,
        dst_fs: Any,
        path1: str | Sequence[str],
        path2: str,
        *,
        cp_entry: _Transfer | None = None,
        same_storage: bool | None = None,
        maxdepth: int | None = None,
        root_entry: dict[str, Any] | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[tuple[str, str], CpState]:
        """`_flat_cp` from this handler."""
        return _flat_cp(
            self,
            dst_fs,
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
        dst_fs: Any,
        path1: str | Sequence[str],
        path2: str,
        *,
        cp_entry: _Transfer | None = None,
        mv_entry: _Transfer | None = None,
        same_storage: bool | None = None,
        maxdepth: int | None = None,
        root_entry: dict[str, Any] | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[tuple[str, str], MvState]:
        """`_flat_mv` from this handler."""
        return _flat_mv(
            self,
            dst_fs,
            path1,
            path2,
            cp_entry=cp_entry,
            mv_entry=mv_entry,
            same_storage=same_storage,
            maxdepth=maxdepth,
            root_entry=root_entry,
            ils_kwargs=ils_kwargs,
        )
