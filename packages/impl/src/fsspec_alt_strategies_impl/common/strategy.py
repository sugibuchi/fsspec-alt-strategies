# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The strategies a handler runs on itself, for a filesystem's recursive methods.

`Strategy` declares one method per strategy: ``walk``, ``find``, ``glob``,
``du``, ``rm``, ``rm_files``, ``cp``, ``cp_files``, ``mv`` and ``mv_files``. Each takes a strategy's arguments, but the
handler it works on, which is ``self``. `TreeFileOperationHandler` implements
them by the tree strategies, e.g. its ``rm`` by ``_tree_rm``. A storage's handler overrides one to choose another, e.g. its
own.

`handler.FileOperationHandler` is a `StorageHandler` with them: what
`BaseFileSystem` takes. Each method returns an operation, not started, which
the filesystem runs, watched by the monitors. The filesystem keeps its
handler private, so that no caller of a filesystem gets an operation.

This module depends on `cp_files` and `rm_files`, for the batch strategies' defaults, and `handler`, `operation` and `transfer`, for their types alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, cast

from .cp_files import _cp_each_file, _mv_each_file
from .rm_files import _rm_each_file

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fsspec_alt_strategies.states import (
        CpState,
        DuState,
        FindState,
        GlobState,
        MvState,
        RmState,
        WalkState,
    )

    from .handler import Entry, FileTransfer, StorageHandler, StoragePath
    from .operation import Operation
    from .transfer import _Transfer

__all__ = ["Strategy"]


class Strategy(ABC):
    """The strategies of a handler, one method each, run on the handler itself.

    Each takes the arguments of its strategy, e.g. ``_tree_walk``'s,
    without the handler it works on: the handler is ``self``. A copy or move
    takes the destination's handler, ``dst``, ``self`` again within one
    storage. Each returns an operation, not started: the filesystem runs it.
    The batch strategies, `rm_files`, `cp_files` and `mv_files`, have
    defaults here, a request per file, which the tree handler keeps: a storage
    sending several by one request overrides them.

    Paths come as the filesystem verified them, without a protocol: each a
    `StoragePath`, a glob pattern too, by the handler's ``verified``. Each
    method raises argument errors, e.g. ValueError for ``maxdepth`` below 1,
    when called. The operation yields what it did, mixed with an error per
    failed path, as its strategy says.

    See Also
    --------
    TreeFileOperationHandler : These, by the tree strategies.
    """

    # How the strategies run in bulk: the most files each batch strategy takes
    # at once. They tell of the strategies, not of how the storage is
    # accessed: a handler's primitives each act on one item.

    rm_batch_size: int = 1
    """The most files the strategies give ``rm_files`` at once: 1 by default.

    By default, a file is deleted as soon as it is found. A storage deleting
    several files by one request overrides ``rm_files``, and declares its
    most here, e.g. 1000 keys as S3's DeleteObjects takes, or 256 blobs as
    Azure's Blob Batch: the files then wait for their batch to fill, or for
    the listing to end.
    """

    cp_batch_size: int = 1
    """The most files the strategies give ``cp_files`` at once: 1 by default.

    By default, a file is copied as soon as its destination's directory is
    ready. A storage copying several files by one request, e.g. the Hugging
    Face Hub's commit, overrides ``cp_files``, and declares its most here.
    The files then wait for their batch to fill, or for the walk to end with
    every directory made. Only a copy within one storage batches: between
    two, each file streams on its own.
    """

    mv_batch_size: int = 1
    """The most files the strategies give ``mv_files`` at once: 1 by default.

    As ``cp_batch_size``, for the files a move renames within one storage. A
    move copying then deleting batches its copies by ``cp_batch_size``, and
    its deletions by ``rm_batch_size``.
    """

    @abstractmethod
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
        """Walk a directory tree, as fsspec's ``walk`` does with ``detail=True``.

        Parameters
        ----------
        path : StoragePath
            The directory the walk starts from.
        maxdepth : int, optional
            The levels walked: 1 yields only ``path``.
        topdown : bool, default True
            Whether a directory comes before those below it.
        on_error : {"omit", "raise"} or callable, optional
            What becomes of the error of a directory that fails to list: None
            yields it.
        ils_kwargs : dict, optional
            The options of each listing.

        Returns
        -------
        Operation
            Yielding ``(path, dirs, files)`` per directory, or an error.
        """

    @abstractmethod
    def find(
        self,
        path: StoragePath,
        *,
        maxdepth: int | None = None,
        withdirs: bool = False,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], FindState]:
        """Find what is below a path, as fsspec's ``_find`` finds it.

        Parameters
        ----------
        path : StoragePath
            The directory searched: a file finds itself.
        maxdepth : int, optional
            The levels searched: 1 finds only what is in ``path``.
        withdirs : bool, default False
            Whether directories are found too, ``path`` among them.
        ils_kwargs : dict, optional
            The options of each listing.

        Returns
        -------
        Operation
            Yielding the entry of each path found, or an error.
        """

    @abstractmethod
    def glob(
        self,
        path: StoragePath | Sequence[StoragePath],
        *,
        maxdepth: int | None = None,
        withdirs: bool = True,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], GlobState]:
        """Find the paths matching glob patterns, as fsspec's ``_glob`` matches them.

        Parameters
        ----------
        path : StoragePath or sequence of StoragePath
            A pattern, or several: a trailing "/" matches only directories.
        maxdepth : int, optional
            How deep a ``**`` goes.
        withdirs : bool, default True
            Whether directories can match.
        ils_kwargs : dict, optional
            The options of each listing.

        Returns
        -------
        Operation
            Yielding the entry of each path matched, or an error.
        """

    @abstractmethod
    def du(
        self,
        path: StoragePath,
        *,
        maxdepth: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], DuState]:
        """Measure the files below a path, as fsspec's ``_du`` does.

        Parameters
        ----------
        path : StoragePath
            The directory measured: a file measures itself.
        maxdepth : int, optional
            The levels counted: 1 counts only the files in ``path``.
        ils_kwargs : dict, optional
            The options of each listing.

        Returns
        -------
        Operation
            Yielding the entry of each file counted, or an error. Its state's
            ``size`` is their total.
        """

    @abstractmethod
    def rm(
        self,
        path: StoragePath | Sequence[StoragePath],
        *,
        maxdepth: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[str, RmState]:
        """Delete a directory and everything in it, as fsspec's ``_rm`` does when recursive.

        Parameters
        ----------
        path : StoragePath or sequence of StoragePath
            A directory, a file, a glob pattern, or a list of them.
        maxdepth : int, optional
            The levels deleted below each path.
        ils_kwargs : dict, optional
            The options of each listing.

        Returns
        -------
        Operation
            Yielding each path deleted, or an error.
        """

    def rm_files(self, entries: list[Entry]) -> Operation[str, RmState]:
        """Delete files the caller found, e.g. a batch of a recursive ``rm``'s.

        The recursive deletions and moves send their files to it, in batches
        of the handler's ``rm_batch_size``. By default, a handler deletes each by
        ``rm_file``, as many at once as ``operation_limit`` allows. A storage
        deleting several files by one request overrides it, e.g. with
        `rm_files.rm_by_request` and S3's DeleteObjects.

        Parameters
        ----------
        entries : list of dict
            The entries of the files, as a listing gave them: at most
            ``rm_batch_size``. A missing file is deleted, as no error.

        Returns
        -------
        Operation
            Yielding each path deleted, or an error noting its path. It raises
            only an error failing all the files, e.g. a request deleting them
            all.
        """
        return _rm_each_file(cast("StorageHandler", self), entries)

    @abstractmethod
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
        """Copy a directory and everything in it, as fsspec's ``copy`` does when recursive.

        Parameters
        ----------
        dst : StorageHandler
            The handler copied to: ``self`` within one storage.
        path1 : StoragePath or sequence of StoragePath
            A directory, a file, a glob pattern, or a list of them.
        path2 : StoragePath
            Where they go, placed as fsspec's ``copy`` places them.
        cp_entry : callable, optional
            Copies a file by its source entry: by default the source's
            ``cp_file`` within one storage, else a stream between the two.
        same_storage : bool, optional
            Whether ``dst`` is ``self``, unless given.
        maxdepth : int, optional
            The levels copied below each path.
        path1_entry : dict, optional
            The entry of ``path1``, as the caller found it.
        ils_kwargs : dict, optional
            The options of each of the source's listings.

        Returns
        -------
        Operation
            Yielding ``(source, destination)`` per file copied, or an error.
        """

    def cp_files(
        self, transfers: list[FileTransfer]
    ) -> Operation[tuple[str, str], CpState]:
        """Copy files within this storage, e.g. a batch of a recursive ``cp``'s.

        The recursive copies within one storage send their files to it, in
        batches of the handler's ``cp_batch_size``, once each destination's
        directory is ready. By default, a handler copies each by ``cp_file``, as
        many at once as ``operation_limit`` allows. A storage copying several
        files by one request overrides it, e.g. with `cp_files.cp_by_request`
        and a commit.

        Parameters
        ----------
        transfers : list of FileTransfer
            The files, each with where it goes: at most ``cp_batch_size``.

        Returns
        -------
        Operation
            Yielding ``(source, destination)`` per file copied, or an error
            noting its paths, with its source's path as its ``source``
            attribute. A FileExistsError, a destination refusing to overwrite,
            is retried by the caller, replacing the file on its own. It raises
            only an error failing all the files, e.g. a request copying them
            all.
        """
        return _cp_each_file(cast("StorageHandler", self), transfers)

    @abstractmethod
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
        """Move a directory and everything in it, as fsspec's ``mv`` does when recursive.

        Parameters
        ----------
        dst : StorageHandler
            The handler moved to: ``self`` within one storage.
        path1 : StoragePath or sequence of StoragePath
            A directory, a file, a glob pattern, or a list of them.
        path2 : StoragePath
            Where they go, placed as fsspec's ``mv`` places them.
        cp_entry : callable, optional
            Copies a file by its source entry, where the move copies.
        mv_entry : callable, optional
            Moves a file by its source entry, where the move renames: by
            default the source's ``mv_file`` within one storage.
        same_storage : bool, optional
            Whether ``dst`` is ``self``, unless given.
        maxdepth : int, optional
            The levels moved below each path.
        path1_entry : dict, optional
            The entry of ``path1``, as the caller found it.
        ils_kwargs : dict, optional
            The options of each of the source's listings.

        Returns
        -------
        Operation
            Yielding ``(source, destination)`` per path moved, or an error.
        """

    def mv_files(
        self, transfers: list[FileTransfer]
    ) -> Operation[tuple[str, str], MvState]:
        """Move files within this storage, e.g. a batch of a recursive ``mv``'s.

        As `cp_files`, for the files a move renames, in batches of the
        handler's ``mv_batch_size``: each by ``mv_file`` by default. A file
        it could not move stays where it was.

        Parameters
        ----------
        transfers : list of FileTransfer
            The files, each with where it goes: at most ``mv_batch_size``.

        Returns
        -------
        Operation
            Yielding ``(source, destination)`` per file moved, or an error
            noting its paths, with its source's path as its ``source``
            attribute. It raises only an error failing all the files.
        """
        return _mv_each_file(cast("StorageHandler", self), transfers)


def _check_batch_size(batch_size: int, name: str) -> None:
    """Refuse a handler's batch size below 1, e.g. its ``rm_batch_size``, as no batch could be sent."""
    if batch_size < 1:
        raise ValueError(f"{name} must be at least 1: {batch_size}")
