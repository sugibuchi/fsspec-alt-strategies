# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The strategies a handler runs on itself, for a filesystem's recursive methods.

`Strategy` declares one method per strategy: ``walk``, ``find``, ``glob``,
``du``, ``rm``, ``cp`` and ``mv``. Each takes a strategy's arguments, but the
handler it works on, which is ``self``. A family's handler implements them by
its family's strategies, e.g. `TreeFileOperationHandler`'s ``rm`` by
``_tree_rm``. A storage's handler overrides one to choose another, e.g. its
own.

`handler.FileOperationHandler` is a `StorageHandler` with them: what
`BaseFileSystem` takes. Each method returns an operation, not started, which
the filesystem runs, watched by the monitors. The filesystem keeps its
handler private, so that no caller of a filesystem gets an operation.

This module depends on `jobs` and `operation`, for their types alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

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

    from .handler import Entry, StorageHandler
    from .jobs import _Transfer
    from .operation import Operation

__all__ = ["Strategy"]


class Strategy(ABC):
    """The strategies of a handler, one method each, run on the handler itself.

    Each takes the arguments of a family's strategy, e.g. ``_tree_walk``'s,
    without the handler it works on: the handler is ``self``. A copy or move
    takes the destination's handler, ``dst``, ``self`` again within one
    storage. Each returns an operation, not started: the filesystem runs it.

    Paths come as the filesystem verified them, without a protocol. Each
    method raises argument errors, e.g. ValueError for ``maxdepth`` below 1,
    when called. The operation yields what it did, mixed with an error per
    failed path, as its strategy says.

    See Also
    --------
    TreeFileOperationHandler : These, by the tree strategies.
    FlatFileOperationHandler : These, by the flat strategies.
    """

    @abstractmethod
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
        """Walk a directory tree, as fsspec's ``walk`` does with ``detail=True``.

        Parameters
        ----------
        path : str
            The directory the walk starts from.
        maxdepth : int, optional
            The levels walked: 1 yields only ``path``.
        topdown : bool, default True
            Whether a directory comes before those below it.
        descend : callable, optional
            Called with each subdirectory's entry: the walk enters it only if
            it returns True.
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
        path: str,
        *,
        maxdepth: int | None = None,
        withdirs: bool = False,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], FindState]:
        """Find what is below a path, as fsspec's ``_find`` finds it.

        Parameters
        ----------
        path : str
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
        path: str | Sequence[str],
        *,
        maxdepth: int | None = None,
        withdirs: bool = True,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], GlobState]:
        """Find the paths matching glob patterns, as fsspec's ``_glob`` matches them.

        Parameters
        ----------
        path : str or sequence of str
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
        path: str,
        *,
        maxdepth: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[dict[str, Any], DuState]:
        """Measure the files below a path, as fsspec's ``_du`` does.

        Parameters
        ----------
        path : str
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
        path: str | Sequence[str],
        *,
        maxdepth: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[str, RmState]:
        """Delete a directory and everything in it, as fsspec's ``_rm`` does when recursive.

        Parameters
        ----------
        path : str or sequence of str
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

    @abstractmethod
    def rm_files(self, entries: list[Entry]) -> Operation[str, RmState]:
        """Delete files the caller found, e.g. a batch of a recursive ``rm``'s.

        The recursive deletions and moves send their files to it, in batches
        of the handler's ``rm_batch_size``. A family's handler deletes each by
        ``rm_file``, as many at once as ``operation_limit`` allows. A storage
        deleting several files by one request overrides it, e.g. with
        `rm_files._rm_by_request` and S3's DeleteObjects.

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

    @abstractmethod
    def cp(
        self,
        dst: StorageHandler,
        path1: str | Sequence[str],
        path2: str,
        *,
        cp_entry: _Transfer | None = None,
        same_storage: bool | None = None,
        maxdepth: int | None = None,
        root_entry: dict[str, Any] | None = None,
        ils_kwargs: dict[str, Any] | None = None,
    ) -> Operation[tuple[str, str], CpState]:
        """Copy a directory and everything in it, as fsspec's ``copy`` does when recursive.

        Parameters
        ----------
        dst : FileOperationHandler
            The handler copied to: ``self`` within one storage.
        path1 : str or sequence of str
            A directory, a file, a glob pattern, or a list of them.
        path2 : str
            Where they go, placed as fsspec's ``copy`` places them.
        cp_entry : callable, optional
            Copies a file by its source entry: by default the source's
            ``cp_file`` within one storage, else a stream between the two.
        same_storage : bool, optional
            Whether ``dst`` is ``self``, unless given.
        maxdepth : int, optional
            The levels copied below each path.
        root_entry : dict, optional
            The entry of ``path1``, as the caller found it.
        ils_kwargs : dict, optional
            The options of each of the source's listings.

        Returns
        -------
        Operation
            Yielding ``(source, destination)`` per file copied, or an error.
        """

    @abstractmethod
    def mv(
        self,
        dst: StorageHandler,
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
        """Move a directory and everything in it, as fsspec's ``mv`` does when recursive.

        Parameters
        ----------
        dst : FileOperationHandler
            The handler moved to: ``self`` within one storage.
        path1 : str or sequence of str
            A directory, a file, a glob pattern, or a list of them.
        path2 : str
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
        root_entry : dict, optional
            The entry of ``path1``, as the caller found it.
        ils_kwargs : dict, optional
            The options of each of the source's listings.

        Returns
        -------
        Operation
            Yielding ``(source, destination)`` per path moved, or an error.
        """
