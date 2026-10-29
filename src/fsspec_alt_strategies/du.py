# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The size of a tree, from its listings alone.

`tree_du` is an alternative to fsspec's ``_du`` with ``total=True``. fsspec's
finds the files, then asks each file's size with a request of its own.
`tree_du` adds up the sizes the listings already give.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .control import Result, State, _finish, _report
from .walk import walk

if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem

    from .control import Callback

__all__ = ["DuResult", "DuState", "tree_du"]


@dataclass(frozen=True, kw_only=True)
class DuState(State):
    """The state of a `tree_du`, as its callback gets it.

    Attributes
    ----------
    folders_listed : int
        The folders listed so far.
    files : int
        The files counted so far.
    size : int
        Their size so far, in bytes.
    """

    folders_listed: int = 0
    files: int = 0
    size: int = 0


@dataclass(kw_only=True)
class DuResult(Result):
    """The size of a tree, as a `tree_du` measured it.

    Attributes
    ----------
    files : int
        The files counted.
    folders : int
        The folders listed, ``path`` and those that failed included.
    size : int
        The files' total size, in bytes. It is what fsspec's ``_du`` returns.
    """

    files: int = 0
    folders: int = 0
    size: int = 0


async def tree_du(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    batch_size: int | None = None,
    callback: Callback[DuState] | None = None,
) -> DuResult:
    """Measure a tree's size from its listings, listing several folders at a time.

    It assumes that ``_ls`` gives each file's size, as fsspec's filesystems do.
    A missing ``path`` measures nothing, as with fsspec's ``_du``. A file
    measures itself. A folder that fails to list is in ``errors``, and the
    rest of the tree is still measured.

    It wins over fsspec's ``_du`` on any storage, as it sends no ``_info`` per
    file. On object stores, which list a whole tree in one paginated request,
    summing the sizes from fsspec's ``_find`` with ``detail=True`` makes fewer
    requests still.

    For each file's size, as fsspec's ``_du`` gives with ``total=False``, use
    `tree_find`: each of its entries has its ``size``.

    Requests: one ``_ls`` per folder, at most ``batch_size`` at a time. One
    ``_info`` for ``path``, if a file.

    The callback is called after each folder listed. A stop returns the size
    measured so far. ``Stop(cleanup=True)`` undoes nothing, as nothing is
    changed.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path : str
        The folder to measure.
    maxdepth : int, optional
        The deepest level to measure: 1 counts only the files in ``path``. No
        limit by default.
    batch_size : int, optional
        The most listings at a time. fsspec's default if None.
    callback : callable, optional
        Called with a `DuState` after each step. It may return a `Stop`.

    Returns
    -------
    DuResult
        The size, the counts, and the folders that failed.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.
    OperationStopped
        If the callback returned ``Stop(raise_error=True)``.

    See Also
    --------
    walk : The walk under it.
    tree_find : The size of each file.

    Examples
    --------
    >>> result = await tree_du(fs, "data", batch_size=32)
    >>> result.size, result.files
    (73400320, 70)
    """
    root = fs._strip_protocol(path)
    result = DuResult()
    try:
        async with aclosing(
            walk(fs, root, maxdepth=maxdepth, batch_size=batch_size)
        ) as listings:
            async for listing in listings:
                result.folders += 1
                if listing.error is not None:
                    result.errors[listing.folder] = listing.error
                for entry in listing.entries:
                    if entry["type"] != "directory":
                        result.files += 1
                        result.size += entry.get("size") or 0
                state = DuState(
                    errors=len(result.errors),
                    current=listing.folder,
                    folders_listed=result.folders,
                    files=result.files,
                    size=result.size,
                )
                if (stop := await _report(callback, state)) is not None:
                    result.stop = stop
                    break
    except FileNotFoundError:
        if result.folders:
            raise
        # As fsspec's _du: a missing path measures nothing.
    except NotADirectoryError:
        if result.folders:
            raise
        result.files = 1
        result.size = (await fs._info(root)).get("size") or 0
    return _finish(result)
