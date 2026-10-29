# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive find that lists several folders at a time.

`tree_find` is an alternative to fsspec's ``_find``, with the same answers.
fsspec's walks a tree one folder at a time, unless the filesystem lists whole
trees. `tree_find` lists up to ``batch_size`` folders at a time with `walk`.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .control import Result, State, _finish, _report
from .walk import walk

if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem

    from .control import Callback

__all__ = ["FindResult", "FindState", "tree_find"]


@dataclass(frozen=True, kw_only=True)
class FindState(State):
    """The state of a `tree_find`, as its callback gets it.

    Attributes
    ----------
    folders_listed : int
        The folders listed so far.
    found : int
        The paths found so far.
    """

    folders_listed: int = 0
    found: int = 0


@dataclass(kw_only=True)
class FindResult(Result):
    """The paths a `tree_find` found.

    Attributes
    ----------
    entries : dict of str to dict
        Each path found, sorted, with its entry from ``_ls``. ``list(entries)``
        is what fsspec's ``_find`` returns, and ``entries`` is what it returns
        with ``detail=True``.
    """

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)


async def tree_find(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    withdirs: bool = False,
    batch_size: int | None = None,
    callback: Callback[FindState] | None = None,
) -> FindResult:
    """Find the files in a tree, listing several folders at a time.

    It answers as fsspec's ``_find`` does. A missing ``path`` finds nothing, and
    a file finds itself. A folder that fails to list is in ``errors``, and the
    rest of the tree is still searched.

    It wins over fsspec's ``_find`` on storages that list one folder per request,
    with many folders: their listings overlap. It loses on object stores, which
    list a whole tree in one paginated request.

    Requests: one ``_ls`` per folder, at most ``batch_size`` at a time. One
    ``_info`` more for ``path`` itself, if a file, or a folder with
    ``withdirs``.

    The callback is called after each folder listed. A stop returns the paths
    found so far. ``Stop(cleanup=True)`` undoes nothing, as nothing is changed.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path : str
        The folder to search.
    maxdepth : int, optional
        The deepest level to search: 1 finds only the files in ``path``. No
        limit by default.
    withdirs : bool
        Whether to find folders too, ``path`` included.
    batch_size : int, optional
        The most listings at a time. fsspec's default if None.
    callback : callable, optional
        Called with a `FindState` after each step. It may return a `Stop`.

    Returns
    -------
    FindResult
        The paths found, and the folders that failed.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.
    OperationStopped
        If the callback returned ``Stop(raise_error=True)``.

    See Also
    --------
    walk : The walk under it.

    Examples
    --------
    >>> result = await tree_find(fs, "data", batch_size=32)
    >>> list(result.entries)
    ['data/2024/01/part-0.parquet', 'data/2024/02/part-0.parquet']
    """
    root = fs._strip_protocol(path)
    result = FindResult()
    found: dict[str, dict[str, Any]] = {}
    listed = 0
    try:
        async with aclosing(
            walk(fs, root, maxdepth=maxdepth, batch_size=batch_size)
        ) as listings:
            async for listing in listings:
                listed += 1
                if listing.error is not None:
                    result.errors[listing.folder] = listing.error
                found.update(
                    (e["name"].rstrip("/"), e)
                    for e in listing.entries
                    if withdirs or e["type"] != "directory"
                )
                state = FindState(
                    errors=len(result.errors),
                    current=listing.folder,
                    folders_listed=listed,
                    found=len(found),
                )
                if (stop := await _report(callback, state)) is not None:
                    result.stop = stop
                    break
    except FileNotFoundError:
        if listed:
            raise
        # As fsspec's _find: a missing path finds nothing.
    except NotADirectoryError:
        if listed:
            raise
        found[root] = await fs._info(root)
    else:
        if withdirs and root:
            found[root] = await fs._info(root)
    result.entries = dict(sorted(found.items()))
    return _finish(result)
