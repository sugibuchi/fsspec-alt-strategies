# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive find that lists several folders at a time.

`find` is an alternative to fsspec's ``_find``, with the same answers. fsspec's
walks a tree one folder at a time, unless the filesystem lists whole trees.
`find` lists up to ``batch_size`` folders at a time, as `iwalk` does.

`ifind` is the same search as an async iterator, so that the paths can be
worked on as they arrive.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .control import State, _finish, _report
from .walk import _listings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from fsspec.asyn import AsyncFileSystem

    from .control import Callback, Stop

__all__ = ["FindState", "find", "ifind"]


@dataclass(frozen=True, kw_only=True)
class FindState(State):
    """The state of a `find`, as its callback gets it.

    Attributes
    ----------
    folders_listed : int
        The folders listed so far.
    found : int
        The paths found so far.
    """

    folders_listed: int = 0
    found: int = 0


async def find(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    withdirs: bool = False,
    batch_size: int | None = None,
    callback: Callback[FindState] | None = None,
) -> tuple[list[dict[str, Any]], list[Exception]]:
    """Find the files in a tree, listing several folders at a time.

    It answers as fsspec's ``_find`` does. A missing ``path`` finds nothing, and
    a file finds itself. A folder that fails to list adds its error, and the
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
    entries : list of dict
        The ``_ls`` entry of each path found, sorted by name. Their names are
        what fsspec's ``_find`` returns.
    errors : list of Exception
        The error of each folder that failed to list, in the order they failed.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.
    OperationStopped
        If the callback returned ``Stop(raise_error=True)``.

    See Also
    --------
    ifind : The same search, as an async iterator.
    iwalk : The walk under it.

    Examples
    --------
    >>> entries, errors = await find(fs, "data", batch_size=32)
    >>> [e["name"] for e in entries], errors
    (['data/2024/01/part-0.parquet', 'data/2024/02/part-0.parquet'], [])
    """
    found: dict[str, dict[str, Any]] = {}
    errors: list[Exception] = []
    listed = 0
    stop: Stop | None = None
    async with aclosing(
        _steps(fs, path, maxdepth=maxdepth, withdirs=withdirs, batch_size=batch_size)
    ) as steps:
        async for folder, items in steps:
            for item in items:
                if isinstance(item, Exception):
                    errors.append(item)
                else:
                    # Keyed by name as fsspec's _find keys them, so a path listed
                    # twice, e.g. a folder and its marker object, is found once.
                    found[item["name"].rstrip("/")] = item
            if folder is None:
                continue
            listed += 1
            state = FindState(
                errors=len(errors),
                current=folder,
                folders_listed=listed,
                found=len(found),
            )
            if (stop := await _report(callback, state)) is not None:
                break
    return _finish(([found[name] for name in sorted(found)], errors), stop, errors)


async def ifind(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    withdirs: bool = False,
    batch_size: int | None = None,
) -> AsyncIterator[dict[str, Any] | Exception]:
    """Yield the files in a tree, listing several folders at a time.

    It finds what `find` finds, with the same requests. It yields each path as
    its folder's listing arrives, so the paths are not in order. A folder that
    fails to list yields its error, and the rest of the tree is still searched.

    Like `iwalk`, it takes no callback: the caller drives it. Breaking out of
    the loop stops it. Close it with `contextlib.aclosing`, so that its
    listings in flight are cancelled at once.

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

    Yields
    ------
    dict or Exception
        The ``_ls`` entry of each path found, or the error of a folder that
        failed to list.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    find : The same search, collected and sorted.
    iwalk : The walk under it.

    Examples
    --------
    >>> from contextlib import aclosing
    >>> async with aclosing(ifind(fs, "data")) as paths:
    ...     async for path in paths:
    ...         if isinstance(path, Exception):
    ...             raise path
    ...         print(path["name"])
    data/2024/01/part-0.parquet
    """
    async with aclosing(
        _steps(fs, path, maxdepth=maxdepth, withdirs=withdirs, batch_size=batch_size)
    ) as steps:
        async for _, items in steps:
            for item in items:
                yield item


async def _steps(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None,
    withdirs: bool,
    batch_size: int | None,
) -> AsyncGenerator[tuple[str | None, list[dict[str, Any] | Exception]]]:
    """Search as `find` does, yielding what each step found.

    A step is a folder listed, with its folder, or the ``_info`` of ``path``
    itself, with None.
    """
    root = fs._strip_protocol(path)
    listed = False
    try:
        async with aclosing(
            _listings(fs, root, maxdepth=maxdepth, batch_size=batch_size)
        ) as listings:
            async for listing in listings:
                listed = True
                items: list[dict[str, Any] | Exception] = []
                if listing.error is not None:
                    items.append(listing.error)
                items += (
                    e for e in listing.entries if withdirs or e["type"] != "directory"
                )
                yield listing.folder, items
    except FileNotFoundError:
        if listed:
            raise
        # As fsspec's _find: a missing path finds nothing.
        return
    except NotADirectoryError:
        if listed:
            raise
        yield None, [await fs._info(root)]
        return
    if withdirs and root:
        yield None, [await fs._info(root)]
