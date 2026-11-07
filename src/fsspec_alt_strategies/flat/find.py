# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive find that streams one listing of a prefix.

`flat_find` is an alternative to fsspec's ``_find`` for flat storages, with the
same paths. fsspec's, and the object stores' own, gather every page of the
listing before answering. `flat_find` yields each path as its page arrives.

This module depends on `control`, `_keys`, and `paths` and `states` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..common.paths import _check_maxdepth
from ..common.states import FindState
from ..control import Operation
from ._keys import _scan

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator

    from ..common.operations import _Info
    from ._keys import _Ils

__all__ = ["FindState", "flat_find"]


def flat_find(
    path: str,
    *,
    ils: _Ils,
    info: _Info,
    maxdepth: int | None = None,
    withdirs: bool = False,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[dict[str, Any], FindState]:
    """Find the files under a prefix, from one listing streamed as it comes.

    It finds the paths fsspec's ``_find`` finds. A missing ``path`` finds
    nothing, and a file finds itself. With ``withdirs``, the folders are
    derived from the keys, as fsspec's filesystems derive them for object
    stores, and a key ending with "/" is a folder's marker. A derived folder's
    entry has only its ``name``, ``size`` 0 and ``type``.

    It yields each path as its page of the listing arrives, so the first come
    after one request, not after the last. They come in the listing's order,
    with each folder before its first key. fsspec's ``_find`` sorts them, by
    name.

    It wins over fsspec's ``_find`` on large prefixes of flat storages: the
    caller starts on the first page, and holds one page at a time, not the
    whole listing. It sends the same listing requests. It loses on storages
    with a hierarchical namespace, whose folders a listing by prefix does not
    see when empty: `tree_find` suits those.

    Once cancelled, it takes nothing more from the listing. An error from the
    listing ends the operation, raised by its iteration.

    Requests: the pages of one ``ils`` of ``path``, one after another. One
    ``info`` for ``path`` itself, if nothing is under it, or with ``withdirs``.

    Parameters
    ----------
    path : str
        The folder to search, as ``ils`` takes paths.
    ils : callable
        Lists every key starting with a prefix, called as ``ils(prefix)``: an
        async iterator of entries as fsspec's ``_ls`` gives them, in any order.
    info : callable
        Describes a path, called as ``info(path)``: e.g. the filesystem's
        ``_info``.
    maxdepth : int, optional
        The deepest level to find: 1 finds only the files in ``path``. No
        limit by default. The listing still lists every key under it.
    withdirs : bool
        Whether to find folders too, ``path`` included.
    concurrency_limit : asyncio.Semaphore
        Held by each ``info`` while it runs. The listing holds none.

    Returns
    -------
    Operation
        Yielding the entry of each path found. Its state is a `FindState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    tree_find : The same, for storages listing one folder per request.

    Examples
    --------
    >>> limit = asyncio.Semaphore(32)  # Shared, e.g. by a filesystem's operations.
    >>> find = flat_find("data", ils=my_ils, info=fs._info, concurrency_limit=limit)
    >>> async with find as op:
    ...     async for entry in op:
    ...         print(entry["name"])
    """
    _check_maxdepth(maxdepth)

    async def run(
        op: Operation[dict[str, Any], FindState],
    ) -> AsyncGenerator[dict[str, Any]]:
        steps = _scan(
            ils,
            info,
            path,
            maxdepth=maxdepth,
            withdirs=withdirs,
            concurrency_limit=concurrency_limit,
            stopping=op.cancelling,
        )
        async with aclosing(steps):
            async for entry in steps:
                op.state = replace(
                    op.state, current=entry["name"], found=op.state.found + 1
                )
                yield entry

    return Operation(run, FindState())
