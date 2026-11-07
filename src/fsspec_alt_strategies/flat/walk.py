# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A walk of the keys under a prefix, from one listing streamed as it comes.

`flat_walk` is the counterpart of `tree_walk` for flat storages. fsspec's
``walk`` lists one folder per request, even on object stores, which can list
all the keys under a prefix at once. `flat_walk` makes that one listing, and
derives the folders from the keys.

This module depends on `control`, `_keys`, and `paths` and `states` of
`common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.paths import _check_maxdepth, _trim
from ..common.states import WalkState
from ..control import Operation
from ._keys import _close, _folder, _folders

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator, Callable

    from ..common.operations import _Info
    from ._keys import _Ils

__all__ = ["WalkState", "flat_walk"]


def flat_walk(
    path: str,
    *,
    ils: _Ils,
    info: _Info,
    maxdepth: int | None = None,
    topdown: bool = True,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[dict[str, Any], WalkState]:
    """Walk the keys under a prefix, from one listing streamed as it comes.

    It yields the entry of each file and folder below ``path``, as `tree_walk`
    does, but not ``path`` itself. The files are the keys, as ``ils`` gives
    them. The folders are derived from the keys, as fsspec's filesystems derive
    them for object stores, and a key ending with "/" is a folder's marker. A
    derived folder's entry has only its ``name``, ``size`` 0 and ``type``.

    With ``topdown``, each folder comes before what is in it, as soon as its
    first key is listed. Unlike `tree_walk`'s, a folder's files may come after
    those of its subfolders: the listing is in no order of folders. Without
    ``topdown``, each folder comes after all that is in it, as fsspec's
    ``walk`` orders them. The files come as listed, and the folders once the
    listing ends, the deepest first, as only then is a folder known to be
    done. So a walk bottom up that is cancelled leaves out the folders.

    ``maxdepth`` and ``descend`` leave paths out, but save no request: the
    listing gives all the keys under ``path`` anyway.

    It wins over fsspec's ``walk`` on flat storages, with many folders: one
    listing, of as many pages as there are keys, rather than a listing per
    folder. It loses on storages with a hierarchical namespace, whose empty
    folders a listing by prefix does not see: `tree_walk` suits those.

    Once cancelled, it takes nothing more from the listing. Iterating it raises
    these errors, which end it:

    - FileNotFoundError, if ``path`` does not exist;
    - NotADirectoryError, if ``path`` is a file;
    - the error of the listing, as there is no other to go on with.

    Requests: the pages of one ``ils`` of ``path``, one after another. One
    ``info`` for ``path``, if nothing is under it.

    Parameters
    ----------
    path : str
        The folder to start from, as ``ils`` takes paths.
    ils : callable
        Lists every key starting with a prefix, called as ``ils(prefix)``: an
        async iterator of entries as fsspec's ``_ls`` gives them, in any order.
    info : callable
        Describes a path, called as ``info(path)``: e.g. the filesystem's
        ``_info``.
    maxdepth : int, optional
        The deepest level to yield: 1 yields only what is in ``path``. No limit
        by default.
    topdown : bool
        Whether a folder comes before what is in it, rather than after.
    descend : callable, optional
        Called with the entry of each folder found. What is in it is yielded
        only if this returns True. All by default.
    concurrency_limit : asyncio.Semaphore
        Held by the ``info`` while it runs. The listing holds none. Share one
        to limit several operations, or files, together.

    Returns
    -------
    Operation
        Yielding the entry of each file and folder found. Its state is a
        `WalkState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    tree_walk : The same, for storages listing one folder per request.
    flat_find : The files under a prefix, as fsspec's ``_find`` finds them.

    Examples
    --------
    >>> limit = asyncio.Semaphore(16)  # Shared, e.g. by a filesystem's operations.
    >>> walk = flat_walk("data", ils=my_ils, info=fs._info, concurrency_limit=limit)
    >>> async with walk as entries:
    ...     async for entry in entries:
    ...         print(entry["name"], entry["type"])
    """
    _check_maxdepth(maxdepth)

    async def run(
        op: Operation[dict[str, Any], WalkState],
    ) -> AsyncGenerator[dict[str, Any]]:
        entries = _entries(
            ils,
            info,
            path,
            maxdepth=maxdepth,
            topdown=topdown,
            descend=descend,
            concurrency_limit=concurrency_limit,
            stopping=op.cancelling,
        )
        try:
            async for entry in entries:
                op.state = WalkState(current=entry["name"], found=op.state.found + 1)
                yield entry
        finally:
            await entries.aclose()

    return Operation(run, WalkState())


async def _entries(
    ils: _Ils,
    info: _Info,
    path: str,
    *,
    maxdepth: int | None,
    topdown: bool,
    descend: Callable[[dict[str, Any]], bool] | None,
    concurrency_limit: asyncio.Semaphore,
    stopping: Callable[[], object],
) -> AsyncGenerator[dict[str, Any]]:
    """The entries `flat_walk` yields, in its order."""
    root = _trim(path)
    prefix = f"{root}/" if root else ""
    under = False
    seen: set[str] = set()
    # The folders whose contents descend left out.
    pruned: set[str] = set()
    # Bottom up, the folders, held until the listing ends.
    held: list[dict[str, Any]] = []
    keys = ils(prefix)
    try:
        async for key in keys:
            under = True
            relative = key["name"][len(prefix) :]
            for entry in _found(key, relative, prefix, seen, pruned, maxdepth, descend):
                if topdown or entry["type"] != "directory":
                    yield entry
                else:
                    held.append(entry)
            if stopping():
                return
    finally:
        await _close(keys)
    if not under and root:
        # Nothing under it: a file, or missing, as tree_walk raises for them.
        async with concurrency_limit:
            entry = await info(root)
        if entry["type"] != "directory":
            raise NotADirectoryError(root)
    # The deepest first, so that each folder comes after all in it.
    for entry in sorted(held, key=lambda e: e["name"].count("/"), reverse=True):
        yield entry


def _found(
    key: dict[str, Any],
    relative: str,
    prefix: str,
    seen: set[str],
    pruned: set[str],
    maxdepth: int | None,
    descend: Callable[[dict[str, Any]], bool] | None,
) -> list[dict[str, Any]]:
    """What a key adds to the walk: the folders it implies first seen, then itself.

    Nothing below a folder left out by ``descend``, nor deeper than
    ``maxdepth``, is added.
    """
    if not relative:
        return []  # The root's own marker.
    found = []
    for folder in _folders(relative):
        full = prefix + folder
        if full in pruned:
            return found
        depth = folder.count("/") + 1
        if maxdepth is not None and depth > maxdepth:
            return found
        if full not in seen:
            seen.add(full)
            entry = _folder(full)
            found.append(entry)
            if descend is not None and not descend(entry):
                pruned.add(full)
                return found
    depth = relative.count("/") + 1
    if not relative.endswith("/") and (maxdepth is None or depth <= maxdepth):
        found.append(key)
    return found
