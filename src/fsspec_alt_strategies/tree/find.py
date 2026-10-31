# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive find that lists several folders at a time.

`tree_find` is an alternative to fsspec's ``_find``, with the same paths. fsspec's
walks a tree one folder at a time, unless the filesystem lists whole trees.
`tree_find` lists up to ``batch_size`` folders at a time, as `tree_walk` does.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..control import Operation, State
from .walk import _check_maxdepth, _scan

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fsspec.asyn import AsyncFileSystem

__all__ = ["FindState", "tree_find"]


@dataclass(frozen=True, kw_only=True)
class FindState(State):
    """The progress of a `tree_find`.

    Attributes
    ----------
    folders_listed : int
        The folders listed so far, those that failed included.
    found : int
        The paths found so far.
    """

    folders_listed: int = 0
    found: int = 0


def tree_find(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    withdirs: bool = False,
    batch_size: int | None = None,
) -> Operation[dict[str, Any], FindState]:
    """Find the files in a tree, listing several folders at a time.

    It finds the paths fsspec's ``_find`` finds. A missing ``path`` finds
    nothing, and a file finds itself. A folder that fails to list yields its
    error, and the rest of the tree is still searched.

    It yields each path as its folder's listing arrives, so the paths are not
    in order. fsspec's ``_find`` sorts them, by name. A storage listing a
    folder twice, e.g. as a prefix and as its marker object, yields it twice.

    It wins over fsspec's ``_find`` on storages that list one folder per request,
    with many folders: their listings overlap. It loses on object stores, which
    list a whole tree in one paginated request.

    Once cancelled, it lists no new folder, and yields the paths of the
    listings in flight.

    Requests: one ``_ls`` per folder, at most ``batch_size`` at a time. One
    ``_info`` more for ``path`` itself, if a file, or a folder with
    ``withdirs``.

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

    Returns
    -------
    Operation
        Yielding the ``_ls`` entry of each path found, or the error of a folder
        that failed to list. Its state is a `FindState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    tree_walk : The walk under it.
    fsspec_alt_strategies.control.collect : All the paths, at once.

    Examples
    --------
    >>> entries, errors = await collect(tree_find(fs, "data", batch_size=32))
    >>> sorted(e["name"] for e in entries), errors
    (['data/2024/01/part-0.parquet', 'data/2024/02/part-0.parquet'], [])
    """
    _check_maxdepth(maxdepth)

    async def run(
        op: Operation[dict[str, Any], FindState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        root = fs._strip_protocol(path)
        steps = _scan(
            fs,
            root,
            maxdepth=maxdepth,
            batch_size=batch_size,
            stopping=op.cancelling,
            file_root=True,
            folder_root=withdirs and bool(root),
        )
        state = op.state
        async with aclosing(steps):
            async for step in steps:
                if isinstance(step, dict):
                    # The root's own entry.
                    op.state = state = replace(state, found=state.found + 1)
                    yield step
                    continue
                found = [
                    e for e in step.entries if withdirs or e["type"] != "directory"
                ]
                op.state = state = FindState(
                    errors=state.errors + (step.error is not None),
                    current=step.folder,
                    folders_listed=state.folders_listed + 1,
                    found=state.found + len(found),
                )
                if step.error is not None:
                    yield step.error
                for entry in found:
                    yield entry

    return Operation(run, FindState())
