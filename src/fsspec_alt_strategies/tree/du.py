# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The size of a tree, from its listings alone.

`tree_du` is an alternative to fsspec's ``_du``. fsspec's finds the files, then asks
each file's size with a request of its own. `tree_du` adds up the sizes the
listings already give.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ..control import Operation, State
from .walk import _check_maxdepth, _max_concurrency, _scan

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fsspec.asyn import AsyncFileSystem

__all__ = ["DuState", "tree_du"]


@dataclass(frozen=True, kw_only=True)
class DuState(State):
    """The progress of a `tree_du`.

    Attributes
    ----------
    folders_listed : int
        The folders listed so far, those that failed included.
    files : int
        The files counted so far.
    size : int
        Their size so far, in bytes. Once the operation has run to its end, it
        is what fsspec's ``_du`` returns.
    """

    folders_listed: int = 0
    files: int = 0
    size: int = 0


def tree_du(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    max_concurrency: int | None = None,
) -> Operation[dict[str, Any], DuState]:
    """Measure a tree's size from its listings, listing several folders at a time.

    It assumes that ``_ls`` gives each file's size, as fsspec's filesystems do.
    It yields the entry of each file counted, with its ``size``, as fsspec's
    ``_du`` gives with ``total=False``. Its state's ``size`` is their total, as
    fsspec's ``_du`` gives by default.

    A missing ``path`` measures nothing, as with fsspec's ``_du``. A file
    measures itself. A folder that fails to list yields its error, and the rest
    of the tree is still measured.

    It wins over fsspec's ``_du`` on any storage, as it sends no ``_info`` per
    file. On object stores, which list a whole tree in one paginated request,
    summing the sizes from fsspec's ``_find`` with ``detail=True`` makes fewer
    requests still.

    Once cancelled, it lists no new folder, and counts the listings in flight.

    Requests: one ``_ls`` per folder, at most ``max_concurrency`` at a time. One
    ``_info`` for ``path``, if a file.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path : str
        The folder to measure.
    maxdepth : int, optional
        The deepest level to measure: 1 counts only the files in ``path``. No
        limit by default.
    max_concurrency : int, optional
        The most listings at a time. 32 if None.

    Returns
    -------
    Operation
        Yielding the ``_ls`` entry of each file counted, or the error of a
        folder that failed to list. Its state is a `DuState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    tree_walk : The walk under it.

    Examples
    --------
    >>> async with tree_du(fs, "data", max_concurrency=32) as op:
    ...     async for entry in op:
    ...         if isinstance(entry, Exception):
    ...             raise entry
    >>> op.state.size, op.state.files
    (73400320, 70)
    """
    _check_maxdepth(maxdepth)
    _max_concurrency(max_concurrency)

    async def run(
        op: Operation[dict[str, Any], DuState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        steps = _scan(
            fs,
            path,
            maxdepth=maxdepth,
            max_concurrency=max_concurrency,
            stopping=op.cancelling,
            file_root=True,
        )
        state = op.state
        async with aclosing(steps):
            async for step in steps:
                if isinstance(step, dict):
                    # The root's own entry, as it is a file.
                    op.state = state = replace(
                        state,
                        current=step["name"],
                        files=state.files + 1,
                        size=state.size + (step.get("size") or 0),
                    )
                    yield step
                    continue
                files = [e for e in step.entries if e["type"] != "directory"]
                op.state = state = DuState(
                    errors=state.errors + (step.error is not None),
                    current=step.folder,
                    folders_listed=state.folders_listed + 1,
                    files=state.files + len(files),
                    size=state.size + sum(e.get("size") or 0 for e in files),
                )
                if step.error is not None:
                    yield step.error
                for entry in files:
                    yield entry

    return Operation(run, DuState())
