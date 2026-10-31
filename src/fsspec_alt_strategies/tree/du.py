# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The size of a tree, from its listings alone.

`du` is an alternative to fsspec's ``_du``. fsspec's finds the files, then asks
each file's size with a request of its own. `du` adds up the sizes the
listings already give.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..control import Operation, State
from .walk import _check_maxdepth, _listings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fsspec.asyn import AsyncFileSystem

__all__ = ["DuState", "du"]


@dataclass(frozen=True, kw_only=True)
class DuState(State):
    """The progress of a `du`.

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


def du(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    batch_size: int | None = None,
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

    Requests: one ``_ls`` per folder, at most ``batch_size`` at a time. One
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
    batch_size : int, optional
        The most listings at a time. fsspec's default if None.

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
    walk : The walk under it.

    Examples
    --------
    >>> async with du(fs, "data", batch_size=32) as op:
    ...     async for entry in op:
    ...         if isinstance(entry, Exception):
    ...             raise entry
    >>> op.state.size, op.state.files
    (73400320, 70)
    """
    _check_maxdepth(maxdepth)

    async def run(
        op: Operation[dict[str, Any], DuState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        root = fs._strip_protocol(path)
        listings = _listings(
            fs, root, maxdepth=maxdepth, batch_size=batch_size, stopping=op.cancelling
        )
        state = op.state
        try:
            async with aclosing(listings):
                async for listing in listings:
                    files = [e for e in listing.entries if e["type"] != "directory"]
                    op.state = state = DuState(
                        errors=state.errors + (listing.error is not None),
                        current=listing.folder,
                        folders_listed=state.folders_listed + 1,
                        files=state.files + len(files),
                        size=state.size + sum(e.get("size") or 0 for e in files),
                    )
                    if listing.error is not None:
                        yield listing.error
                    for entry in files:
                        yield entry
        except FileNotFoundError:
            if state.folders_listed:
                raise
            # As fsspec's _du: a missing path measures nothing.
        except NotADirectoryError:
            if state.folders_listed:
                raise
            entry = await fs._info(root)
            op.state = DuState(current=root, files=1, size=entry.get("size") or 0)
            yield entry

    return Operation(run, DuState())
