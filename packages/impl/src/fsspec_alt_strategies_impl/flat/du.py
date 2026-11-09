# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The size of the files under a prefix, from one listing streamed as it comes.

`_flat_du` is an alternative to fsspec's ``_du`` for flat storages. fsspec's
finds the files, then asks each file's size with a request of its own. `_flat_du`
adds up the sizes the listing already gives, page by page.

This module depends on `find`, and `operation` and `paths` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import DuState

from ..common.operation import Operation
from ..common.paths import _check_maxdepth
from .find import _find_entries

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

__all__: list[str] = []


def _flat_du(
    fs: Any,
    path: str,
    *,
    maxdepth: int | None = None,
) -> Operation[dict[str, Any], DuState]:
    """Measure the files under a prefix, from one listing streamed as it comes.

    It assumes that ``_iexpand`` gives each file's size, as listings of object
    stores do. It yields the entry of each file counted, with its ``size``, as
    fsspec's ``_du`` gives with ``total=False``. Its state's ``size`` is their
    total, as fsspec's ``_du`` gives by default.

    A missing ``path`` measures nothing, as with fsspec's ``_du``. A file
    measures itself. A key ending with "/" is a folder's marker, and not
    counted.

    It wins over fsspec's ``_du`` on any flat storage, as it sends no ``info``
    per file, and holds one page of the listing at a time.

    Once cancelled, it takes nothing more from the listing. An error from the
    listing ends the operation, raised by its iteration.

    Requests: the pages of one ``_iexpand`` of ``path``, one after another.
    One ``_info`` for ``path``, if nothing is under it, holding a permit of
    the filesystem's ``operation_limit``. The listing holds none.

    Parameters
    ----------
    fs : FlatFileSystem
        The filesystem measured. Its ``_iexpand`` lists every key starting
        with a prefix, in order of their names, each key once.
    path : str
        The folder to measure, as ``_iexpand`` takes paths.
    maxdepth : int, optional
        The deepest level to measure: 1 counts only the files in ``path``. No
        limit by default. The listing still lists every key under it.

    Returns
    -------
    Operation
        Yielding the entry of each file counted. Its state is a `DuState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    _tree_du : The same, for storages listing one folder per request.
    """
    _check_maxdepth(maxdepth)

    async def run(
        op: Operation[dict[str, Any], DuState],
    ) -> AsyncGenerator[dict[str, Any]]:
        steps = _find_entries(fs, path, maxdepth=maxdepth, stopping=op.cancelling)

        async with aclosing(steps):
            async for entry in steps:
                op.state = DuState(
                    current=entry["name"],
                    files=op.state.files + 1,
                    size=op.state.size + (entry.get("size") or 0),
                )
                yield entry

    return Operation(run, DuState())
