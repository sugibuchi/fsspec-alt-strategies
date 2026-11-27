# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The size of a tree, from its listings alone.

`_tree_du` is an alternative to fsspec's ``_du``. fsspec's finds the files, then asks
each file's size with a request of its own. `_tree_du` adds up the sizes the
listings already give, as a map-reduce on `common.tree`'s traversal, by
`_Du`: a file's value is its size, a directory's the sum of its items'.

This module depends on `find` and `walk`, and `glob`, `operation` and `paths`
of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import DuState

from ..common.operation import Operation
from ..common.paths import _check_maxdepth
from .find import _listed, _read, _Reading
from .walk import _MAX_RUNNING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ..common.handler import Entry, StoragePath
    from .handler import TreeFileOperationHandler


__all__: list[str] = []


def _tree_du(
    handler: TreeFileOperationHandler,
    path: StoragePath,
    *,
    maxdepth: int | None = None,
    ils_kwargs: dict[str, Any] | None = None,
    max_running: int = _MAX_RUNNING,
) -> Operation[dict[str, Any], DuState]:
    """Measure a tree's size from its listings, listing several directories at a time.

    It assumes that ``ils`` gives each file's size, as fsspec's filesystems do.
    It yields the entry of each file counted, with its ``size``, as fsspec's
    ``_du`` gives with ``total=False``. Its state's ``size`` is their total, as
    fsspec's ``_du`` gives by default.

    A missing ``path`` measures nothing, as with fsspec's ``_du``. A file
    measures itself. A directory that fails to list yields its error, the
    root's too, and the rest of the tree is still measured.

    It wins over fsspec's ``_du`` on any storage, as it sends no ``info`` per
    file. Where the handler's profile declares ``recursive_ls``, e.g. an
    object store listing a whole tree in one paginated request, it sends that
    one listing, as summing the sizes from fsspec's ``_find`` with
    ``detail=True`` would.

    Once cancelled, it starts no new directory, and counts those started.

    Requests: one ``ils_recursive`` of ``path``, as above, or one ``ils`` per
    directory, each page holding a permit of the handler's
    ``operation_limit``. One ``info`` for ``path``, if a file.

    ``path`` is the directory measured on ``handler``, by its ``ils`` and ``info``.
    ``maxdepth`` of 1 counts only the files in ``path``. ``ils_kwargs`` go to
    each ``ils``, as fsspec's ``du`` passes its own on. ``max_running`` bounds
    the directories in flight.

    The operation yields the ``ils`` entry of each file counted, or the error
    of a directory that failed to list, and its state is a `DuState`. The call
    raises ValueError if ``maxdepth`` is less than 1.
    """
    _check_maxdepth(maxdepth)

    def counted(state: DuState, entry: Entry) -> DuState:
        size = entry.get("size") or 0
        return replace(state, files=state.files + 1, size=state.size + size)

    async def run(
        op: Operation[dict[str, Any], DuState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        steps = _read(
            handler,
            handler.trim(path),
            op,
            lambda emit, root: _Du(handler, emit, op, root),
            count=counted,
            max_running=max_running,
            maxdepth=maxdepth,
            ils_kwargs=ils_kwargs,
        )
        async with aclosing(steps):
            async for item in steps:
                yield item

    return Operation(run, DuState(directories_listed=0))


class _Du(_Reading):
    """du as a map-reduce: a file's value is its size, a directory's the sum of its items'.

    Each file's entry is emitted as it is counted. A directory that failed
    adds nothing.
    """

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        if entry["type"] != "directory":
            await self._emit_event(entry)
            return entry.get("size") or 0
        total = 0
        async for size in values:
            total += size or 0
        if entry is self.root:
            self.op.state = _listed(self.op.state, self.handler.trim(entry["name"]))
        return total
