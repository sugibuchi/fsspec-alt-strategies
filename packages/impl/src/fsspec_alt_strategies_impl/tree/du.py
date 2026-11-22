# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The size of a tree, from its listings alone.

`_tree_du` is an alternative to fsspec's ``_du``. fsspec's finds the files, then asks
each file's size with a request of its own. `_tree_du` adds up the sizes the
listings already give, as a map-reduce on `common.tree`'s traversal, by
`_Du`: a file's value is its size, a directory's the sum of its items'.

This module depends on `walk`, and `glob`, `operation`, `paths` and `tree` of
`common`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import DuState

from ..common.glob import PathAction
from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim
from ..common.tree import TreeTraversal
from .walk import _MAX_RUNNING, _root_task, _unlisted

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
    file. Where the handler's profile declares
    ``recursive_ls_depth_first_preorder``, e.g. an object store listing a
    whole tree in one paginated request, it sends that one listing, as summing
    the sizes from fsspec's ``_find`` with ``detail=True`` would.

    Once cancelled, it starts no new directory, and counts those started.

    Requests: one recursive ``ils`` of ``path``, as above, or one ``ils`` per
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

    async def run(
        op: Operation[dict[str, Any], DuState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        engine = TreeTraversal(max_running=max_running, split=handler.segments)
        op._on_cancel = engine.cancel
        await handler.open()  # Its profile, read by _root_task.
        top = len(tuple(handler.segments(path)))
        root = _root_task(
            engine,
            {"name": path, "type": "directory"},
            _Du(engine._emit_event, handler, top, maxdepth),
            handler,
            maxdepth=maxdepth,
            ils_kwargs=ils_kwargs,
        )
        try:
            async with aclosing(engine.events([root])) as events:
                async for event in events:
                    state = op.state
                    if isinstance(event, tuple):  # A directory failed: (entry, error).
                        entry, error = event
                        op.state = replace(
                            state,
                            current=_trim(entry["name"]),
                            errors=state.errors + 1,
                            directories_listed=(state.directories_listed or 0) + 1,
                        )
                        yield error
                    elif event["type"] == "directory":  # Listed, and counted.
                        op.state = replace(
                            state,
                            current=_trim(event["name"]),
                            directories_listed=(state.directories_listed or 0) + 1,
                        )
                    else:
                        op.state = _counted(state, event)
                        yield event
        except FileNotFoundError:
            return  # A missing root measures nothing, as fsspec's _du.
        except NotADirectoryError:
            # A file measures itself.
            async with handler.operation_limit:
                entry = await handler.info(handler.entry(_trim(path)))
            op.state = _counted(op.state, entry)
            yield entry
        except Exception as error:
            # The root's other failures are a failed listing, as any is.
            op.state = replace(
                op.state, current=_trim(path), errors=op.state.errors + 1
            )
            yield error

    return Operation(run, DuState(directories_listed=0))


class _Du(PathAction):
    """du as a map-reduce: a file's value is its size, a directory's the sum of its items'.

    Each file's entry is emitted as it is counted, and each directory's as it
    is reduced, once listed: one at ``maxdepth`` is reduced unlisted, and
    emits nothing. A directory that failed adds nothing.
    """

    def __init__(
        self,
        emit_event: Any,
        handler: TreeFileOperationHandler,
        top: int,
        maxdepth: int | None,
    ) -> None:
        super().__init__(emit_event)
        self.handler = handler
        self.top = top
        self.maxdepth = maxdepth

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: list[Any]
    ) -> Any:
        if entry["type"] != "directory":
            await self._emit_event(entry)
            return entry.get("size") or 0
        if not _unlisted(self.handler, self.top, self.maxdepth, entry):
            await self._emit_event(entry)
        return sum(v for v in values if v is not None)


def _counted(state: DuState, entry: Entry) -> DuState:
    """The state once a file is counted."""
    return replace(
        state, files=state.files + 1, size=state.size + (entry.get("size") or 0)
    )
