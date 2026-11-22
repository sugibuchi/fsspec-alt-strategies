# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A concurrent walk of a directory tree, as fsspec's ``walk`` answers.

`_tree_walk` yields each directory's ``(path, dirs, files)``, as fsspec's
``walk`` gives them, on `common.tree`'s traversal, in the order fsspec's
``walk`` promises:

- top down, a directory before those below it, by a `_WalkTopDown` per
  directory: its caller may prune by changing ``dirs``;
- bottom up, a directory after all of those below it, by `_WalkBottomUp`,
  which emits each directory as `common.glob`'s tasks reduce it.

This module depends on `glob`, `handler`, `operation`, `paths` and `tree` of
`common`.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import WalkState

from ..common.glob import (
    OnError,
    PathAction,
    TraverseSubtree,
    VisitDirectory,
    _TraversalTask,
)
from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim, _triples
from ..common.tree import TreeTraversal

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator

    from ..common.handler import Entry, StoragePath
    from .handler import TreeFileOperationHandler

__all__: list[str] = []

# The directories in flight at most: the requests are bounded by the handler's
# operation_limit, so this bounds only the tasks held in memory.
_MAX_RUNNING = 64

_Walked = tuple[str, dict[str, Any], dict[str, Any]]


def _tree_walk(
    handler: TreeFileOperationHandler,
    path: StoragePath,
    *,
    maxdepth: int | None = None,
    topdown: bool = True,
    on_error: OnError = None,
    ils_kwargs: dict[str, Any] | None = None,
    max_running: int = _MAX_RUNNING,
) -> Operation[_Walked, WalkState]:
    """Walk a directory tree, listing several directories at a time.

    It yields each directory listed as ``(path, dirs, files)``, as fsspec's
    ``walk`` does with ``detail=True``. ``dirs`` and ``files`` map the names
    in the directory to their ``ils`` entries. An empty directory yields empty
    ones. A directory that fails to list yields its error instead, as the
    storage raised it, and the walk goes on with the others: the state's
    ``current`` is then its path. ``on_error``, below, may handle those errors
    instead, as with fsspec's ``walk``. A subdirectory that failed is still in
    its parent's ``dirs``, as its parent's listing found it.

    With ``topdown``, a directory comes before those below it, each listed on
    its own, and a caller may prune by changing ``dirs`` in place, as with
    fsspec's ``walk``. Without, a directory comes after all of those below it.
    Where the handler's profile declares ``recursive_ls_depth_first_preorder``
    and no ``maxdepth`` prunes it, the tree is listed by one recursive ``ils``,
    streamed: a listing broken off part way fails the whole tree. Otherwise
    each directory is listed on its own.

    Once cancelled, it starts no new directory, and yields those started:
    those whose pages wait for a permit too, as the permits bound the
    requests, not the directories. Iterating it raises these errors, which end
    it:

    - FileNotFoundError, if ``path`` does not exist;
    - NotADirectoryError, if ``path`` is a file;
    - with ``on_error="raise"``, the error of the first directory that failed to
      list.

    Requests: one ``ils`` per directory walked, or bottom up by one listing,
    one recursive ``ils`` of ``path``. Each page holds a permit of the
    handler's ``operation_limit``.

    ``path`` is the directory the walk starts from, listed by the ``ils`` of
    ``handler``. ``maxdepth`` of 1 lists only ``path``. ``ils_kwargs`` go to
    each ``ils``, as fsspec's ``walk`` passes its own on. ``max_running``
    bounds the directories in flight.

    ``on_error`` says what becomes of the error of a directory that fails to
    list:

    - None yields it, as every strategy does;
    - "omit" leaves it out;
    - "raise" raises it, ending the walk;
    - a callable is called with it, and it is left out.

    The operation yields ``(path, dirs, files)`` for each directory listed, or an
    error, and its state is a `WalkState`. The call raises ValueError if
    ``maxdepth`` is less than 1, or ``on_error`` is none of the above.
    """
    _check_maxdepth(maxdepth)
    if not (on_error in (None, "omit", "raise") or callable(on_error)):
        raise ValueError(f"on_error must be 'omit', 'raise' or callable: {on_error!r}")

    def root(engine: TreeTraversal) -> _TraversalTask:
        entry = {"name": path, "type": "directory"}
        options: dict[str, Any] = {
            "maxdepth": maxdepth,
            # "raise" is the walk's, so that the error comes bare, not in a group.
            "on_error": None if on_error == "raise" else on_error,
            "ils_kwargs": ils_kwargs,
        }
        if topdown:
            action = PathAction(engine._emit_event)
            return _WalkTopDown(engine, entry, action, handler, **options)
        top = len(tuple(handler.segments(path)))
        action = _WalkBottomUp(engine._emit_event, handler, top, maxdepth)
        return _bottom_up(engine, entry, action, handler, **options)

    async def run(
        op: Operation[_Walked, WalkState],
    ) -> AsyncGenerator[_Walked | Exception]:
        engine = TreeTraversal(max_running=max_running, split=handler.segments)
        op._on_cancel = engine.cancel
        await handler.open()  # Its profile, read by root.
        async with aclosing(engine.events([root(engine)])) as events:
            async for event in events:
                state = op.state
                if len(event) == 2:  # A directory failed: (entry, error).
                    entry, error = event
                    if on_error == "raise":
                        raise error
                    # Its path is the state's current, for a caller yielding it.
                    op.state = replace(
                        state, current=_trim(entry["name"]), errors=state.errors + 1
                    )
                    yield error
                    continue
                walked, directory, dirs, files = event
                op.state = replace(
                    state,
                    current=directory,
                    directories_listed=(state.directories_listed or 0) + 1,
                    found=state.found + len(dirs) + len(files),
                )
                try:
                    yield directory, dirs, files
                finally:
                    if walked is not None:
                        walked.set()  # The directory goes on, with the dirs left.

    return Operation(run, WalkState(directories_listed=0))


def _bottom_up(
    engine: TreeTraversal,
    entry: Entry,
    action: PathAction,
    handler: TreeFileOperationHandler,
    **options: Any,
) -> _TraversalTask:
    """The root task of a traversal whose action needs no order but its own: one listing where it streams right.

    Where the handler's profile declares ``recursive_ls_depth_first_preorder``
    and no ``maxdepth`` prunes it, each directory's descendants come together,
    so one recursive listing is streamed. Else a directory per listing: one
    failing fails only itself. The handler is open, its profile read.
    """
    if (
        options.get("maxdepth") is None
        and handler.profile.recursive_ls_depth_first_preorder
    ):
        return TraverseSubtree(engine, entry, action, handler, **options)
    return VisitDirectory(engine, entry, action, handler, **options)


def _unlisted(
    handler: TreeFileOperationHandler, top: int, maxdepth: int | None, entry: Entry
) -> bool:
    """Whether a directory is reduced unlisted, at ``maxdepth``: ``top`` is the root's segments counted."""
    depth = len(tuple(handler.segments(entry["name"]))) - top
    return maxdepth is not None and depth >= maxdepth


class _WalkTopDown(VisitDirectory):
    """A directory of a walk top down: its ``(path, dirs, files)`` out before any below it, pruned by the caller.

    Its listing pruned, it emits ``(walked, path, dirs, files)``, and waits
    for ``walked``, an `asyncio.Event`, which the caller sets once it has
    taken them. A subdirectory the caller removed from ``dirs`` meanwhile is
    not walked, as with `os.walk` top down. So a directory comes after its
    parent's, which waits: never before it.

    A directory that fails to list emits nothing of its own: `VisitDirectory`
    handles its error as ``on_error`` says.
    """

    async def prune(self, entries: AsyncIterable[Entry]) -> AsyncIterator[Entry]:
        kept = [e async for e in super().prune(entries)]
        if self._listing_error is not None:
            return  # Listed in part: never walked.
        path, dirs, files = _triples(_trim(self.entry["name"]), kept, detail=True)
        walked = asyncio.Event()
        await self._emit_event((walked, path, dirs, files))
        await walked.wait()
        for entry in dirs.values():  # Those the caller left.
            yield entry


class _WalkBottomUp(PathAction):
    """A walk bottom up: each directory's ``(None, path, dirs, files)`` emitted as it is reduced, after all below it.

    Each item's value is its entry, a failed subdirectory's too, so that its
    parent names it in ``dirs``, as its listing found it. A directory at
    ``maxdepth`` is reduced unlisted, and emits nothing: ``top`` is the
    root's segments counted.
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

    async def map(self, entry: Entry, match: bool) -> Any:
        return entry

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: list[Any]
    ) -> Any:
        unlisted = _unlisted(self.handler, self.top, self.maxdepth, entry)
        if entry["type"] == "directory" and not unlisted:
            path, dirs, files = _triples(_trim(entry["name"]), values, detail=True)
            await self._emit_event((None, path, dirs, files))
        return entry

    async def failed(self, entry: Entry) -> Any:
        return entry
