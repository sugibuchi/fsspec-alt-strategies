# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive find that lists several directories at a time.

`_tree_find` is an alternative to fsspec's ``_find``, with the same paths.
fsspec's walks a tree one directory at a time, unless the filesystem lists
whole trees. `_tree_find` lists the tree on `common.tree`'s traversal: by one
recursive ``ils``, or a directory per request, several at a time.

`_read` runs the strategies that only read, find, glob and du, on the
traversal, and handles a tree's root as fsspec's ``_find`` does.

This module depends on `walk`, and `glob`, `handler`, `operation`, `paths`
and `tree` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar

from fsspec_alt_strategies.states import DuState, FindState, GlobState

from ..common.glob import PathAction
from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim
from ..common.tree import TreeTraversal
from .walk import _MAX_RUNNING, _root_task

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Sequence

    from ..common.handler import Entry, StoragePath
    from ..common.patterns import _Search
    from .handler import TreeFileOperationHandler

__all__: list[str] = []

_S = TypeVar("_S", FindState, DuState, GlobState)


def _tree_find(
    handler: TreeFileOperationHandler,
    path: StoragePath,
    *,
    maxdepth: int | None = None,
    withdirs: bool = False,
    ils_kwargs: dict[str, Any] | None = None,
    max_running: int = _MAX_RUNNING,
) -> Operation[dict[str, Any], FindState]:
    """Find the files in a tree, listing several directories at a time.

    It finds the paths fsspec's ``_find`` finds. A missing ``path`` finds
    nothing, and a file finds itself. A directory that fails to list yields its
    error, and the rest of the tree is still searched. A tree listed by one
    recursive ``ils``, broken off, is its root's failure: what it passed is
    found, and its error yielded.

    It yields each path as it is listed, so the paths are not in order.
    fsspec's ``_find`` sorts them, by name. A storage listing a directory
    twice, e.g. as a prefix and as its marker object, yields it once.

    It wins over fsspec's ``_find`` on storages that list one directory per
    request, with many directories: their listings overlap. It gains nothing on
    object stores, which list a whole tree in one paginated request: it sends
    that same listing, where the profile declares
    ``recursive_ls_depth_first_preorder``.

    Once cancelled, it starts no new directory, and yields the paths of those
    started.

    Requests: one recursive ``ils`` of ``path``, as above, else one ``ils``
    per directory, each page holding a permit of the handler's
    ``operation_limit``. One ``info`` more for ``path`` itself, if a file, or
    a directory with ``withdirs``.

    ``path`` is the directory searched on ``handler``, by its ``ils`` and ``info``.
    ``maxdepth`` of 1 finds only the files in ``path``. ``withdirs`` finds
    directories too, ``path`` included. ``ils_kwargs`` go to each ``ils``, as
    fsspec's ``find`` passes its own on. ``max_running`` bounds the
    directories in flight.

    The operation yields the ``ils`` entry of each path found, or the error
    of a directory that failed to list, and its state is a `FindState`. The call
    raises ValueError if ``maxdepth`` is less than 1.
    """
    _check_maxdepth(maxdepth)

    def found(state: FindState, entry: Entry) -> FindState:
        return replace(state, found=state.found + 1)

    async def run(
        op: Operation[dict[str, Any], FindState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        root = _trim(path)
        steps = _read(
            handler,
            root,
            op,
            lambda emit, root_entry: _Found(emit, op, root_entry, withdirs=withdirs),
            count=found,
            max_running=max_running,
            maxdepth=maxdepth,
            ils_kwargs=ils_kwargs,
            dir_root=withdirs and bool(root),
        )
        async with aclosing(steps):
            async for item in steps:
                yield item

    return Operation(run, FindState(directories_listed=0))


class _Reading(PathAction):
    """What the actions that only read share: each directory traversed counted as listed.

    A directory counts once visited, before it is listed: one at
    ``maxdepth``, or matched alone, counts too. ``root``, the root task's own
    entry, made, counts once its listing is read through: a file root's
    fails.
    """

    def __init__(self, emit_event: Any, op: Operation[Any, Any], root: Entry) -> None:
        super().__init__(emit_event)
        self.op = op
        self.root = root

    async def visit_child(self, entry: Entry, parent_value: Any) -> bool:
        if entry["type"] == "directory":
            self.op.state = _listed(self.op.state, entry)
        return True

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        async for _ in values:
            pass
        if entry is self.root:
            self.op.state = _listed(self.op.state, entry)


class _Found(_Reading):
    """find's and glob's action: each path matched emitted as it is listed.

    ``root`` is never emitted: `_read` asks its ``info`` instead. Without
    ``withdirs``, only files are. A path in ``found`` already, e.g. a glob's
    literal, is not emitted again.
    """

    def __init__(
        self,
        emit_event: Any,
        op: Operation[Any, Any],
        root: Entry,
        *,
        withdirs: bool,
        found: set[str] | None = None,
    ) -> None:
        super().__init__(emit_event, op, root)
        self.withdirs = withdirs
        self.found = found if found is not None else set()

    async def map(self, entry: Entry, match: bool, parent_value: Any) -> Any:
        if (
            match
            and entry is not self.root
            and (self.withdirs or entry["type"] != "directory")
            and entry["name"].rstrip("/") not in self.found
        ):
            await self._emit_event(entry)


def _listed(state: _S, entry: Entry) -> _S:
    """A state once a directory is counted as listed."""
    return replace(
        state,
        current=_trim(entry["name"]),
        directories_listed=(state.directories_listed or 0) + 1,
    )


async def _read(
    handler: TreeFileOperationHandler,
    path: str,
    op: Operation[Any, _S],
    action: Callable[[Any, Entry], PathAction],
    *,
    count: Callable[[_S, Entry], _S],
    max_running: int,
    maxdepth: int | None = None,
    patterns: Sequence[_Search] | None = None,
    ils_kwargs: dict[str, Any] | None = None,
    file_root: bool = True,
    dir_root: bool = False,
) -> AsyncGenerator[dict[str, Any] | Exception]:
    """Traverse a tree as the strategies that only read do, such as find, du and glob: each entry the action emits, or an error.

    ``action`` makes the traversal's action from its ``emit_event`` and the
    root's own entry. Each entry it emits is yielded, counted in the state by
    ``count``.

    It treats ``path`` itself as fsspec's ``_find`` and ``_glob`` do:

    - missing, it yields nothing;
    - a file, it yields the file's ``info`` entry if ``file_root``, else
      nothing;
    - a directory, it yields the directory's own ``info`` entry after its
      traversal, if ``dir_root`` and the operation is not cancelled.

    Any other failure of the root is a failed directory, yielded as any is,
    the state's ``current`` naming it.
    """
    engine = TreeTraversal(max_running=max_running, split=handler.segments)
    op._on_cancel = engine.cancel
    await handler.open()  # Its profile, read by _root_task.
    root_entry: Entry = {"name": path, "type": "directory"}
    root = _root_task(
        engine,
        root_entry,
        action(engine._emit_event, root_entry),
        handler,
        maxdepth=maxdepth,
        patterns=patterns,
        ils_kwargs=ils_kwargs,
    )
    try:
        async with aclosing(engine.events([root])) as events:
            async for event in events:
                if isinstance(event, tuple):  # A directory failed: (entry, error).
                    entry, error = event
                    op.state = _failed(op.state, _trim(entry["name"]))
                    yield error
                else:
                    op.state = count(op.state, event)
                    yield event
    except FileNotFoundError:
        return  # A missing root has nothing, as with fsspec's _find.
    except NotADirectoryError:
        if file_root:  # A file finds itself.
            entry = await _info(handler, path)
            op.state = count(op.state, entry)
            yield entry
        return
    except Exception as error:
        # The root's other failures are a failed directory, as any is.
        op.state = _failed(op.state, _trim(path), root=True)
        yield error
        return

    if dir_root and not op.cancelling():
        entry = await _info(handler, path)
        op.state = count(op.state, entry)
        yield entry


def _failed(state: _S, directory: str, *, root: bool = False) -> _S:
    """A state once a directory failed to list: its error.

    A directory below the root was counted as listed once visited. The root,
    never visited, is counted here, though failed.
    """
    return replace(
        state,
        current=directory,
        errors=state.errors + 1,
        directories_listed=(state.directories_listed or 0) + root,
    )


async def _info(handler: TreeFileOperationHandler, path: str) -> Entry:
    """A path's ``info``, holding a permit."""
    async with handler.operation_limit:
        return await handler.info(handler.entry(_trim(path)))
