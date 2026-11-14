# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive find that lists several directories at a time.

`_tree_find` is an alternative to fsspec's ``_find``, with the same paths.
fsspec's walks a tree one directory at a time, unless the filesystem lists
whole trees. `_tree_find` lists the tree as `_tree_walk` does: by one recursive
``ils``, or a directory per request, several at a time.

This module depends on `_visit` and `_walker`, and `handler`, `operation` and `paths` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar

from fsspec_alt_strategies.states import DuState, FindState, GlobState

from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim
from ._visit import _Pruned
from ._walker import _FileRoot, _Part, _TreeWalk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    from ..common.handler import StorageHandler, StoragePath
    from ._visit import _Visit
    from ._walker import _Event
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


def _tree_find(
    handler: TreeFileOperationHandler,
    path: StoragePath,
    *,
    maxdepth: int | None = None,
    withdirs: bool = False,
    ils_kwargs: dict[str, Any] | None = None,
) -> Operation[dict[str, Any], FindState]:
    """Find the files in a tree, listing several directories at a time.

    It finds the paths fsspec's ``_find`` finds. A missing ``path`` finds
    nothing, and a file finds itself. A directory that fails to list yields its
    error, and the rest of the tree is still searched. Without ``maxdepth``, the
    tree's one recursive listing broken off is resumed a directory per request.

    It yields each path as its directory's listing arrives, so the paths are not
    in order. fsspec's ``_find`` sorts them, by name. A storage listing a
    directory twice, e.g. as a prefix and as its marker object, yields it once.

    It wins over fsspec's ``_find`` on storages that list one directory per
    request, with many directories: their listings overlap. It gains nothing on
    object stores, which list a whole tree in one paginated request: it sends
    that same listing.

    Once cancelled, it lists no new directory, and yields the paths of the
    listings in flight.

    Requests: one ``ils_recursive`` of ``path``, which a storage without a
    recursive listing answers with one per directory. With ``maxdepth``, one
    ``ils`` per directory instead, as the handler's ``operation_limit``
    allows. One ``info`` more for ``path`` itself, if a file, or a directory
    with ``withdirs``.

    ``path`` is the directory searched on ``handler``, by its ``ils`` and ``info``.
    ``maxdepth`` of 1 finds only the files in ``path``. ``withdirs`` finds
    directories too, ``path`` included. ``ils_kwargs`` go to each ``ils``, as
    fsspec's ``find`` passes its own on.

    The operation yields the ``ils`` entry of each path found, or the error
    of a directory that failed to list, and its state is a `FindState`. The call
    raises ValueError if ``maxdepth`` is less than 1.
    """
    _check_maxdepth(maxdepth)

    async def run(
        op: Operation[dict[str, Any], FindState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        root = _trim(path)
        steps = _find_listings(
            handler,
            root,
            _Pruned(maxdepth=maxdepth),
            ils_kwargs=ils_kwargs,
            stopping=op.cancelling,
            file_root=True,
            dir_root=withdirs and bool(root),
        )
        state = op.state

        async with aclosing(steps):
            async for listing in steps:
                found = [
                    e for e in listing.entries if withdirs or e["type"] != "directory"
                ]
                op.state = state = _listed(
                    state, listing, found=state.found + len(found)
                )

                if listing.error is not None:
                    yield listing.error

                for entry in found:
                    yield entry

    return Operation(run, FindState(directories_listed=0))


_S = TypeVar("_S", FindState, DuState, GlobState)


def _listed(state: _S, listing: _Part, **counts: int) -> _S:
    """A state once a part of a listing is out, with ``counts`` its own: its error, and its directory once its last part comes.

    Depth 0 is the root's own entry, which is no directory listed.
    """
    return replace(
        state,
        errors=state.errors + (listing.error is not None),
        current=listing.directory,
        directories_listed=(state.directories_listed or 0)
        + (listing.depth > 0 and listing.last),
        **counts,
    )


async def _find_listings(
    handler: TreeFileOperationHandler,
    path: str,
    visit: _Visit,
    *,
    ils_kwargs: dict[str, Any] | None = None,
    stopping: Callable[[], object] | None = None,
    file_root: bool = False,
    dir_root: bool = False,
) -> AsyncGenerator[_Part]:
    """Walk as the strategies that only read do, such as find, du and glob.

    It yields each directory's listing in parts, as `_TreeWalk` does, but
    treats ``path`` itself as fsspec's ``_find`` and ``_glob`` do:

    - missing, it yields nothing;
    - a file, it yields the file's ``info`` entry if ``file_root``, else
      nothing;
    - a directory, it yields the directory's own ``info`` entry after its
      listing's last part, if ``dir_root`` and the walk is not stopping.

    The root's own entry comes as a `_Part` of depth 0, which is no listing.
    So the strategies filter and count it as any entry, but count no directory
    listed for it.
    """
    root = _trim(path)
    listings = _TreeWalk(
        handler, root, visit, ils_kwargs=ils_kwargs, stopping=stopping
    ).events()

    async with aclosing(listings):
        try:
            first = await anext(listings)
        except StopAsyncIteration:
            return  # Stopping before the root was listed.
        except FileNotFoundError:
            # Only the root's own listing raises it: any later one is a part's.
            return
        if isinstance(first, _FileRoot):
            if file_root:
                yield await _file_entry(handler, root, first)
            return

        async for event in _then(first, listings):
            if isinstance(event, _Part):
                yield event
                if event.directory == root and event.last:
                    async for own in _root_entry(handler, root, dir_root, stopping):
                        yield own


async def _then(first: _Event, rest: AsyncIterator[_Event]) -> AsyncGenerator[_Event]:
    """An event taken already, then the rest."""
    yield first
    async for event in rest:
        yield event


async def _file_entry(handler: StorageHandler, root: str, file: _FileRoot) -> _Part:
    """A root that is a file, as a listing of depth 0: its entry as its listing gave it, else asked."""
    entry = getattr(file.error, "entry", None)
    if entry is None:
        async with handler.operation_limit:
            entry = await handler.info(handler.entry(root))
    return _Part(root, 0, [entry])


async def _root_entry(
    handler: StorageHandler,
    root: str,
    dir_root: bool,
    stopping: Callable[[], object] | None,
) -> AsyncGenerator[_Part]:
    """The root directory's own ``info`` entry, as a listing of depth 0, if asked."""
    if dir_root and not (stopping and stopping()):
        async with handler.operation_limit:
            entry = await handler.info(handler.entry(root))
        yield _Part(root, 0, [entry])
