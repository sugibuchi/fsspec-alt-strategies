# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A concurrent walk of a directory tree, as fsspec's ``walk`` answers.

`_tree_walk` yields each directory's ``(path, dirs, files)``, as fsspec's
``walk`` gives them, by `_walker._TreeWalk`, the walk under every tree
strategy. That reads each directory in parts, as its listing comes. So
`_tree_walk` gathers a directory's parts until its listing is whole, and
yields it in the order fsspec's ``walk`` promises: top down, a directory
before those below it; bottom up, after all of them.

This module depends on `_visit` and `_walker`, and `handler`, `operation` and
`paths` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal

from fsspec_alt_strategies.states import WalkState

from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _triples
from ._visit import _Pruned, _TopDown, _Visit
from ._walker import _FileRoot, _Part, _Skipped, _SubtreeListed, _TreeWalk

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from ..common.handler import StoragePath
    from ._walker import _Event
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


def _tree_walk(
    handler: TreeFileOperationHandler,
    path: StoragePath,
    *,
    maxdepth: int | None = None,
    topdown: bool = True,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    on_error: Literal["omit", "raise"] | Callable[[Exception], object] | None = None,
    ils_kwargs: dict[str, Any] | None = None,
) -> Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState]:
    """Walk a directory tree, listing several directories at a time.

    It yields each directory listed as ``(path, dirs, files)``, as fsspec's
    ``walk`` does with ``detail=True``: ``path`` itself first, then each
    directory below it. ``dirs`` and ``files`` map the names in the directory
    to their ``ils`` entries. An empty directory yields empty ones. A directory
    that fails to list yields its error instead, and the walk goes on with the
    others. The error names its directory in its note, and as its
    ``directory`` attribute. ``on_error``, below, may handle those errors
    instead, as with fsspec's ``walk``.

    The directories are listed depth first, as fsspec's ``walk`` lists them,
    several at a time. With ``topdown``, a directory comes before those below
    it, so each is listed on its own: a recursive listing passes a directory
    only after all below it. A subdirectory whose listing ends first waits
    for its parent's. Without, a directory comes after all of those below
    it, and the tree is listed by one ``ils_recursive``, unless ``maxdepth`` or
    ``descend`` prunes it. A listing broken off part way yields its error, as
    soon as it comes, and leaves out the directories it had not passed whole,
    as a walk bottom up that is cancelled does.

    It departs from fsspec's ``walk`` top down in one way: a caller may not
    prune by changing ``dirs`` in place, as the subdirectories are listed
    already. ``descend`` prunes instead.

    Once cancelled, it lists no new directory, and yields the directories of the
    listings in flight. Iterating it raises these errors, which end it:

    - FileNotFoundError, if ``path`` does not exist;
    - NotADirectoryError, if ``path`` is a file;
    - with ``on_error="raise"``, the error of the first directory that failed to
      list.

    Requests: top down, or when pruned, one ``ils`` per directory walked, as
    the handler's ``operation_limit`` allows. Bottom up otherwise, one
    ``ils_recursive`` of ``path``, which a storage without a recursive listing
    answers with one per directory.

    ``path`` is the directory the walk starts from, listed by the ``ils`` of
    ``handler``. ``maxdepth`` of 1 lists only ``path``. ``topdown`` puts a directory
    before those below it. ``descend``, called with the entry of each
    subdirectory found, lets the walk list it only if it returns True.
    ``ils_kwargs`` go to each ``ils``, as fsspec's ``walk`` passes its own on.

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

    visit: _Visit = _Pruned(maxdepth=maxdepth, descend=descend)
    if topdown:
        visit = _TopDown(visit)

    async def run(
        op: Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState],
    ) -> AsyncGenerator[tuple[str, dict[str, Any], dict[str, Any]] | Exception]:
        order = _Order(topdown)
        walk = _TreeWalk(
            handler,
            path,
            visit,
            ils_kwargs=ils_kwargs,
            stopping=op.cancelling,
            subtrees=not topdown,
        )
        events = _parts(walk.events())

        async with aclosing(events):
            async for event in events:
                for listing in order.ready(event):
                    if listing.error is not None and on_error == "raise":
                        raise listing.error
                    op.state = _walk_state(op.state, listing)
                    if (item := _walked(listing, on_error)) is not None:
                        yield item

    return Operation(run, WalkState(directories_listed=0))


async def _parts(
    events: AsyncGenerator[_Event],
) -> AsyncGenerator[_Part | _SubtreeListed]:
    """A walk's parts and subtrees, as fsspec's walk takes them: a root that is a file raises, and what is left is not walked."""
    async with aclosing(events):
        async for event in events:
            if isinstance(event, _FileRoot):
                raise event.error
            elif not isinstance(event, _Skipped):
                yield event


class _Order:
    """The whole listings a walk yields, in fsspec's order: each directory's parts held until its last, and listings held for another.

    - Top down, a directory comes before those below it. A subdirectory's
      listing may end before its parent's, which goes on in parts: it waits
      in ``early`` until its parent's is out, then comes with those waiting
      for it.
    - Bottom up, a directory comes after all below it: its whole listing
      waits in ``waiting`` until the walk tells its subtree is listed. A
      failed listing comes at once: the subtree below it may never end.

    So it holds whole listings, as only `_tree_walk` does, as fsspec's ``walk``
    gives a directory whole.
    """

    def __init__(self, topdown: bool) -> None:
        self.topdown = topdown
        # The entries of each directory whose listing has begun, until its last part.
        self.parts: dict[str, list[dict[str, Any]]] = {}
        # Top down: the whole listings waiting for their parent's, by parent,
        # and their directories, as one waiting holds back those below it too.
        self.early: dict[str, list[_Part]] = {}
        self.held: set[str] = set()
        # Bottom up: the whole listings waiting for their subtree to be listed.
        self.waiting: dict[str, _Part] = {}

    def ready(self, event: _Part | _SubtreeListed) -> list[_Part]:
        """The whole listings an event makes ready, in the order to yield them."""
        if isinstance(event, _SubtreeListed):
            listing = self.waiting.pop(event.directory, None)
            return [] if listing is None else [listing]
        if (listing := self._whole(event)) is None:
            return []
        elif not self.topdown and listing.error is None:
            self.waiting[listing.directory] = listing
            return []
        elif not self.topdown:
            return [listing]
        elif (
            parent := _parent(listing.directory)
        ) in self.parts or parent in self.held:
            # Its parent's listing has not ended, or waits itself: it comes first.
            self.early.setdefault(parent, []).append(listing)
            self.held.add(listing.directory)
            return []
        else:
            return self._released(listing)

    def _whole(self, part: _Part) -> _Part | None:
        """A directory's whole listing once its last part comes, its parts held till then, or None.

        A failed listing comes whole with its error and no entries.
        """
        self.parts.setdefault(part.directory, []).extend(part.entries)
        if not part.last:
            return None
        entries = self.parts.pop(part.directory)
        if part.error is not None:
            return _Part(part.directory, part.depth, [], part.error)
        else:
            return _Part(part.directory, part.depth, entries)

    def _released(self, listing: _Part) -> list[_Part]:
        """A listing out, then those that waited for it, and for them in turn: each after its parent."""
        ready = [listing]
        for out in ready:  # Grows as each releases those below it.
            self.held.discard(out.directory)
            ready += self.early.pop(out.directory, [])
        return ready


def _parent(directory: str) -> str:
    """A directory's parent, by its path: "/" for one of an absolute root's."""
    parent = directory.rpartition("/")[0]
    return (
        "/" if not parent and directory.startswith("/") and directory != "/" else parent
    )


def _walked(
    listing: _Part,
    on_error: Literal["omit", "raise"] | Callable[[Exception], object] | None,
) -> tuple[str, dict[str, Any], dict[str, Any]] | Exception | None:
    """What the walk yields of a whole listing: its directory, its error, or None, as ``on_error`` says."""
    if listing.error is None:
        return _triples(listing.directory, listing.entries, detail=True)
    if on_error is None:
        return listing.error
    if callable(on_error):
        on_error(listing.error)
    return None  # "omit", or handled by the callable: left out.


def _walk_state(state: WalkState, listing: _Part) -> WalkState:
    """The walk's state once a directory's listing is out."""
    return WalkState(
        errors=state.errors + (listing.error is not None),
        current=listing.directory,
        directories_listed=(state.directories_listed or 0) + 1,
        found=state.found + len(listing.entries),
    )
