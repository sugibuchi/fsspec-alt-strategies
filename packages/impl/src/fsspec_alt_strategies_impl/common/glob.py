# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Two tasks traversing a tree for a `PathAction`, interchangeably, as glob patterns prune it.

`VisitDirectory` lists one directory by ``ils``, and runs each item as a
task: the items' `map` run in parallel, once their directory's `map` is done.
`TraverseSubtree` takes a whole subtree's listing, as
``ils_recursive`` gives it, and acts on its paths as it streams,
one at a time. Both call the action alike: `map` on a path before what is
below it, `reduce` after. So either may traverse any subtree, and a
`VisitDirectory` hands a subtree the patterns take whole to a
`TraverseSubtree`, if asked to.

Patterns, `patterns._Search`s, prune: the directories descended into, and
the subtrees listed whole. A directory not descended into is visited only if
a pattern matches it, and then not listed: its `reduce` gets no values. A
file is visited only if a pattern matches it. Every path visited is acted on,
and told whether a pattern matches it: a directory may be visited only on the
way to others.

A listing holds a permit of the handler's ``operation_limit`` around each
step, never across the whole: see `_TraversalTask._listing`. ``maxdepth``
bounds the directories listed, as fsspec's walk counts it.

With ``expand``, a match takes all below it, as fsspec's ``_expand_path``
with ``recursive=True`` expands it: every path below is matched, and
``maxdepth`` counts from each match, not from the root. A path matched
without wildcards has ``maxdepth`` levels below it, as fsspec's find walks
it, and a pattern's match one less, as fsspec's glob counted that level
already. A match below another counts as the outer one's.

An error of the action ends the path's task. Below a `VisitDirectory`, the
other items run on, then its `reduce` is skipped, and the error goes up in an
ExceptionGroup, to the strategy running the traversal. A `TraverseSubtree`
stops at once. A listing's error ends the path's task alike, its directories
not reduced, but goes as ``on_error`` says: emitted, called back, left out
or raised. The root's is always raised.

This module depends on `handler`, `patterns` and `tree`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, Literal

from .tree import PathTask

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterable,
        AsyncIterator,
        Coroutine,
        Sequence,
    )

    from .handler import Entry, StorageHandler
    from .patterns import _Search
    from .tree import TreeTraversal

__all__: list[str] = []

OnError = Literal["omit", "raise"] | Callable[[Exception], object] | None
"""What becomes of a listing's error, as fsspec's walk takes ``on_error``: see `_TraversalTask._listing_failed`."""


class PathAction:
    """What an operation does on each path a traversal visits: the strategy is the task's.

    A directory goes through these steps:

    1. `map` acts on it;
    2. its listing gives its items, and `visit_child` prunes them further
       than the task does;
    3. a subtask traverses each item kept, the same steps;
    4. `reduce` reads the items' values as their subtasks end.

    An item's value is what `reduce` returned for it. Each step has a
    default, so a subclass writes only those it needs.

    The root is a path like any other: `map` and `reduce` are called on it,
    whether a file or a directory. Being no child, it is never given to
    `visit_child`.

    Which paths are visited is first the task's, e.g. within a ``maxdepth``
    or a glob's pattern. So are the errors: where `map` raises, the path's
    task ends. An error below a path is raised by its ``values``, through
    `reduce`, and the task handles it.
    """

    def __init__(self, emit_event: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        # The traversal's, which a strategy passes: one action serves every task.
        # `map` and `reduce` emit events to the traversal's caller by it,
        # waiting while too many are unread.
        self._emit_event = emit_event

    async def map(self, entry: Entry, match: bool, parent_value: Any) -> Any:
        """Act on a path, before its subtree is traversed: a value for `reduce`, or None.

        ``entry`` is the path's entry, as a listing or ``info`` gave it.
        ``match`` is whether the task's patterns match the path, True where it
        has none: a directory visited only on the way to others is not
        matched. A file below the root is visited only if matched.
        ``parent_value`` is what `map` returned for the directory the path is
        in, None for the traversal's root.
        """
        return None

    def descends(self, entry: Entry, value: Any) -> bool:
        """Whether the traversal goes below a directory, given what `map` returned for it: True by default.

        False reduces it with no values, unlisted, as a directory at a
        ``maxdepth``: e.g. one the action deleted whole.
        """
        return True

    async def visit_child(self, entry: Entry, parent_value: Any) -> bool:
        """Whether an item of a directory listed is traversed: False prunes it with all below it.

        Called after the directory's `map`, whose value is ``parent_value``,
        once on each item the task keeps, in the listing's order, before the
        item's subtask starts. True by default.
        """
        return True

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: AsyncGenerator[Any]
    ) -> Any:
        """Act on a path, reading the values of the items below it as they are traversed: its value, or None.

        ``match`` is as `map` was given it. ``value`` is what `map` returned
        for it. ``values`` yields the values of the items in it, in no order,
        as they are reduced: none for a file, nor for a directory not listed.
        An item whose listing failed, its error handled as ``on_error`` says,
        is never reduced: its value is what `failed` returned for it.

        ``values`` raises, once read through, the errors below: the path's
        own listing's, and those of the items. They end `reduce`, unless it
        catches them. What it leaves unread is read once it returns, and an
        error then still ends the path's task, its value lost.
        """
        async for _ in values:
            pass
        return None

    async def failed(self, entry: Entry) -> Any:
        """The value of a directory whose listing failed, for its parent's `reduce`: None by default.

        Its error is handled as ``on_error`` says, apart from this: e.g. a
        walk still names the directory in its parent's ``dirs``, as fsspec's
        does.
        """
        return None


class _TraversalTask(PathTask):
    """What both tasks share: the handler, the action, and the patterns pruning.

    Paths are compared by their segments, the handler's, never as strings: a listing may name a
    directory with a trailing delimiter, as an object store does, or escape
    its names. A listing's entries are passed on whole, as the storage gave
    them.
    """

    def __init__(
        self,
        engine: TreeTraversal,
        entry: Entry,
        action: PathAction,
        handler: StorageHandler,
        *,
        patterns: Sequence[_Search] | None = None,
        maxdepth: int | None = None,
        on_error: OnError = None,
        top: int | None = None,
        ils_kwargs: dict[str, Any] | None = None,
        parent_value: Any = None,
        expand: bool = False,
        above: float | None = None,
    ) -> None:
        super().__init__(engine, entry["name"])
        self.entry = entry
        # What map returned for the directory the path is in: None for a root.
        self.parent_value = parent_value
        # What map returned for the path, once called: its children's parent_value.
        self._value: Any = None
        self.action = action
        self.handler = handler
        self.patterns = patterns
        # The depth listed at most below the root, as fsspec's walk counts it:
        # 1 lists the root alone. With expand, below each match instead.
        self.maxdepth = maxdepth
        # Whether a match takes all below it: see the module's docstring.
        self.expand = expand
        # With expand, the levels left below the directory the path is in,
        # under a match: None where none is above it.
        self.above = above
        self.on_error = on_error
        # The root's segments counted: a path's depth is the segments it has more.
        self.top = len(self._segments(entry["name"])) if top is None else top
        # The options of each listing, as fsspec's walk passes its own on.
        self.ils_kwargs = ils_kwargs or {}
        # The error of the path's own listing, apart from the action's.
        self._listing_error: Exception | None = None
        # A recursive listing's prefix: see TraverseSubtree.
        self.prefix: str | None = None

    def _options(self) -> dict[str, Any]:
        """What a task below this one is made with, as it was."""
        return {
            "patterns": self.patterns,
            "maxdepth": self.maxdepth,
            "on_error": self.on_error,
            "top": self.top,
            "ils_kwargs": self.ils_kwargs,
            "expand": self.expand,
        }

    def _segments(self, path: str) -> tuple[str, ...]:
        """A path's segments, the handler's: equal for two names `equals` matches."""
        return tuple(self.handler.segments(path))

    async def _listing(self, recursive: bool) -> AsyncGenerator[Entry]:
        """The path's ``ils``, or with ``recursive`` its ``ils_recursive``, a permit of the handler's ``operation_limit`` held around each step.

        Never across the whole listing: a listing streaming into tasks waits
        for room among those pending, while they wait for permits. Its error
        is kept as ``_listing_error``, then raised.

        A root that is a file lists itself: its listing fails with
        NotADirectoryError.
        """
        handler = self.handler
        prefix = self.prefix
        if not recursive:
            entries = handler.ils(self.entry, **self.ils_kwargs)
        elif prefix is None:
            entries = handler.ils_recursive(self.entry, **self.ils_kwargs)
        else:
            narrowed = {"name": prefix}
            entries = handler.ils_recursive(narrowed, prefix=True, **self.ils_kwargs)
        root = self.entry["name"] if self._group is None else None
        try:
            while True:
                async with self.handler.operation_limit:
                    try:
                        entry = await anext(entries)
                        if (
                            root is not None
                            and entry["type"] != "directory"
                            and self.handler.equals(entry["name"], root)
                        ):
                            raise NotADirectoryError(root)
                    except StopAsyncIteration:
                        return
                    except Exception as error:
                        self._listing_error = error
                        raise
                yield entry
        finally:
            if (aclose := getattr(entries, "aclose", None)) is not None:
                await aclose()

    async def _listing_failed(self) -> None:
        """Handle the path's listing error as ``on_error`` says.

        The root's is raised whatever it says, as fsspec's walk raises it,
        and so is any with "raise". None emits ``(entry, error)`` as an event,
        the path's entry naming what failed, a callable is called with the
        error, and "omit" leaves it out.
        """
        error = self._listing_error
        assert error is not None
        if self._group is None or self.on_error == "raise":
            raise error
        elif self.on_error is None:
            await self._emit_event((self.entry, error))
        elif callable(self.on_error):
            self.on_error(error)

    async def prune(self, entries: AsyncIterable[Entry]) -> AsyncIterator[Entry]:
        """A directory's listing as traversed: without its own marker, each subdirectory once, only those to descend into or matched, and only the files matched.

        A storage may list a directory's marker object in it, and a
        subdirectory twice, e.g. as a prefix and as its marker. A
        subdirectory the patterns neither descend into nor match is dropped,
        with all below it: one matched alone is kept, as a glob reports it,
        but not listed. A file is kept only where a pattern matches it. The
        action's `PathAction.visit_child` prunes those kept further, given
        the directory's `map` value. The entries kept are as the listing gave
        them, as it streams: only the subdirectories' segments are held.
        """
        own = self._segments(self.entry["name"])
        left = self._left(self.entry, self.above)
        subdirs: set[tuple[str, ...]] = set()
        async for found in entries:
            key = self._segments(found["name"])
            if key == own:
                continue
            if found["type"] == "directory":
                if key in subdirs or not (
                    self._descend(found, left) or self._matched(found, left)
                ):
                    continue
                subdirs.add(key)  # Visited once, kept or not.
            elif not self._matched(found, left):
                continue
            if await self.action.visit_child(found, self._value):
                yield found

    def _left(self, entry: Entry, above: float | None) -> float | None:
        """With ``expand``, the levels below a path to traverse, under a match: None where it is under none.

        ``above`` is its directory's.
        """
        if not self.expand:
            return None
        if above is not None:
            return above - 1
        if self.patterns is None:
            plain = True
        else:
            hits = [p for p in self.patterns if p.matched(entry)]
            if not hits:
                return None
            plain = any(p.plain for p in hits)
        if self.maxdepth is None:
            return math.inf
        return self.maxdepth if plain else self.maxdepth - 1

    def _descend(self, entry: Entry, above: float | None = None) -> bool:
        """Whether a directory is listed: within ``maxdepth``, and as a pattern descends."""
        if self.expand:
            left = self._left(entry, above)
            if left is not None:
                return left > 0
        elif self.maxdepth is not None:
            depth = len(self._segments(entry["name"])) - self.top
            if depth >= self.maxdepth:
                return False
        return self.patterns is None or any(p.descend(entry) for p in self.patterns)

    def _matched(self, entry: Entry, above: float | None = None) -> bool:
        """Whether a pattern matches a path, or with ``expand`` a match above it: every path does where there is none."""
        if self.expand and above is not None:
            return True
        return self.patterns is None or any(p.matched(entry) for p in self.patterns)

    def _whole(self, entry: Entry, above: float | None = None) -> bool:
        """Whether all below a directory is traversed: no pattern prunes it, nor ``maxdepth``."""
        if self.expand and self._left(entry, above) == math.inf:
            return True  # A match's, with no maxdepth.
        if self.maxdepth is not None:
            return False  # A listing whole would go past it.
        return self.patterns is None or any(p.whole(entry) for p in self.patterns)


class VisitDirectory(_TraversalTask):
    """A directory listed by one ``ils`` of its handler, with a task per item.

    Each subdirectory is a `VisitDirectory` of its own, or, with ``use_one_listing``,
    a `TraverseSubtree` where the patterns take all below it.
    """

    def __init__(
        self,
        engine: TreeTraversal,
        entry: Entry,
        action: PathAction,
        handler: StorageHandler,
        *,
        use_one_listing: bool = False,
        **options: Any,
    ) -> None:
        super().__init__(engine, entry, action, handler, **options)
        self.use_one_listing = use_one_listing

    async def run(self) -> Any:
        entry = self.entry
        match = self._matched(entry, self.above)
        value = self._value = await self.action.map(entry, match, self.parent_value)
        if (
            entry.get("type") != "directory"
            or not self._descend(entry, self.above)
            or not self.action.descends(entry, value)
        ):
            # A file, a directory kept as matched, or one the action takes
            # whole: reduced with no values.
            return await _reduce(self.action, entry, match, value, _no_values())
        try:
            # Listed, its items visited and their subtasks run, as reduce reads.
            return await _reduce(self.action, entry, match, value, self._values())
        except Exception as error:
            if error is not self._listing_error:
                raise
            # The items taken ran: the directory, listed in part, is not reduced.
            await self._listing_failed()
            return await self.action.failed(entry)

    async def _values(self) -> AsyncGenerator[Any]:
        """The items' values, as their tasks end, then the listing's error, if any.

        Each item kept is visited, then its task made, as the listing
        streams. A task is taken once there is room among those pending: a
        wide directory's are never all in memory before they start.
        """
        async for item in self._run_tasks(
            self._subtree(e) async for e in self.prune(self._items())
        ):
            yield item
        if self._listing_error is not None:
            raise self._listing_error

    async def _items(self) -> AsyncIterator[Entry]:
        """The directory's listing, ended by its error, which `run` handles once the items taken ran."""
        try:
            async for entry in self._listing(recursive=False):
                yield entry
        except Exception as error:
            if error is not self._listing_error:
                raise

    def _subtree(self, entry: Entry) -> PathTask:
        """All below a subdirectory by one listing, where asked and whole, else listed alone."""
        left = self._left(self.entry, self.above)
        options = {**self._options(), "parent_value": self._value, "above": left}
        if (
            self.use_one_listing
            and entry["type"] == "directory"
            and self._whole(entry, left)
        ):
            return TraverseSubtree(
                self._engine, entry, self.action, self.handler, **options
            )
        else:
            # Its own class: a subclass's directories are each one of it.
            return type(self)(
                self._engine,
                entry,
                self.action,
                self.handler,
                use_one_listing=self.use_one_listing,
                **options,
            )


_UNREAD: Any = object()
"""A subtree's first entry, before its root's `PathAction.map` lets it be read."""


async def _no_values() -> AsyncGenerator[Any]:
    """The values of a path with nothing traversed below it: none."""
    return
    yield


async def _reduce(
    action: PathAction,
    entry: Entry,
    match: bool,
    value: Any,
    values: AsyncGenerator[Any],
) -> Any:
    """A path's `PathAction.reduce`, then what it left of ``values`` read: their errors raised."""
    async with aclosing(values):
        reduced = await action.reduce(entry, match, value, values)
        async for _ in values:
            pass
    return reduced


def _below(directory: tuple[str, ...], key: tuple[str, ...]) -> bool:
    """Whether a path's segments are below a directory's."""
    return len(directory) < len(key) and key[: len(directory)] == directory


class TraverseSubtree(_TraversalTask):
    """A subtree traversed from one listing of all below it, its handler's ``ils_recursive``, as it streams: no task per item.

    The listing gives each directory's descendants together, with nothing
    else between them, as ``ils_recursive`` promises. Each directory is a
    `_Span` of the listing, which maps it, acts on what is below it, and
    reduces it as the first path not below it comes. Only the directories
    above the path read are held, and their subdirectories listed but not
    reached yet.

    It takes more than the promise, as a listing in byte order gives it. A
    directory named without the delimiter may come apart from what is below
    it: ``a/b``, ``a/b.txt``, ``a/b/c``. A subdirectory listed is so held
    until the first path below it comes, or its parent's span ends. A
    directory not listed is made from the path below it, by `subdir_entry`,
    and one's marker listed inside its span is skipped.

    So the items are acted on one at a time, in the listing's order. An
    error ends the whole subtree's task: no directory open is reduced.
    """

    def __init__(
        self,
        engine: TreeTraversal,
        entry: Entry,
        action: PathAction,
        handler: StorageHandler,
        *,
        prefix: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(engine, entry, action, handler, **options)
        # A string all the paths traversed start with, e.g. a glob's "data/2024-":
        # the listing is then narrowed to it, by ils_recursive with prefix.
        self.prefix = prefix

    async def run(self) -> Any:
        # Listed as the task starts: one pending, or cancelled, holds nothing.
        entries = self._listing(recursive=True)
        try:
            # Its first entry read once the root's map lets it be.
            span = _Span(self, self.entry, _UNREAD, entries, self.above)
            return await span.map_reduce(self.parent_value)
        except Exception as error:
            if error is not self._listing_error:
                raise  # The action's.
            # No directory open is reduced, the subtree's included.
            await self._listing_failed()
            return await self.action.failed(self.entry)
        finally:
            await entries.aclose()

    def subdir_entry(self, root: Entry, entry: Entry) -> Entry:
        """The entry of the directory in ``root`` that ``entry`` is below, made.

        E.g. ``{"name": "a/b/c"}`` for ``a/b`` and ``a/b/c/d``.
        """
        depth = len(self._segments(root["name"]))
        name = self.handler.join(root["name"], self._segments(entry["name"])[depth])
        return {"name": name, "size": 0, "type": "directory"}


class _Span:
    """A directory, and the run of a listing below it, starting at ``next``.

    `map_reduce` maps the directory, then reduces it, as its values act on
    each path below it, a subdirectory by a span of its own. It returns its
    value. Its ``next`` is then the first entry not below it, for its parent
    to go on from.
    """

    def __init__(
        self,
        task: TraverseSubtree,
        root: Entry,
        next: Entry | None,
        entries: AsyncIterator[Entry],
        above: float | None = None,
    ) -> None:
        self.task = task
        self.root = root
        self.next = next
        self.entries = entries
        # With expand, the levels left below the directory's parent, and its own.
        self.above = above
        self.left = task._left(root, above)
        # What map returned for the directory: its paths' parent_value.
        self.value: Any = None

    async def map_reduce(self, parent_value: Any) -> Any:
        task, root = self.task, self.root
        match = task._matched(root, self.above)
        value = self.value = await task.action.map(root, match, parent_value)
        directory = root.get("type") == "directory"
        descend = task._descend(root, self.above)
        if directory and descend and task.action.descends(root, value):
            if self.next is _UNREAD:
                self.next = await anext(self.entries, None)
            # Its paths visited and traversed, as reduce reads.
            return await _reduce(task.action, root, match, value, self._values())
        if self.next is _UNREAD:
            self.next = None  # The subtree's root, taken whole: none listed.
        elif directory:
            # A directory matched alone, or taken whole: reduced with no
            # values, nothing below.
            self.next = await self._skip()
        return await _reduce(task.action, root, match, value, _no_values())

    async def _values(self) -> AsyncGenerator[Any]:
        """The values of the paths in the directory, as the listing gives them: ``next`` then follows them."""
        task, root, value = self.task, self.root, self.value
        key = task._segments(root["name"])
        # The subdirectories listed, held until the first path below each
        # comes: in byte order, "a/b.txt" comes between "a/b" and "a/b/c".
        # One the action pruned is held as None.
        held: dict[tuple[str, ...], Entry | None] = {}
        next = self.next
        while next is not None:
            found = task._segments(next["name"])
            in_root = len(found) == len(key) + 1
            if found == key:
                next = await anext(self.entries, None)  # Its own marker.
            elif not _below(key, found):
                break
            elif in_root and next["type"] != "directory":
                # Else pruned, as a file is.
                if task._matched(next, self.left) and await task.action.visit_child(
                    next, value
                ):
                    item = await task.action.map(next, True, value)
                    yield await _reduce(task.action, next, True, item, _no_values())
                next = await anext(self.entries, None)
            elif in_root:
                # A subdirectory listed, or listed again, e.g. as its marker.
                await self._hold(held, found, next)
                next = await anext(self.entries, None)
            else:
                # A path below a subdirectory, listed or made from it.
                span, after = await self._enter(held, found[: len(key) + 1], next)
                if span is None:
                    next = after
                else:
                    yield await span.map_reduce(value)
                    next = span.next
        self.next = next
        for subdir in held.values():  # Listed, with nothing below.
            if subdir is not None:
                span = _Span(task, subdir, None, self.entries, self.left)
                yield await span.map_reduce(value)

    async def _enter(
        self,
        held: dict[tuple[str, ...], Entry | None],
        sub: tuple[str, ...],
        next: Entry,
    ) -> tuple[_Span | None, Entry | None]:
        """The span of the subdirectory ``next`` is below, to traverse: else None, with the entry after it.

        One pruned is skipped, with all below it. So is one whose name the
        handler refuses, e.g. "..": its error is emitted, as a listing's is.
        """
        try:
            span, kept = await self._subspan(held, sub, next)
        except ValueError as refused:
            await self.task.action._emit_event((next, refused))
            return None, await self._skip_below(sub)
        if not kept:
            return None, await span._skip()
        return span, None

    async def _hold(
        self,
        held: dict[tuple[str, ...], Entry | None],
        sub: tuple[str, ...],
        subdir: Entry,
    ) -> None:
        """Hold a subdirectory listed, visited once, in the listing's order: as None if pruned."""
        if sub not in held:
            held[sub] = subdir if await self._visit(subdir) else None

    async def _subspan(
        self,
        held: dict[tuple[str, ...], Entry | None],
        sub: tuple[str, ...],
        next: Entry,
    ) -> tuple[_Span, bool]:
        """The span of the subdirectory ``next`` is below, from ``next``, and whether it is traversed.

        A subdirectory held was visited as listed. One not listed is made,
        and visited now.
        """
        made = self.task.subdir_entry(self.root, next)
        if sub in held:
            subdir = held.pop(sub)
        else:
            subdir = made if await self._visit(made) else None
        span = _Span(self.task, subdir or made, next, self.entries, self.left)
        return span, subdir is not None

    async def _visit(self, subdir: Entry) -> bool:
        """Whether a subdirectory is traversed: as the patterns, then the action, say."""
        task = self.task
        if not (task._descend(subdir, self.left) or task._matched(subdir, self.left)):
            return False
        return await task.action.visit_child(subdir, self.value)

    async def _skip_below(self, sub: tuple[str, ...]) -> Entry | None:
        """Read past the paths below ``sub``, whose directory has no entry: the first entry not below it."""
        next = await anext(self.entries, None)
        while (
            next is not None
            and _below(sub[:-1], found := self.task._segments(next["name"]))
            and found[: len(sub)] == sub
        ):
            next = await anext(self.entries, None)
        return next

    async def _skip(self) -> Entry | None:
        """Read past the span unread: the first entry not below the directory."""
        key = self.task._segments(self.root["name"])
        next = self.next
        while next is not None:
            found = self.task._segments(next["name"])
            if found != key and not _below(key, found):
                break
            next = await anext(self.entries, None)
        return next
