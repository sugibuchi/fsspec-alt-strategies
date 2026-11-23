# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Two tasks traversing a tree for a `PathAction`, interchangeably, as glob patterns prune it.

`VisitDirectory` lists one directory by ``ils``, and runs each item as a
task: the items' `map` run in parallel, once their directory's `map` is done.
`TraverseSubtree` takes a whole subtree's listing, as
``ils(..., recursive=True)`` gives it, and acts on its paths as it streams,
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

An error of the action ends the path's task. Below a `VisitDirectory`, the
other items run on, then its `reduce` is skipped, and the error goes up in an
ExceptionGroup, to the strategy running the traversal. A `TraverseSubtree`
stops at once. A listing's error ends the path's task alike, its directories
not reduced, but goes as ``on_error`` says: emitted, called back, left out
or raised. The root's is always raised.

This module depends on `handler`, `patterns` and `tree`.
"""

from __future__ import annotations

from collections.abc import Callable
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

    A task calls `map` on a path before traversing what is below it, and
    `reduce` after: on a file at once, with no values. An item's value is what
    `reduce` returned for it. Each defaults to None, so a subclass writes only
    those it needs.

    The root is a path like any other: `map` and `reduce` are called on it,
    whether a file or a directory.

    Which paths are visited is the task's, e.g. within a ``maxdepth`` or a
    glob's pattern: an action never prunes. So are its errors: where `map`
    raises, or anything below the path fails, `reduce` is not called, and the
    task handles the error.
    """

    def __init__(self, emit_event: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        # The traversal's, which a strategy passes: one action serves every task.
        # `map` and `reduce` emit events to the traversal's caller by it,
        # waiting while too many are unread.
        self._emit_event = emit_event

    async def map(self, entry: Entry, match: bool) -> Any:
        """Act on a path, before its subtree is traversed: a value for `reduce`, or None.

        ``entry`` is the path's entry, as a listing or ``info`` gave it.
        ``match`` is whether the task's patterns match the path, True where it
        has none: a directory visited only on the way to others is not
        matched. A file below the root is visited only if matched.
        """
        return None

    async def reduce(
        self, entry: Entry, match: bool, value: Any, values: list[Any]
    ) -> Any:
        """Act on a path, after its subtree is traversed: its value, or None.

        ``match`` is as `map` was given it. ``value`` is what `map` returned
        for it, and ``values`` the values of the items in it, in no order:
        none for a file. An item whose listing failed, its error handled as
        ``on_error`` says, is never reduced: its value is what `failed`
        returned for it.
        """
        return None

    async def listed(self, entry: Entry) -> None:
        """A directory's listing is whole, its items acted on: called before its `reduce`.

        Never for a directory reduced unlisted, e.g. at ``maxdepth``, nor one
        that failed: e.g. a strategy counts the directories listed by it.
        """

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
    ) -> None:
        super().__init__(engine, entry["name"])
        self.entry = entry
        self.action = action
        self.handler = handler
        self.patterns = patterns
        # The depth listed at most below the root, as fsspec's walk counts it:
        # 1 lists the root alone.
        self.maxdepth = maxdepth
        self.on_error = on_error
        # The root's segments counted: a path's depth is the segments it has more.
        self.top = len(self._segments(entry["name"])) if top is None else top
        # The options of each listing, as fsspec's walk passes its own on.
        self.ils_kwargs = ils_kwargs or {}
        # The error of the path's own listing, apart from the action's.
        self._listing_error: Exception | None = None

    def _options(self) -> dict[str, Any]:
        """What a task below this one is made with, as it was."""
        return {
            "patterns": self.patterns,
            "maxdepth": self.maxdepth,
            "on_error": self.on_error,
            "top": self.top,
            "ils_kwargs": self.ils_kwargs,
        }

    def _segments(self, path: str) -> tuple[str, ...]:
        """A path's segments, the handler's: equal for two names `equals` matches."""
        return tuple(self.handler.segments(path))

    async def _listing(self, recursive: bool) -> AsyncGenerator[Entry]:
        """The path's ``ils``, a permit of the handler's ``operation_limit`` held around each step.

        Never across the whole listing: a listing streaming into tasks waits
        for room among those pending, while they wait for permits. Its error
        is kept as ``_listing_error``, then raised.

        A root that is a file lists itself: its listing fails with
        NotADirectoryError.
        """
        entries = self.handler.ils(self.entry, recursive=recursive, **self.ils_kwargs)
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
        entries kept are as the listing gave them, as it streams: only the
        subdirectories' segments are held.
        """
        own = self._segments(self.entry["name"])
        subdirs: set[tuple[str, ...]] = set()
        async for found in entries:
            key = self._segments(found["name"])
            if key == own:
                continue
            if found["type"] == "directory":
                if key in subdirs or not (self._descend(found) or self._matched(found)):
                    continue
                subdirs.add(key)
            elif not self._matched(found):
                continue
            yield found

    def _descend(self, entry: Entry) -> bool:
        """Whether a directory is listed: within ``maxdepth``, and as a pattern descends."""
        if self.maxdepth is not None:
            depth = len(self._segments(entry["name"])) - self.top
            if depth >= self.maxdepth:
                return False
        return self.patterns is None or any(p.descend(entry) for p in self.patterns)

    def _matched(self, entry: Entry) -> bool:
        """Whether a pattern matches a path: every path does where there is none."""
        return self.patterns is None or any(p.matched(entry) for p in self.patterns)

    def _whole(self, entry: Entry) -> bool:
        """Whether all below a directory is traversed: no pattern prunes it, nor ``maxdepth``."""
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
        match = self._matched(entry)
        value = await self.action.map(entry, match)
        if entry.get("type") != "directory" or not self._descend(entry):
            # A file, or a directory kept as matched: reduced with no values.
            values = []
        else:
            # Each item's task is made as the listing streams, and taken once
            # there is room among those pending: a wide directory's are never
            # all in memory before they start.
            values = [
                value
                async for value in self._run_tasks(
                    self._subtree(e) async for e in self.prune(self._items())
                )
            ]
            if self._listing_error is not None:
                # The items taken ran: the directory, listed in part, is not reduced.
                await self._listing_failed()
                return await self.action.failed(entry)
            await self.action.listed(entry)
        return await self.action.reduce(entry, match, value, values)

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
        if self.use_one_listing and entry["type"] == "directory" and self._whole(entry):
            return TraverseSubtree(
                self._engine, entry, self.action, self.handler, **self._options()
            )
        else:
            # Its own class: a subclass's directories are each one of it.
            return type(self)(
                self._engine,
                entry,
                self.action,
                self.handler,
                use_one_listing=self.use_one_listing,
                **self._options(),
            )


def _below(directory: tuple[str, ...], key: tuple[str, ...]) -> bool:
    """Whether a path's segments are below a directory's."""
    return len(directory) < len(key) and key[: len(directory)] == directory


class TraverseSubtree(_TraversalTask):
    """A subtree traversed from one listing of all below it, its handler's ``ils(..., recursive=True)``, as it streams: no task per item.

    The listing must give each directory's descendants together, with nothing
    else between them. The handler's listing answers for it, as any of these
    orders does:

    - the byte order of names, as S3, GCS and Azure Blob list objects, or as
      fsspec's ``find(withdirs=True)`` gives them;
    - the order of segments, as ``sorted(..., key=handler.segments)`` gives it;
    - depth first, a directory at a time.

    In byte order, a directory named without the delimiter comes apart from
    what is below it: ``a/b``, ``a/b.txt``, ``a/b/c``. A subdirectory listed
    is so held until the first path below it comes, or its parent's span
    ends, with nothing below it.

    Each directory is a `_Span` of the listing, which maps it, acts on what
    is below it, and reduces it as the first path not below it comes. Only
    the directories above the path read are held, and their subdirectories
    listed but not reached yet.

    A directory not listed, as an object store lists only files, is made from
    the path below it, by `subdir_entry`. A directory's marker listed inside
    its span is skipped.

    So the items are acted on one at a time, in the listing's order. An
    error ends the whole subtree's task: no directory open is reduced.
    """

    async def run(self) -> Any:
        # Listed as the task starts: one pending, or cancelled, holds nothing.
        entries = self._listing(recursive=True)
        try:
            first = await anext(entries, None)
            value, _ = await _Span(self, self.entry, first, entries).map_reduce()
            return value
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

    `map_reduce` maps the directory, acts on each path below it, a
    subdirectory by a span of its own, and reduces it. It returns its value,
    and the first entry not below it, for its parent to go on from.
    """

    def __init__(
        self,
        task: TraverseSubtree,
        root: Entry,
        next: Entry | None,
        entries: AsyncIterator[Entry],
    ) -> None:
        self.task = task
        self.root = root
        self.next = next
        self.entries = entries

    async def map_reduce(self) -> tuple[Any, Entry | None]:
        task, root = self.task, self.root
        match = task._matched(root)
        value = await task.action.map(root, match)
        if root.get("type") != "directory":
            return await task.action.reduce(root, match, value, []), self.next
        if not task._descend(root):
            # A directory matched alone: reduced with no values, nothing below.
            return await task.action.reduce(root, match, value, []), await self._skip()

        key = task._segments(root["name"])
        values: list[Any] = []
        # The subdirectories listed, held until the first path below each
        # comes: in byte order, "a/b.txt" comes between "a/b" and "a/b/c".
        held: dict[tuple[str, ...], Entry] = {}
        next = self.next
        while next is not None:
            found = task._segments(next["name"])
            in_root = len(found) == len(key) + 1
            if found == key:
                next = await anext(self.entries, None)  # Its own marker.
            elif not _below(key, found):
                break
            elif in_root and next["type"] != "directory":
                if task._matched(next):  # Else pruned, as a file is.
                    item = await task.action.map(next, True)
                    values.append(await task.action.reduce(next, True, item, []))
                next = await anext(self.entries, None)
            elif in_root:
                # A subdirectory listed, or listed again, e.g. as its marker.
                held.setdefault(found, next)
                next = await anext(self.entries, None)
            else:
                # A path below a subdirectory, listed or made from it.
                sub = found[: len(key) + 1]
                subdir = held.pop(sub, None) or task.subdir_entry(root, next)
                item, next = await self._below(subdir, next)
                values.extend(item)
        for subdir in held.values():  # Listed, with nothing below.
            item, _ = await self._below(subdir, None)
            values.extend(item)
        await task.action.listed(root)
        return await task.action.reduce(root, match, value, values), next

    async def _below(
        self, subdir: Entry, next: Entry | None
    ) -> tuple[list[Any], Entry | None]:
        """A subdirectory's span, from ``next``: its value, if not pruned, and the first entry not below it."""
        task = self.task
        span = _Span(task, subdir, next, self.entries)
        if task._descend(subdir) or task._matched(subdir):
            item, next = await span.map_reduce()
            return [item], next
        return [], await span._skip()  # Pruned, with all below it.

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
