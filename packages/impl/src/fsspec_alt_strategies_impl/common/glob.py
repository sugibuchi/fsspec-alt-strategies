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

An error, of ``ils`` or of the action, ends the path's task. Below a
`VisitDirectory`, the other items run on, then its `reduce` is skipped, and
the error goes up in an ExceptionGroup, to the strategy running the
traversal. A `TraverseSubtree` stops at once.

This module depends on `handler`, `patterns` and `tree`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .tree import PathTask

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine, Sequence

    from .handler import Entry, StorageHandler
    from .patterns import _Search
    from .tree import TreeTraversal

__all__: list[str] = []


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
        none for a file.
        """
        return None


class _TraversalTask(PathTask):
    """What both tasks share: the handler, the action, and the patterns pruning.

    Paths are compared by the handler, never as strings: a listing may name a
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
    ) -> None:
        super().__init__(engine, entry["name"])
        self.entry = entry
        self.action = action
        self.handler = handler
        self.patterns = patterns

    def _key(self, path: str) -> tuple[str, ...]:
        """A path's segments, as the handler splits them: equal for two names `equals` matches."""
        return tuple(self.handler._split(path))

    def prune(self, entries: list[Entry]) -> list[Entry]:
        """A directory's listing as traversed: without its own marker, each subdirectory once, only those to descend into or matched, and only the files matched.

        A storage may list a directory's marker object in it, and a
        subdirectory twice, e.g. as a prefix and as its marker. A
        subdirectory the patterns neither descend into nor match is dropped,
        with all below it: one matched alone is kept, as a glob reports it,
        but not listed. A file is kept only where a pattern matches it. The
        entries kept are as the listing gave them.
        """
        kept: list[Entry] = []
        subdirs: set[tuple[str, ...]] = set()
        for found in entries:
            if self.handler.equals(found["name"], self.entry["name"]):
                continue
            if found["type"] == "directory":
                key = self._key(found["name"])
                if key in subdirs or not (self._descend(found) or self._matched(found)):
                    continue
                subdirs.add(key)
            elif not self._matched(found):
                continue
            kept.append(found)
        return kept

    def _descend(self, entry: Entry) -> bool:
        return self.patterns is None or any(p.descend(entry) for p in self.patterns)

    def _matched(self, entry: Entry) -> bool:
        """Whether a pattern matches a path: every path does where there is none."""
        return self.patterns is None or any(p.matched(entry) for p in self.patterns)

    def _whole(self, entry: Entry) -> bool:
        """Whether all below a directory is traversed: no pattern prunes it."""
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
        patterns: Sequence[_Search] | None = None,
        use_one_listing: bool = False,
    ) -> None:
        super().__init__(engine, entry, action, handler, patterns=patterns)
        self.use_one_listing = use_one_listing

    async def run(self) -> Any:
        entry = self.entry
        match = self._matched(entry)
        value = await self.action.map(entry, match)
        if entry.get("type") != "directory" or not self._descend(entry):
            # A file, or a directory kept as matched: reduced with no values.
            entries = []
        else:
            entries = [e async for e in self.handler.ils(entry)]
            entries = self.prune(entries)
        values = await self._run_tasks([self._subtree(e) for e in entries])
        return await self.action.reduce(entry, match, value, values)

    def _subtree(self, entry: Entry) -> PathTask:
        """All below a subdirectory by one listing, where asked and whole, else listed alone."""
        if self.use_one_listing and entry["type"] == "directory" and self._whole(entry):
            return TraverseSubtree(
                self._engine,
                entry,
                self.action,
                self.handler,
                patterns=self.patterns,
            )
        else:
            return VisitDirectory(
                self._engine,
                entry,
                self.action,
                self.handler,
                patterns=self.patterns,
                use_one_listing=self.use_one_listing,
            )


def _below(directory: tuple[str, ...], key: tuple[str, ...]) -> bool:
    """Whether a path's key is below a directory's."""
    return len(directory) < len(key) and key[: len(directory)] == directory


class TraverseSubtree(_TraversalTask):
    """A subtree traversed from one listing of all below it, its handler's ``ils(..., recursive=True)``, as it streams: no task per item.

    The listing must give each directory's descendants together, with nothing
    else between them. The handler's listing answers for it, as any of these
    orders does:

    - the byte order of names, a directory either not listed or named with
      the trailing delimiter, as S3, GCS and Azure Blob list objects;
    - the order of segments, as ``sorted(..., key=handler._split)`` gives it;
    - depth first, a directory at a time.

    Names in byte order with a directory named without the delimiter, as
    fsspec's ``find(withdirs=True)`` gives them, do not: ``a/b``, ``a/b.txt``,
    ``a/b/c`` puts a file between ``a/b`` and what is below it. A directory
    so listed apart is acted on twice. A handler whose listing is so sorts
    it by segments, appends the delimiter, or leaves the directories out.

    Each directory is a `_Span` of the listing, which maps it, acts on what
    is below it, and reduces it as the first path not below it comes. Only
    the directories above the path read are held.

    A directory not listed, as an object store lists only files, is made from
    the path below it, by `subdir_entry`. A directory's marker listed inside
    its span is skipped.

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
        patterns: Sequence[_Search] | None = None,
    ) -> None:
        super().__init__(engine, entry, action, handler, patterns=patterns)

    async def run(self) -> Any:
        # Listed as the task starts: one pending, or cancelled, holds nothing.
        entries = self.handler.ils(self.entry, recursive=True)
        try:
            first = await anext(entries, None)
            value, _ = await _Span(self, self.entry, first, entries).map_reduce()
            return value
        finally:
            if (aclose := getattr(entries, "aclose", None)) is not None:
                await aclose()

    def subdir_entry(self, root: Entry, entry: Entry) -> Entry:
        """The entry of the directory in ``root`` that ``entry`` is below, made.

        E.g. ``{"name": "a/b/c"}`` for ``a/b`` and ``a/b/c/d``.
        """
        depth = len(self._key(root["name"]))
        name = self.handler.join(root["name"], self._key(entry["name"])[depth])
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

        key = task._key(root["name"])
        values: list[Any] = []
        next = self.next
        while next is not None:
            found = task._key(next["name"])
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
            else:
                # A subdirectory listed, or made from a path below it. Its span
                # starts at next either way: a subdirectory listed is its marker.
                subdir = next if in_root else task.subdir_entry(root, next)
                if task._descend(subdir) or task._matched(subdir):
                    span = _Span(task, subdir, next, self.entries)
                    item, next = await span.map_reduce()
                    values.append(item)
                else:
                    # Pruned, with all below it.
                    next = await _Span(task, subdir, next, self.entries)._skip()
        return await task.action.reduce(root, match, value, values), next

    async def _skip(self) -> Entry | None:
        """Read past the span unread: the first entry not below the directory."""
        key = self.task._key(self.root["name"])
        next = self.next
        while next is not None:
            found = self.task._key(next["name"])
            if found != key and not _below(key, found):
                break
            next = await anext(self.entries, None)
        return next
