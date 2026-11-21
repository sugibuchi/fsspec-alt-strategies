# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Two tasks traversing a tree for a `PathAction`, interchangeably, as glob patterns prune it.

`VisitDirectory` lists one directory by ``ils``. `TraverseSubtree` takes a
whole subtree's listing, as ``ils(..., recursive=True)`` gives it, and holds
it. Both call the action alike: `map` on a path before what is below it,
`reduce` after. Both run each item of a directory as a task, so the items'
`map` run in parallel, once their directory's `map` is done. So
either may traverse any subtree, and a `VisitDirectory` hands a subtree the
patterns take whole to a `TraverseSubtree`, if asked to.

Patterns, `patterns._Search`s, prune: the directories descended into, and
the subtrees listed whole. A directory not descended into is visited only if
a pattern matches it, and then not listed: its `reduce` gets no values.
Every path visited is acted on, and told whether a pattern matches it: a path
may be visited only on the way to others.

An error, of ``ils`` or of the action, ends the path's task: what is below it
runs on, then its directory's `reduce` is skipped, and the error goes up in an
ExceptionGroup, to the strategy running the traversal.

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
        has none: a path visited only on the way to others is not matched.
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
        """A directory's listing as traversed: without its own marker, each subdirectory once, and only those to descend into or matched.

        A storage may list a directory's marker object in it, and a
        subdirectory twice, e.g. as a prefix and as its marker. A
        subdirectory the patterns neither descend into nor match is dropped,
        with all below it: one matched alone is kept, as a glob reports it,
        but not listed. The entries kept are as the listing gave them.
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
                self.handler.ils(entry, recursive=True),
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


class TraverseSubtree(_TraversalTask):
    """A subtree traversed from one listing of all below it, in any order.

    The listing is read whole before any path is acted on, as a path's
    descendants may come before it: the subtree is held in memory. A
    directory not listed, as an object store lists only files, is made from
    the paths below it, by the handler's ``join``. Then each directory below
    is a `TraverseSubtree` of its own, with no listing: it runs its entries,
    as `VisitDirectory` does, from what its root holds.
    """

    def __init__(
        self,
        engine: TreeTraversal,
        entry: Entry,
        action: PathAction,
        handler: StorageHandler,
        entries: AsyncIterator[Entry] | None,
        *,
        patterns: Sequence[_Search] | None = None,
        _held: dict[tuple[str, ...], dict[tuple[str, ...], Entry]] | None = None,
    ) -> None:
        super().__init__(engine, entry, action, handler, patterns=patterns)
        # The listing, for the subtree's root; None for a directory below it,
        # which shares the root's entries held.
        self.entries = entries
        # Each directory's entries held, both by key.
        self._held = {} if _held is None else _held

    async def run(self) -> Any:
        entry = self.entry
        if self.entries is not None:
            try:
                if entry.get("type") == "directory":
                    root = self._key(entry["name"])
                    async for found in self.entries:
                        self._hold(root, found)
            finally:
                if (aclose := getattr(self.entries, "aclose", None)) is not None:
                    await aclose()
        match = self._matched(entry)
        value = await self.action.map(entry, match)
        held = self._held.pop(self._key(entry["name"]), {})
        if entry.get("type") != "directory" or not self._descend(entry):
            # A file, or a directory kept as matched: reduced with no values.
            entries = []
        else:
            entries = list(held.values())
            entries = self.prune(entries)
        tasks: list[PathTask] = [
            TraverseSubtree(
                self._engine,
                e,
                self.action,
                self.handler,
                None,
                patterns=self.patterns,
                _held=self._held,
            )
            for e in entries
        ]
        values = await self._run_tasks(tasks)
        return await self.action.reduce(entry, match, value, values)

    def _hold(self, root: tuple[str, ...], entry: Entry) -> None:
        """Hold an entry in its directory, making the directories between it and the root."""
        key = self._key(entry["name"])
        depth = len(root)
        if len(key) <= depth or key[:depth] != root:
            return  # The root's own marker, or none of the subtree.
        held = self._held.setdefault(key[:-1], {})
        if entry["type"] == "directory" and key in held:
            held[key] = {**held[key], **entry}  # Listed after made: listed wins.
        else:
            held[key] = entry
        parent = key[:-1]
        while len(parent) > depth:
            siblings = self._held.setdefault(parent[:-1], {})
            if parent in siblings:
                break
            name = self.handler.join(self.entry["name"], *parent[depth:])
            siblings[parent] = {"name": name, "size": 0, "type": "directory"}
            parent = parent[:-1]
