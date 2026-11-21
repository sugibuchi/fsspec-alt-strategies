# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Two tasks traversing a tree for a `PathAction`, interchangeably, as glob patterns prune it.

`VisitDirectory` lists one directory by ``ils``, and runs a task per
subdirectory. `TraverseSubtree` takes a whole subtree's listing, as
``ils(..., recursive=True)`` gives it, and traverses it in itself. Both call
the action alike: `map` on a path before what is below it, `reduce` on a
directory after. So either may traverse any subtree, and a `VisitDirectory`
hands a subtree the patterns take whole to a `TraverseSubtree`, if asked to.

Patterns, `patterns._Search`s, prune the traversal and choose the paths acted
on. A path no pattern matches gets no `map` nor `reduce`: a directory's
passes the values of the items below it on to its parent. Without patterns,
every path is acted on.

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

    from .handler import Entry
    from .patterns import _Search
    from .tree import TreeTraversal

__all__: list[str] = []


class PathAction:
    """What an operation does on each path a traversal visits: the strategy is the task's.

    A task calls `map` on a path before traversing what is below it, and
    `reduce` on a directory after. An item's value is what `reduce` returned
    for a directory, or `map` for a file. Each defaults to None, so a subclass
    writes only those it needs.

    The root is a path like any other: `map` is called on it, and `reduce` too
    if it is a directory. A root that is a file gets `map` only, and its value
    is what `map` returned.

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

    async def map(self, info: Entry) -> Any:
        """Act on a path, before its subtree is traversed: a value for `reduce`, or None.

        ``info`` is the path's entry, as a listing or ``info`` gave it.
        """
        return None

    async def reduce(self, info: Entry, value: Any, values: list[Any]) -> Any:
        """Act on a directory, after its subtree is traversed: its value, or None.

        ``value`` is what `map` returned for it, and ``values`` the values of
        the items in it, in no order.
        """
        return None


class _Through:
    """The value of a directory no pattern matched: the values below it, for its parent."""

    def __init__(self, values: list[Any]) -> None:
        self.values = values


class _TraversalTask(PathTask):
    """What both tasks share: the action, the patterns, and a path's map and reduce."""

    def __init__(
        self,
        engine: TreeTraversal,
        entry: Entry,
        action: PathAction,
        *,
        patterns: Sequence[_Search] | None = None,
    ) -> None:
        # The path is the entry's, as a listing or an info gave it.
        super().__init__(engine, entry["name"].rstrip("/"))
        self.entry = entry
        self.action = action
        self.patterns = patterns

    def _matched(self, entry: Entry) -> bool:
        return self.patterns is None or any(p.matched(entry) for p in self.patterns)

    def _descend(self, entry: Entry) -> bool:
        return self.patterns is None or any(p.descend(entry) for p in self.patterns)

    def _whole(self, entry: Entry) -> bool:
        """Whether all below a directory is traversed: no pattern prunes it."""
        return self.patterns is None or any(p.whole(entry) for p in self.patterns)

    async def _map(self, entry: Entry) -> Any:
        """A path's value before what is below it: what `map` returns, if matched."""
        return await self.action.map(entry) if self._matched(entry) else None

    async def _leaf(self, entry: Entry) -> Any:
        """The value of a path not traversed below: a file, or a directory pruned."""
        return await self._map(entry) if self._matched(entry) else _Through([])

    async def _reduce(
        self, entry: Entry, value: Any, results: list[Any], errors: list[BaseException]
    ) -> Any:
        """A directory's value once all below it is done: its `reduce`, unless anything failed."""
        if errors:
            raise BaseExceptionGroup(f"Traversing {entry['name']!r} failed", errors)
        values: list[Any] = []
        for result in results:
            if isinstance(result, _Through):
                values.extend(result.values)
            else:
                values.append(result)
        if self._matched(entry):
            return await self.action.reduce(entry, value, values)
        else:
            return _Through(values)


class VisitDirectory(_TraversalTask):
    """A directory listed by one ``ils``, with a task per subdirectory to traverse.

    The files in it are mapped here, as the listing gives them. Each
    subdirectory is a `VisitDirectory` of its own, or, with ``recursive``,
    a `TraverseSubtree` where the patterns take all below it.
    """

    def __init__(
        self,
        engine: TreeTraversal,
        entry: Entry,
        action: PathAction,
        ils: Callable[..., AsyncIterator[Entry]],
        *,
        patterns: Sequence[_Search] | None = None,
        recursive: bool = False,
    ) -> None:
        super().__init__(engine, entry, action, patterns=patterns)
        self.ils = ils
        self.recursive = recursive

    async def run(self) -> Any:
        entry = self.entry
        if entry.get("type") != "directory":
            return await self._leaf(entry)
        value = await self._map(entry)

        results: list[Any] = []
        errors: list[BaseException] = []
        children: list[PathTask] = []
        subdirs: set[str] = set()
        listing = self.ils(entry)
        try:
            async for listed in listing:
                name = listed["name"].rstrip("/")
                if name == self.path:
                    continue  # A directory's own marker is nothing in it.
                found = {**listed, "name": name}
                if found["type"] == "directory":
                    if name in subdirs:
                        continue  # Listed twice, e.g. as a prefix and a marker.
                    subdirs.add(name)
                    if self._descend(found):
                        children.append(self._child(found))
                        continue
                try:
                    results.append(await self._leaf(found))
                except Exception as e:
                    errors.append(e)
        finally:
            if (aclose := getattr(listing, "aclose", None)) is not None:
                await aclose()

        try:
            results += await self._run_tasks(children)
        except BaseExceptionGroup as group:
            errors += group.exceptions
        return await self._reduce(entry, value, results, errors)

    def _child(self, entry: Entry) -> PathTask:
        """The task traversing a subdirectory: all below it by one listing, where asked and whole."""
        if self.recursive and self._whole(entry):
            return TraverseSubtree(
                self._engine,
                entry,
                self.action,
                self.ils(entry, recursive=True),
                patterns=self.patterns,
            )
        else:
            return VisitDirectory(
                self._engine,
                entry,
                self.action,
                self.ils,
                patterns=self.patterns,
                recursive=self.recursive,
            )


class TraverseSubtree(_TraversalTask):
    """A subtree traversed from one listing of all below it, in any order.

    The listing is read whole before any path is acted on, as a path's
    descendants may come before it: the subtree is held in memory. A
    directory not listed, as an object store lists only files, is made from
    the paths below it.
    """

    def __init__(
        self,
        engine: TreeTraversal,
        entry: Entry,
        action: PathAction,
        entries: AsyncIterator[Entry],
        *,
        patterns: Sequence[_Search] | None = None,
    ) -> None:
        super().__init__(engine, entry, action, patterns=patterns)
        self.entries = entries
        # Each directory's items, by name: a directory listed replaces one made.
        self._below: dict[str, dict[str, Entry]] = {}

    async def run(self) -> Any:
        entry = self.entry
        try:
            if entry.get("type") != "directory":
                return await self._leaf(entry)
            async for found in self.entries:
                self._hold({**found, "name": found["name"].rstrip("/")})
        finally:
            if (aclose := getattr(self.entries, "aclose", None)) is not None:
                await aclose()
        return await self._traverse(entry)

    def _hold(self, entry: Entry) -> None:
        """Hold an entry in its directory, making the directories between it and the root."""
        name = entry["name"]
        prefix = self.path + "/" if self.path else ""
        if name == self.path or not name.startswith(prefix):
            return  # The root's own marker, or none of the subtree.
        parent, _, _ = name.rpartition("/")
        items = self._below.setdefault(parent, {})
        if entry["type"] != "directory" or name not in items:
            items[name] = entry
        else:
            items[name] = {**items[name], **entry}  # Listed after made: listed wins.
        while parent != self.path:
            above, _, _ = parent.rpartition("/")
            siblings = self._below.setdefault(above, {})
            if parent in siblings:
                break
            siblings[parent] = {"name": parent, "size": 0, "type": "directory"}
            parent = above

    async def _traverse(self, entry: Entry) -> Any:
        """A directory's value: map, all below it, then reduce, as `VisitDirectory` does."""
        value = await self._map(entry)
        results: list[Any] = []
        errors: list[BaseException] = []
        for item in self._below.pop(entry["name"], {}).values():
            try:
                if item["type"] == "directory" and self._descend(item):
                    results.append(await self._traverse(item))
                else:
                    results.append(await self._leaf(item))
            except BaseExceptionGroup as group:
                errors += group.exceptions
            except Exception as e:
                errors.append(e)
        return await self._reduce(entry, value, results, errors)
