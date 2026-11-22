# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The walk under the tree strategies: one listing per directory, as a standard walk gives them.

`_TreeWalk` lists the directories below one or more roots, several at once,
and yields each directory's listing whole: its entries, or its error. It asks
a `_Visit` which directories the operation needs. A directory is listed alone
by ``ils``, or a subtree the visit takes whole by one recursive ``ils``, which
`_SubtreeReader` splits into each directory's listing. How the handler lists
it recursively is its own: in one request, or a directory per request.

The caller chooses the order, as with a standard walk:

- "topdown": a directory before those below it. Each is listed alone: a
  subdirectory is found only in its parent's whole listing.
- "bottomup": a directory after those below it. A recursive listing gives that
  order; a directory listed alone waits for those below it.
- "any": each as it completes.

The directories to list wait on a stack, so the walk goes depth first, as
fsspec's walk does.

This module depends on `_visit` and `listing`, and `handler` and `paths` of `common`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from ..common.paths import _trim
from ._visit import _Visit
from .listing import _DepthFirst

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Collection

    from fsspec_alt_strategies.profile import StorageProfile

    from ..common.handler import Entry
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _Listing:
    """A directory's listing: its entries, or its error.

    ``depth`` is 1 for a root the walk started from, 2 for its subdirectories,
    and so on. ``entries`` are as ``ils`` gives them, without the directory
    itself, a subdirectory listed twice kept once. A failed listing has its
    ``error``, noted with the directory, and no entries: a root that is a file
    fails with NotADirectoryError, its ``entry`` attribute the file's entry.

    ``depth`` 0 is no listing, but the walk's root itself, as `_find_listings`
    yields it: ``entries`` hold only the root's ``info`` entry.
    """

    directory: str
    depth: int
    entries: list[dict[str, Any]]
    error: Exception | None = None


_Order = Literal["any", "topdown", "bottomup"]
"""The order of a walk's listings: each as it completes, or a directory before or after those below it."""

_Pending = tuple[str, int, "Entry | None", bool]
"""A directory to list: its path, its depth, its entry if known, and whether all below it is listed with it."""

_Read = Callable[[], Awaitable[list[_Listing]]]
"""A read to start: a directory's listing alone, or a recursive listing's next."""


class _TreeWalk:
    """One walk of a tree: the directories to list, the listings in flight, and what waits to be yielded.

    ``path`` may be several directories, e.g. a glob's matches, walked at once,
    each as a root. None may be inside another. A root's failure is a failed
    listing like any other: the caller decides what it means.

    ``visit`` says which directories the walk lists: by default all below
    each root. A subtree the visit takes whole is listed by one recursive
    ``ils`` of the handler, where ``order`` allows it.

    Each listing holds a permit of the handler's ``operation_limit`` until it
    ends: the walk takes one before it starts a listing, while one is free, or,
    when none of its own runs, by waiting for one. Once ``stopping`` returns
    true, nothing new starts: the listings in flight are yielded.
    """

    def __init__(
        self,
        handler: TreeFileOperationHandler,
        path: str | Entry | Collection[str | Entry],
        visit: _Visit | None = None,
        *,
        order: _Order = "any",
        ils_kwargs: dict[str, Any] | None = None,
        stopping: Callable[[], object] | None = None,
    ) -> None:
        self.handler = handler
        self.visit = visit or _Visit()
        self.order = order
        self.ils_kwargs = ils_kwargs or {}
        self.stopping = stopping
        self.roots = _roots(path)
        self.limit = handler.operation_limit
        # The directories to list, a stack: the deepest found are listed first.
        self.pending: list[_Pending] = []
        # The reads in flight: a directory's listing, or a recursive listing's next.
        self.reading: dict[asyncio.Task[list[_Listing]], _SubtreeReader | None] = {}
        # What holds a permit: a read of one directory, or a recursive listing.
        self.holders: set[object] = set()
        self.bottom_up = _BottomUp() if order == "bottomup" else None

    async def listings(self) -> AsyncGenerator[_Listing]:
        """Walk, yielding each directory's listing in the walk's order."""
        await self.handler.open()
        for name, entry in sorted(self.roots.items(), reverse=True):
            depth = self.visit.root_depth(name)
            listed = entry or {"name": name, "type": "directory"}
            self.pending.append(
                self._pending(name, depth, entry, self.visit.enter_root(listed, depth))
            )
        try:
            while await self._start():
                await asyncio.wait(self.reading, return_when=asyncio.FIRST_COMPLETED)
                # One read at a time, in the order they started, those done while
                # the caller handles the last included.
                while (task := self._done()) is not None:
                    reader = self.reading.pop(task)
                    ready = self._take(reader, task.result())
                    if reader is not None:
                        await self._continue_or_close(reader)
                    # The next listings start before yielding, so that they run
                    # while the caller handles these.
                    await self._start()
                    for listing in ready:
                        yield listing
        finally:
            await self._close()

    def _stopped(self) -> bool:
        return self.stopping is not None and bool(self.stopping())

    def _done(self) -> asyncio.Task[list[_Listing]] | None:
        """The first read done, in the order they started, if any."""
        return next((t for t in self.reading if t.done()), None)

    async def _start(self) -> bool:
        """Start listings of the deepest directories pending while the walk gets permits: tell whether any runs.

        A permit free is taken at once. While none is, the walk waits for one
        only when none of its own listings runs, as nothing else would wake it:
        else one of them gives one back as it ends.
        """
        if self._stopped():
            self.pending.clear()
        while self.pending and not (self.reading and self.limit.locked()):
            await self.limit.acquire()
            name, depth, entry, whole = self.pending.pop()
            reader = await self._reader(name, depth, entry) if whole else None
            if reader is None:
                holder = object()
                self.holders.add(holder)
                self._launch(partial(self._list, name, depth, entry), holder, None)
            else:
                self.holders.add(reader)
                self._launch(reader.read, reader, reader)
        return bool(self.reading)

    def _launch(
        self, read: _Read, holder: object, reader: _SubtreeReader | None
    ) -> None:
        task = asyncio.ensure_future(self._run(read, holder, reader))
        self.reading[task] = reader

    async def _run(
        self, read: _Read, holder: object, reader: _SubtreeReader | None
    ) -> list[_Listing]:
        """A read, which gives back its permit as soon as its listing ends.

        The read starts here, so that one cancelled before it runs leaves no
        coroutine never awaited.
        """
        try:
            return await read()
        finally:
            if reader is None or reader.ended:
                self._release(holder)

    def _release(self, holder: object) -> None:
        if holder in self.holders:
            self.holders.discard(holder)
            self.limit.release()

    async def _continue_or_close(self, reader: _SubtreeReader) -> None:
        """After a recursive listing's read: start its next, under the permit it holds, or close it once ended or stopped."""
        if reader.ended or self._stopped():
            self._release(reader)
            await reader.close()
        else:
            self._launch(reader.read, reader, reader)

    async def _reader(
        self, name: str, depth: int, entry: Entry | None
    ) -> _SubtreeReader | None:
        """The reader of a subtree listed whole, or None to list it alone.

        Where the handler's recursive ``ils`` is its family's default, a
        directory per request, the walk lists the subtree so itself, depth
        first and concurrently, as it lists any other directory.
        """
        try:
            listed = entry or self.handler.entry(name)
            entries = self.handler.ils(listed, recursive=True, **self.ils_kwargs)
        except Exception as e:
            entries = _raising(e)  # Its read fails as the subtree's.
        if isinstance(entries, _DepthFirst):
            await entries.aclose()
            return None
        return _SubtreeReader(self.handler, name, depth, entries)

    async def _list(self, name: str, depth: int, entry: Entry | None) -> list[_Listing]:
        """A directory's listing alone, by ``ils``, under the permit the walk holds for it.

        It drops the directory's own entry, as some storages list it, e.g. as
        its marker object, and a subdirectory listed again, e.g. as a prefix
        and as its marker. A root that is a file lists itself: it fails with
        NotADirectoryError.
        """
        entries: list[dict[str, Any]] = []
        subdirs: set[str] = set()
        listing: AsyncIterator[Entry] | None = None
        try:
            listing = self.handler.ils(
                entry or self.handler.entry(name), **self.ils_kwargs
            )
            async for found in listing:
                path = found["name"].rstrip("/")
                if path == name:
                    if found["type"] != "directory" and name in self.roots:
                        raise _not_a_directory(name, found)
                    continue  # A directory's own marker is nothing in it.
                if found["type"] == "directory":
                    if path in subdirs:
                        continue
                    subdirs.add(path)
                entries.append(found)
        except Exception as e:
            return [_Listing(name, depth, [], e)]
        finally:
            if listing is not None:
                await _close(listing)
        return [_Listing(name, depth, entries)]

    def _pending(
        self,
        name: str,
        depth: int,
        entry: Entry | None,
        need: Literal["directory", "subtree"],
    ) -> _Pending:
        """A directory to list: with all below it where the visit takes it whole, and the order allows."""
        whole = need == "subtree" and self.order != "topdown"
        return name, depth, entry, whole

    def _take(
        self, reader: _SubtreeReader | None, listings: list[_Listing]
    ) -> list[_Listing]:
        """The listings of one read to yield, in the walk's order: a directory listed alone's subdirectories pushed."""
        ready: list[_Listing] = []
        for listing in listings:
            if listing.error is not None:
                _note(listing.error, listing.directory)
                subdirs: list[str] = []
            elif reader is None:
                subdirs = self._push(listing)
            else:
                subdirs = [
                    e["name"].rstrip("/")
                    for e in listing.entries
                    if e["type"] == "directory"
                ]
            if self.bottom_up is None:
                ready.append(listing)
            else:
                ready += self.bottom_up.add(listing, subdirs)
        return ready

    def _push(self, listing: _Listing) -> list[str]:
        """Push the subdirectories of a directory listed alone that the visit needs, the first found on top: their paths."""
        pushed: list[_Pending] = []
        for entry in listing.entries:
            if entry["type"] == "directory":
                depth = listing.depth + 1
                need = self.visit.enter(entry, depth)
                if need != "skip":
                    name = entry["name"].rstrip("/")
                    listed = {**entry, "name": name}
                    pushed.append(self._pending(name, depth, listed, need))
        self.pending += reversed(pushed)
        return [name for name, _, _, _ in pushed]

    async def _close(self) -> None:
        """Cancel the reads in flight, give back their permits, and close every recursive listing."""
        readers = [r for r in self.reading.values() if r is not None]
        for task in self.reading:
            task.cancel()
        await asyncio.gather(*self.reading, return_exceptions=True)
        for holder in list(self.holders):
            self._release(holder)
        self.reading.clear()
        for reader in readers:
            await reader.close()


def _roots(path: str | Entry | Collection[str | Entry]) -> dict[str, Entry | None]:
    """The roots of a walk, by path, each with its entry if given as one, e.g. a glob's match."""
    roots: dict[str, Entry | None] = {}
    for root in [path] if isinstance(path, (str, dict)) else path:
        if isinstance(root, dict):
            name = _trim(root["name"])
            roots[name] = {**root, "name": name}
        else:
            roots[_trim(root)] = None
    return roots


class _BottomUp:
    """Holds each listing back until those of the subdirectories the walk lists are out.

    A subdirectory's listing may come before its parent's, as a recursive
    listing passes a directory only after all below it: such a subdirectory,
    out already, is not waited for. A failed listing is out at once: what is
    below it may never come.
    """

    def __init__(self) -> None:
        # Each listing held back, with how many of its subdirectories are not out.
        self.held: dict[str, tuple[_Listing, int]] = {}
        self.parents: dict[str, str] = {}
        # The directories out before their parent's listing came.
        self.out: set[str] = set()

    def add(self, listing: _Listing, subdirs: list[str]) -> list[_Listing]:
        """The listings ready once this one is in, each after those below it."""
        waiting = [s for s in subdirs if s not in self.out]
        self.out.difference_update(subdirs)
        if waiting:
            self.held[listing.directory] = listing, len(waiting)
            self.parents.update(dict.fromkeys(waiting, listing.directory))
            return []

        ready = [listing]
        directory = listing.directory
        # A subtree done may complete its parent's, and so on up.
        while (parent := self.parents.pop(directory, None)) is not None:
            held, remaining = self.held[parent]
            if remaining > 1:
                self.held[parent] = held, remaining - 1
                return ready
            del self.held[parent]
            ready.append(held)
            directory = parent

        self.out.add(directory)  # Its parent's listing has not come yet.
        return ready


class _SubtreeReader:
    """One subtree listed whole by one recursive listing: each read the directories it completed.

    `_Split` turns its entries into each directory's listing, as the listing
    passes each directory: depth first, or in name order. A listing in any
    other order, e.g. parents first, as the profile's ``recursive_order`` may
    say, tells no directory done before it ends: it is held whole, then taken
    in name order, which is depth first. A listing that fails, whenever it
    breaks, is the root's failure: what it completed before stands, and the
    directories it was in are not listed. A storage resumes its own listing
    where it can, e.g. a page by its continuation token.
    """

    def __init__(
        self,
        handler: TreeFileOperationHandler,
        root: str,
        depth: int,
        entries: AsyncIterator[Entry],
    ) -> None:
        self.handler = handler
        self.root = root
        self.depth = depth
        self.entries = entries
        self.split: _Split | None = None
        self.ended = False

    async def read(self) -> list[_Listing]:
        """The listings of the directories the recursive listing completed next."""
        done: list[_Listing] = []
        try:
            if self.split is None:
                self.split = _Split(self.root, self.depth)
                if not _depth_first(self.entries, self.handler.profile):
                    held = [e async for e in self.entries]
                    self.ended = True
                    named = sorted(held, key=lambda e: e["name"])
                    return [
                        x for e in named for x in self.split.add(e)
                    ] + self.split.end()
            split = self.split
            while not done:
                try:
                    entry = await anext(self.entries)
                except StopAsyncIteration:
                    self.ended = True
                    return split.end()
                done += split.add(entry)
        except Exception as e:
            self.ended = True
            done.append(_Listing(self.root, self.depth, [], e))
        return done

    async def close(self) -> None:
        await _close(self.entries)


async def _raising(error: Exception) -> AsyncIterator[Entry]:
    """A listing that fails as it starts, as one whose making raised."""
    raise error
    yield  # An async generator, which never yields.


def _depth_first(entries: AsyncIterator[Entry], profile: StorageProfile) -> bool:
    """Whether a recursive listing is depth first, or in name order: `_tree_ils`'s, or as the profile declares it.

    A storage's own listing keeps the order its profile declares. Unknown, it
    is taken as parents first, which is right on any storage.
    """
    return isinstance(entries, _DepthFirst) or profile.recursive_order == "depth_first"


async def _close(entries: AsyncIterator[Entry]) -> None:
    """Close a listing, where it can be: a recursive ``ils`` gives any async iterator."""
    if (aclose := getattr(entries, "aclose", None)) is not None:
        await aclose()


# Splitting a recursive listing into each directory's listing.


class _OpenDirectory:
    """A directory a recursive listing is in: its entries so far, and its subdirectories' names."""

    def __init__(self, name: str, depth: int) -> None:
        self.name = name
        self.depth = depth
        self.entries: list[dict[str, Any]] = []
        # Its subdirectories listed, below which nothing has come yet, in listing order.
        self.pending: dict[str, None] = {}
        # Its subdirectories listed: one listed twice comes once.
        self.subdirs: set[str] = set()

    def hold(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        if entry["type"] == "directory":
            self.subdirs.add(entry["name"].rstrip("/"))

    def listing(self) -> _Listing:
        return _Listing(self.name, self.depth, self.entries)


class _Split:
    """A recursive listing's entries, depth first or in name order, split into each directory's listing.

    All below a directory come together, so it keeps only the directories the
    listing is in, from the root down: those ``open``. A name not below the
    deepest one closes it, for none below it comes later, and its listing is
    complete. A directory's entry may come apart from what is below it, as
    "a/b-x" comes between "a/b" and "a/b/c" in name order: it waits as
    ``pending`` in its parent until a path below it opens it, or the listing
    passes it, empty. Depth first, its siblings may come in any order: what is
    below it comes right after it, so any other name tells it is empty.

    A directory not listed, as an object store lists only files, gets an entry
    in its parent from the paths below it. A directory's listing comes after
    those below it.
    """

    def __init__(self, root: str, root_depth: int) -> None:
        self.root = root
        self.open = [_OpenDirectory(root, root_depth)]

    def add(self, entry: dict[str, Any]) -> list[_Listing]:
        """Hold an entry in its directory, and the listings it completes."""
        name = entry["name"].rstrip("/")
        if name == self.root:
            # A file lists itself. A directory may list its own marker.
            if entry["type"] != "directory":
                raise _not_a_directory(self.root, entry)
            return []
        if not _below(name, self.root):
            return []  # No storage should list it.

        done = self.close(name)
        top = self.open[-1]
        for directory in _between(top.name, name):
            if directory not in top.pending:
                top.hold({"name": directory, "size": 0, "type": "directory"})
            top.pending.pop(directory, None)
            top = _OpenDirectory(directory, top.depth + 1)
            self.open.append(top)

        if entry["type"] == "directory":
            if name in top.subdirs:
                return done  # Listed twice, e.g. as its marker too.
            top.pending[name] = None
        top.hold(entry)
        return done

    def close(self, name: str) -> list[_Listing]:
        """The listings of the directories that ``name`` comes after."""
        done = []
        while len(self.open) > 1 and not _below(name, self.open[-1].name):
            done += self.pop()

        top = self.open[-1]
        for directory in list(top.pending):
            # Name order: those after it are not passed either.
            if _below(name, directory) or name < _prefix(directory):
                break
            done.append(self.empty(directory))
        return done

    def end(self) -> list[_Listing]:
        """The listings of all still open, once the listing has ended, deepest first."""
        done = []
        while self.open:
            done += self.pop()
        return done

    def pop(self) -> list[_Listing]:
        """The listings of the deepest directory open, after those of the empty ones pending in it."""
        top = self.open[-1]
        done = [self.empty(d) for d in list(top.pending)]
        self.open.pop()
        done.append(top.listing())
        return done

    def empty(self, directory: str) -> _Listing:
        """The listing of a directory pending in the deepest one open: nothing is below it."""
        top = self.open[-1]
        del top.pending[directory]
        return _Listing(directory, top.depth + 1, [])


def _prefix(directory: str) -> str:
    """What the paths below a directory start with."""
    return directory if not directory or directory.endswith("/") else f"{directory}/"


def _below(name: str, directory: str) -> bool:
    return name.startswith(_prefix(directory))


def _between(directory: str, name: str) -> list[str]:
    """The directories below ``directory`` that ``name`` is in, parents first."""
    prefix = _prefix(directory)
    parts = name[len(prefix) :].split("/")[:-1]
    return [prefix + "/".join(parts[: i + 1]) for i in range(len(parts))]


def _not_a_directory(root: str, entry: dict[str, Any]) -> NotADirectoryError:
    """The error of a root that is a file, with its entry as the listing gave it.

    A copy or move of the root then hands the entry on, with no ``info``.
    """
    error = NotADirectoryError(root)
    error.entry = {**entry, "name": root}  # type: ignore[attr-defined]
    return error


def _note(error: Exception, directory: str) -> None:
    """Note its directory on a listing's error.

    In its note for people, and as an attribute for code, e.g. fsspec's walk.
    """
    error.add_note(f"Listing {directory!r} failed.")
    error.directory = directory  # type: ignore[attr-defined]
