# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The walk under the tree strategies: a tree's directories, read as the walk asks for them.

`_TreeWalk` walks one tree, or several roots at once. It asks a `_Visit`
which directories the operation needs, and the handler how to list them, by
its ``lists_subtree``. It reads a directory listed alone by a
`_DirectoryReader`, and a subtree listed whole by a `_SubtreeReader`, a part
per read. It awaits a few reads at once, takes whichever comes first, and
asks a reader again only when it wants more. So nothing is listed ahead of
what the walk asks, and nothing is buffered between the readers and the walk.

The directories to list wait on a stack, so the walk goes depth first, as
fsspec's walk does. What waits is the subdirectories along the paths the walk
is in, not a whole level of the tree. While too many wait, a reader that would
add more is paused until they are taken.

A walk yields events: each directory's listing in parts, and what it found but
did not list. The strategies act on each part as it comes, so a directory of
hundreds of thousands of entries is never held whole.

This module depends on `_visit` and `listing`, and `handler` and `paths` of `common`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from ..common.paths import _trim
from ._visit import _Visit
from .listing import _DepthFirst

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Collection,
        Sequence,
    )

    from fsspec_alt_strategies.profile import StorageProfile

    from ..common.handler import Entry
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


_CHUNK_SIZE = 1000
"""The most entries of one part of a listing: a directory's are never all held."""

_WAITING = 32_000
"""The directories found and not listed, past which a reader that would add more is paused."""


# The events a walk yields.


@dataclass(frozen=True, slots=True)
class _Part:
    """A part of a directory's listing, as a walk yields it, or all of it.

    ``depth`` is 1 for the directory the walk started from, 2 for its
    subdirectories, and so on. ``entries`` are as ``ils`` gives them, without
    the directory itself.

    A directory's listing comes in parts, each of at most 1000 entries:
    ``first`` on the first part, ``last`` on the last. So an empty directory is
    one part, both first and last, with no entries. A failed listing ends with
    its ``error``, as its last part. `_tree_walk` holds each directory whole:
    one part, both first and last.

    ``depth`` 0 is no listing, but the walk's root itself, as `_find_listings`
    yields it: ``entries`` hold only the root's ``info`` entry, under
    ``directory``.
    """

    directory: str
    depth: int
    entries: list[dict[str, Any]]
    error: Exception | None = None
    first: bool = True
    last: bool = True


@dataclass(frozen=True, slots=True)
class _Skipped:
    """A subdirectory a listing found that the walk leaves, as its visit says: e.g. one deeper than ``maxdepth``.

    It comes before the part that found it. ``entry`` is as that part gave it,
    its name without a trailing "/".
    """

    entry: Entry
    depth: int


@dataclass(frozen=True, slots=True)
class _FileRoot:
    """A root of the walk that is a file: nothing is listed of it.

    ``error`` is the NotADirectoryError its listing raised. Its ``entry``
    attribute, where set, is the file's entry as the listing gave it, so that
    a caller asks nothing again.
    """

    root: str
    error: NotADirectoryError


@dataclass(frozen=True, slots=True)
class _SubtreeListed:
    """All below a directory is listed: its listing, and those of the subdirectories the walk lists, have ended.

    It comes after the directory's last part, deepest first where several end
    at once. A listing that failed has ended too. A walk stopped never tells
    it of a directory with a listing below it left unended. Only a walk asked
    for them yields these.
    """

    directory: str


_Event = _Part | _Skipped | _FileRoot | _SubtreeListed
"""What a walk yields: the parts of each directory's listing, what it found but did not list, and when a subtree is done."""


# What the walk reads.

_How = Literal["alone", "whole", "filtered"]
"""How a directory is read.

- "alone": its listing alone, by ``ils``;
- "whole": all below it, by the handler's ``ils_recursive``;
- "filtered": all below it so, though the visit prunes: the walk keeps what
  the visit would list.

A handler whose ``ils_recursive`` is the library's default, a directory per
request, has its subtrees read alone, a directory at a time: the walk lists
so itself, concurrently, under its own control.
"""


@dataclass(slots=True)
class _Pending:
    """A directory found and not listed yet, and how it is to be read."""

    name: str
    depth: int
    entry: Entry | None
    """Its entry as its parent's listing gave it, else None: a root given as a path."""
    how: _How
    root: bool = False
    after_break: bool = False
    """Read again after a recursive listing broke off: no subtree below is listed whole again."""
    skip: set[str] | None = None
    """The names a broken listing passed already, left out as it is read again."""
    first: bool = True
    """Whether its first part is still to come."""


@dataclass(slots=True)
class _Read:
    """What one read gives: parts, and after a recursive listing broke off, what to read again."""

    parts: list[_Part]
    again: list[_Pending] = field(default_factory=list)
    """The directories the broken listing was in, each to be listed alone, leaving out what it passed."""
    found: list[tuple[Entry, int]] = field(default_factory=list)
    """Their subdirectories below which nothing came, with their depths: the walk plans them anew."""


class _DirectoryReader:
    """One directory listed alone, by ``ils``: each read a part of it.

    It drops the directory's own entry, as some storages list it, e.g. as its
    marker object, and a subdirectory listed again, e.g. as a prefix and as
    its marker. A root that is a file lists itself: it raises
    NotADirectoryError, as the part's error.
    """

    def __init__(
        self,
        handler: TreeFileOperationHandler,
        pending: _Pending,
        ils_kwargs: dict[str, Any],
        chunk_size: int,
    ) -> None:
        self.handler = handler
        self.pending = pending
        self.ils_kwargs = ils_kwargs
        self.chunk_size = chunk_size
        self.entries: AsyncIterator[Entry] | None = None
        # The subdirectories shown so far: each comes once.
        self.subdirs: set[str] = set()
        self.first = pending.first
        self.ended = False

    async def read(self) -> _Read:
        """The next part, under the permit the walk holds for the read: its error last, if it fails."""
        pending = self.pending
        chunk: list[dict[str, Any]] = []
        try:
            if self.entries is None:
                entry = pending.entry or self.handler.entry(pending.name)
                self.entries = self.handler.ils(entry, **self.ils_kwargs)
            while len(chunk) < self.chunk_size:
                try:
                    entry = await anext(self.entries)
                except StopAsyncIteration:
                    self.ended = True
                    break
                if (kept := self._kept(entry)) is not None:
                    chunk.append(kept)
        except Exception as e:
            self.ended = True
            if pending.skip is not None and isinstance(e, FileNotFoundError):
                # Read again after a break, and gone since, e.g. emptied by a
                # deletion on an object store: nothing is left to list.
                return _Read([self._part(chunk, last=True)])
            parts = [self._part(chunk, last=False)] if chunk else []
            parts.append(self._part([], last=True, error=e))
            return _Read(parts)
        return _Read([self._part(chunk, last=self.ended)])

    def _kept(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        name = entry["name"].rstrip("/")
        pending = self.pending
        if name == pending.name:
            if entry["type"] != "directory" and pending.root:
                raise _not_a_directory(pending.name, entry)
            return None  # A directory's own marker is nothing in it.
        if pending.skip is not None and name in pending.skip:
            return None
        if entry["type"] == "directory":
            if name in self.subdirs:
                return None
            self.subdirs.add(name)
        return entry

    def _part(
        self,
        entries: list[dict[str, Any]],
        *,
        last: bool,
        error: Exception | None = None,
    ) -> _Part:
        part = _Part(
            self.pending.name, self.pending.depth, entries, error, self.first, last
        )
        self.first = False
        return part

    async def close(self) -> None:
        if self.entries is not None:
            await _close(self.entries)


class _SubtreeReader:
    """One subtree listed whole, by one recursive listing: each read the parts of the directories it passed.

    A splitter for the listing's order turns its entries into each directory's
    parts: see `_DepthFirstSplit` and `_ParentsFirstSplit`. In a filtered
    subtree, the parts of a directory ``keep`` refuses are dropped.

    A listing failing before anything came is the root's error, as its first
    and last part. One broken off later is read again a directory per request:
    the read gives the parts passed, and what to list again. A recursive
    listing is long, e.g. a WebDAV body of minutes, and a connection dropped
    in it would otherwise leave an arbitrary part of the tree unwalked, which
    fsspec's ``find`` then omits with no error.
    """

    def __init__(
        self,
        handler: TreeFileOperationHandler,
        pending: _Pending,
        ils_kwargs: dict[str, Any],
        chunk_size: int,
        keep: Callable[[str, int], bool] | None,
        entries: AsyncIterator[Entry],
    ) -> None:
        self.handler = handler
        self.pending = pending
        self.ils_kwargs = ils_kwargs
        self.chunk_size = chunk_size
        self.keep = keep
        # The handler's recursive listing, which the walk made to tell how to read it.
        self.entries = entries
        self.split: _DepthFirstSplit | _ParentsFirstSplit | None = None
        self.ended = False

    async def read(self) -> _Read:
        """The parts the listing passed next, under the permit the walk holds for the read."""
        pending = self.pending
        parts: list[_Part] = []
        try:
            if self.split is None:
                kind = _split_of(self.entries, self.handler.profile)
                self.split = kind(
                    pending.name, pending.depth, self.chunk_size, pending.entry
                )
            split = self.split
            while not parts:
                try:
                    entry = await anext(self.entries)
                except StopAsyncIteration:
                    self.ended = True
                    parts += split.end()
                    break
                parts += split.add(entry)
        except Exception as e:
            self.ended = True
            # What was listed first goes first, so that only the root's own
            # failure is its first part.
            if self.split is not None:
                parts += self.split.flush()
            if self.split is None or self.split.first:
                parts.append(_Part(pending.name, pending.depth, [], e, True, True))
                return _Read(parts)
            return self._broken(parts)
        return _Read(self._kept(parts))

    def _kept(self, parts: list[_Part]) -> list[_Part]:
        keep = self.keep
        return (
            parts if keep is None else [p for p in parts if keep(p.directory, p.depth)]
        )

    def _broken(self, parts: list[_Part]) -> _Read:
        """The parts passed, and the directories the listing was in, to list again leaving out what it passed.

        The directories done have had their last part. Those the listing was
        in are listed again by ``ils``, each on its own: the open ones, which
        keep their names. Their subdirectories below which nothing came are
        found anew. So each break resumes deeper in the tree, and ends.
        """
        assert self.split is not None
        keep = self.keep or (lambda directory, depth: True)
        again, found = [], []
        for directory in self.split.open:
            if not keep(directory.name, directory.depth):
                continue
            again.append(
                _Pending(
                    directory.name,
                    directory.depth,
                    directory.entry,
                    "alone",
                    root=directory.name == self.pending.name and self.pending.root,
                    after_break=True,
                    skip=directory.seen,
                    first=directory.first,
                )
            )
            found += [(e, directory.depth + 1) for e in directory.pending.values()]
        return _Read(self._kept(parts), again, found)

    async def close(self) -> None:
        await _close(self.entries)


async def _raising(error: Exception) -> AsyncIterator[Entry]:
    """A listing that fails as it starts, as one whose making raised."""
    raise error
    yield  # An async generator, which never yields.


def _split_of(
    entries: AsyncIterator[Entry], profile: StorageProfile
) -> type[_DepthFirstSplit | _ParentsFirstSplit]:
    """How a recursive listing is split: depth first where it is `_tree_ils`'s, or the profile declares it so.

    `_tree_ils` lists depth first, whatever the profile says, so a handler
    keeping it as its ``ils_recursive`` declares nothing. A storage's own
    listing keeps the order its profile declares. Unknown, it is taken as
    parents first, which is right on any storage.
    """
    if isinstance(entries, _DepthFirst) or profile.recursive_order == "depth_first":
        return _DepthFirstSplit
    else:
        return _ParentsFirstSplit


async def _close(entries: AsyncIterator[Entry]) -> None:
    """Close a listing, where it can be: an ``ils_recursive`` gives any async iterator."""
    if (aclose := getattr(entries, "aclose", None)) is not None:
        await aclose()


# The walk.


class _TreeWalk:
    """One walk of a tree: what to list, the reads in flight, and what was decided of each directory.

    ``path`` may be several directories, e.g. a glob's matches, walked at once,
    each as a root. None may be inside another. A root's listing failing before
    its first part raises, as fsspec's walk does, since nothing has been done
    yet, e.g. FileNotFoundError. A root that is a file is yielded as a
    `_FileRoot`, and the other roots walked. Any other failed listing is
    yielded as its last part, with its error noted, and the subdirectories it
    had not shown are not walked.

    ``visit`` says which directories the walk lists: by default all below
    each root. The handler's ``lists_subtree`` says how. A subdirectory the
    visit leaves is yielded as a `_Skipped`, before the part that found it.
    With ``subtrees``, a `_SubtreeListed` tells when all below a directory is
    listed, as a walk bottom up needs.

    Each reader holds a permit of the handler's ``operation_limit`` while it
    is read, as a listing is one request to the strategies: the walk takes one
    before it starts a reader, and gives it back once the reader ends, or is
    paused. While none is free, one task waits for the next, beside the reads,
    so that the walk takes a permit others give back as soon as they do, e.g.
    deletions sharing the limit. So a reader started is a listing in flight.
    The parts
    of a directory come in order, but those of several directories mix, as
    their reads run at once. Once ``stopping`` returns true, no new directory
    is read, and no reader asked again: the reads in flight are yielded, and a
    directory stopped so has no last part.
    """

    def __init__(
        self,
        handler: TreeFileOperationHandler,
        path: str | Entry | Collection[str | Entry],
        visit: _Visit | None = None,
        *,
        ils_kwargs: dict[str, Any] | None = None,
        stopping: Callable[[], object] | None = None,
        subtrees: bool = False,
        chunk_size: int = _CHUNK_SIZE,
        waiting: int = _WAITING,
    ) -> None:
        self.handler = handler
        self.visit = visit or _Visit()
        self.ils_kwargs = ils_kwargs or {}
        self.stopping = stopping
        self.chunk_size = chunk_size
        self.waiting = waiting
        self.roots = _roots(path)
        self.limit = handler.operation_limit
        # The directories to read, a stack: the deepest found are read first.
        self.pending: list[_Pending] = []
        # The reads in flight, each with its reader, in the order they started.
        self.reading: dict[asyncio.Task[_Read], _DirectoryReader | _SubtreeReader] = {}
        # The readers holding a permit, until they end or are paused.
        self.permits: set[_DirectoryReader | _SubtreeReader] = set()
        # The one task waiting for a permit, while directories wait for one.
        self.waiter: asyncio.Task[Any] | None = None
        # Readers of a directory not done, not asked again while too many wait.
        self.paused: list[_DirectoryReader] = []
        self.subtrees = _Subtrees() if subtrees else None

    async def events(self) -> AsyncGenerator[_Event]:
        """Walk, yielding each event as the reads come."""
        await self.handler.open()
        self.pending = [
            self._root(n, e) for n, e in sorted(self.roots.items(), reverse=True)
        ]
        try:
            while await self._start():
                await asyncio.wait(
                    [*self.reading, *([self.waiter] if self.waiter else [])],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # One read at a time, in the order they started, those done
                # while the caller handles the last included. The waiter's
                # permit is taken as readers start.
                while (task := self._done()) is not None:
                    reader = self.reading.pop(task)
                    events = self._take(reader, task.result())
                    await self._again(reader)
                    # The next reads start before yielding, so that they run
                    # while the caller handles these.
                    await self._start()
                    for event in events:
                        yield event
        finally:
            await self._close()

    def _done(self) -> asyncio.Task[_Read] | None:
        """The first read done, in the order they started, if any."""
        return next((t for t in self.reading if t.done()), None)

    def _stopped(self) -> bool:
        return self.stopping is not None and bool(self.stopping())

    async def _start(self) -> bool:
        """Start reads while the walk gets permits: tell whether it waits for anything, a read or a permit.

        The deepest directories pending go first, then the readers paused once
        nothing is pending. Once stopped, nothing starts: the readers paused
        are closed.
        """
        if self._stopped():
            self.pending.clear()
            for paused in self.paused:
                await paused.close()
            self.paused.clear()
            await self._drop_waiter()
            return bool(self.reading)

        while self.pending or self.paused:
            if not await self._permit():
                break
            reader: _DirectoryReader | _SubtreeReader
            if self.pending:
                reader = await self._reader(self.pending.pop())
            else:
                reader = self.paused.pop()
            self.permits.add(reader)
            self._read(reader)
        if not (self.pending or self.paused):
            await self._drop_waiter()  # Nothing left to start.
        return bool(self.reading) or self.waiter is not None

    async def _permit(self) -> bool:
        """Take a permit for a reader, if the walk can now: else, wait for one beside the reads.

        A permit free is taken at once, and so is the one the waiter got. While
        no read of the walk's runs, the walk waits for one itself, as nothing
        else would wake it.
        """
        if self.waiter is not None:
            if not self.waiter.done():
                return False
            self.waiter = None  # Its permit is the walk's now.
            return True
        elif not self.limit.locked() or not self.reading:
            await self.limit.acquire()
            return True
        else:
            self.waiter = asyncio.ensure_future(self.limit.acquire())
            return False

    async def _drop_waiter(self) -> None:
        """Stop waiting for a permit, giving back the one got, if any."""
        waiter, self.waiter = self.waiter, None
        if waiter is not None:
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
            if not waiter.cancelled() and waiter.exception() is None:
                self.limit.release()

    def _read(self, reader: _DirectoryReader | _SubtreeReader) -> None:
        """Start a reader's next read, under the permit it holds."""
        self.reading[asyncio.ensure_future(self._reading(reader))] = reader

    async def _reading(self, reader: _DirectoryReader | _SubtreeReader) -> _Read:
        """A reader's read, which gives back its permit as soon as the reader ends.

        The walk takes the read only when its caller next asks: the permit of a
        listing ended goes back at once, as the listing's request is done.
        """
        try:
            read = await reader.read()
        except BaseException:
            self._give_back(reader)
            raise
        if reader.ended:
            self._give_back(reader)
        return read

    def _give_back(self, reader: _DirectoryReader | _SubtreeReader) -> None:
        """Give back a reader's permit, unless it was already."""
        if reader in self.permits:
            self.permits.discard(reader)
            self.limit.release()

    async def _again(self, reader: _DirectoryReader | _SubtreeReader) -> None:
        """Ask a reader for more, keeping its permit; pause it while too many directories wait; or close it once done."""
        if reader.ended or self._stopped():
            self._give_back(reader)
            await reader.close()
        elif isinstance(reader, _DirectoryReader) and len(self.pending) >= self.waiting:
            self._give_back(reader)
            self.paused.append(reader)
        else:
            self._read(reader)

    async def _reader(self, pending: _Pending) -> _DirectoryReader | _SubtreeReader:
        """The reader of a directory to read, as it is to be read.

        A subtree to read whole by the handler's ``ils_recursive`` is read alone
        where that is the library's default, a directory per request: the walk
        then lists it so itself, depth first and concurrently, as it lists any
        other directory. Nothing lists ahead of it beyond its control.
        """
        if pending.how != "alone":
            entry = pending.entry or self.handler.entry(pending.name)
            try:
                entries = self.handler.ils_recursive(entry, **self.ils_kwargs)
            except Exception as e:
                entries = _raising(e)  # Its read fails as the directory's.
            if not isinstance(entries, _DepthFirst):
                keep = (
                    partial(self._kept, pending.name, pending.depth)
                    if pending.how == "filtered"
                    else None
                )
                return _SubtreeReader(
                    self.handler,
                    pending,
                    self.ils_kwargs,
                    self.chunk_size,
                    keep,
                    entries,
                )
            await entries.aclose()
            pending = replace(pending, how="alone")
        return _DirectoryReader(self.handler, pending, self.ils_kwargs, self.chunk_size)

    def _root(self, name: str, entry: Entry | None) -> _Pending:
        depth = self.visit.depth(name)
        listed = entry or {"name": name, "type": "directory"}
        how = self._how(
            listed, depth, self.visit.root(listed, depth), after_break=False
        )
        assert how is not None  # A root is never left.
        return _Pending(name, depth, entry, how, root=True)

    def _how(
        self,
        entry: Entry,
        depth: int,
        need: Literal["skip", "directory", "subtree"],
        *,
        after_break: bool,
    ) -> _How | None:
        """How a directory is read: as the visit needs it, and the handler lists it. None if left."""
        if need == "skip":
            return None
        elif need == "subtree":
            if after_break or not self.handler.lists_subtree(entry):
                return "alone"
            return "whole"
        elif (
            not after_break
            and self.visit.by_path
            and self.handler.lists_subtree(entry, pruned=True)
        ):
            return "filtered"
        else:
            return "alone"

    def _kept(self, root: str, root_depth: int, name: str, depth: int) -> bool:
        """Whether the visit lists a directory of a filtered subtree: it enters each directory down to it, by path."""
        if name == root:
            return True
        parent = name.rpartition("/")[0]
        entry = {"name": name, "type": "directory"}
        return (
            self._kept(root, root_depth, parent, depth - 1)
            and self.visit.enter(entry, depth) != "skip"
        )

    def _take(
        self, reader: _DirectoryReader | _SubtreeReader, read: _Read
    ) -> list[_Event]:
        """The events of one read: its parts, what the visit left, and the subtrees they end."""
        events: list[_Event] = []
        for part in read.parts:
            events += self._part(reader, part)
        # The directories the listing was in go first, then what is below them.
        self._push([self._found(e, d) for e, d in read.found])
        self.pending += reversed(read.again)
        return events

    def _part(
        self, reader: _DirectoryReader | _SubtreeReader, part: _Part
    ) -> list[_Event]:
        """The events of one part, its subdirectories to read pushed."""
        if part.error is not None:
            if reader.pending.root and part.directory in self.roots and part.first:
                if isinstance(part.error, NotADirectoryError):
                    return [_FileRoot(part.directory, part.error)]
                raise part.error  # Raised here, before another read starts.
            _note(part.error, part.directory)
            events: list[_Event] = [part]
            subdirs: list[str] = []
        elif isinstance(reader, _SubtreeReader):
            skipped, subdirs = self._in_subtree(reader, part)
            events = [*skipped, part]
        else:
            skipped, subdirs = self._plan(part, reader.pending.after_break)
            events = [*skipped, part]
        if self.subtrees is not None:
            events += self.subtrees.add(part, subdirs)
        return events

    def _plan(self, part: _Part, after_break: bool) -> tuple[list[_Skipped], list[str]]:
        """Push the subdirectories of a part listed alone, as the visit and the handler say: the skipped ones, and the names pushed."""
        skipped: list[_Skipped] = []
        pushed: list[_Pending] = []
        for entry in part.entries:
            if entry["type"] == "directory":
                depth = part.depth + 1
                name = entry["name"].rstrip("/")
                listed = {**entry, "name": name}
                how = self._how(
                    listed,
                    depth,
                    self.visit.enter(entry, depth),
                    after_break=after_break,
                )
                if how is None:
                    skipped.append(_Skipped(listed, depth))
                else:
                    pushed.append(
                        _Pending(name, depth, listed, how, after_break=after_break)
                    )
        self._push(pushed)
        return skipped, [p.name for p in pushed]

    def _in_subtree(
        self, reader: _SubtreeReader, part: _Part
    ) -> tuple[list[_Skipped], list[str]]:
        """The subdirectories of a part of a subtree: all listed in it, but those a filtered subtree leaves."""
        skipped: list[_Skipped] = []
        subdirs: list[str] = []
        for entry in part.entries:
            if entry["type"] == "directory":
                depth = part.depth + 1
                name = entry["name"].rstrip("/")
                if reader.keep is not None and self.visit.enter(entry, depth) == "skip":
                    skipped.append(_Skipped({**entry, "name": name}, depth))
                else:
                    subdirs.append(name)
        return skipped, subdirs

    def _found(self, entry: Entry, depth: int) -> _Pending | None:
        """A subdirectory a broken listing found, below which nothing came: read anew, as the visit says."""
        name = entry["name"].rstrip("/")
        how = self._how(entry, depth, self.visit.enter(entry, depth), after_break=True)
        return (
            None if how is None else _Pending(name, depth, entry, how, after_break=True)
        )

    def _push(self, pending: Sequence[_Pending | None]) -> None:
        """Push directories to read, the first found on top: read first."""
        self.pending += reversed([p for p in pending if p is not None])

    async def _close(self) -> None:
        """Cancel the reads in flight, give back their permits, and close every reader."""
        readers = [*self.reading.values(), *self.paused]
        for task in self.reading:
            task.cancel()
        await asyncio.gather(*self.reading, return_exceptions=True)
        await self._drop_waiter()
        for reader in list(self.permits):
            self._give_back(reader)
        self.reading.clear()
        self.paused.clear()
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


class _Subtrees:
    """Tells when all below a directory is listed: its own listing, and those of its subdirectories the walk lists.

    A subdirectory's listing may end before its parent's: a recursive listing
    passes a directory only after all below it, and a long listing ends after
    the short ones it found. Such a subdirectory, done already, is not waited
    for.
    """

    def __init__(self) -> None:
        # The subdirectories each directory being listed has shown so far.
        self.subdirs: dict[str, list[str]] = {}
        # Each directory listed whose subtree is not done, with how many of its
        # subdirectories are not, and the directory each of those is in.
        self.waiting: dict[str, int] = {}
        self.parents: dict[str, str] = {}
        # The directories done before their parent's listing ended.
        self.done: set[str] = set()

    def add(self, part: _Part, subdirs: list[str]) -> list[_SubtreeListed]:
        """Note a part's subdirectories the walk lists, and once it is the last, the subtrees it ends, deepest first."""
        self.subdirs.setdefault(part.directory, []).extend(subdirs)
        if not part.last:
            return []
        else:
            return [_SubtreeListed(d) for d in self._ended(part.directory)]

    def _ended(self, directory: str) -> list[str]:
        subdirs = self.subdirs.pop(directory)
        waiting = [s for s in subdirs if s not in self.done]
        self.done.difference_update(subdirs)
        if waiting:
            self.waiting[directory] = len(waiting)
            self.parents.update(dict.fromkeys(waiting, directory))
            return []

        ready = [directory]
        # A subtree done may complete its parent's, and so on up.
        while (parent := self.parents.pop(directory, None)) is not None:
            if self.waiting[parent] > 1:
                self.waiting[parent] -= 1
                return ready
            del self.waiting[parent]
            ready.append(parent)
            directory = parent

        self.done.add(directory)  # Its parent's listing has not ended yet.
        return ready


# Splitting a recursive listing into each directory's parts.


class _OpenDirectory:
    """A directory a recursive listing is in, with its entries held.

    It keeps the names of all its entries come so far, put or held, so that a
    listing broken off in it leaves them out as it is listed again. That holds
    a directory's names whole, but only of the directories open, and only
    their names.
    """

    def __init__(self, name: str, depth: int, entry: Entry | None = None) -> None:
        self.name = name
        self.depth = depth
        # Its entry as its parent's listing gave it, None if derived or the root's path alone.
        self.entry = entry
        self.entries: list[dict[str, Any]] = []
        self.first = True
        # Its subdirectories listed, below which nothing has come yet, in
        # listing order, each with its entry.
        self.pending: dict[str, Entry] = {}
        self.seen: set[str] = set()

    def hold(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        self.seen.add(entry["name"].rstrip("/"))

    def part(self, *, last: bool) -> _Part:
        part = _Part(self.name, self.depth, self.entries, None, self.first, last)
        self.entries, self.first = [], False
        return part


class _DepthFirstSplit:
    """A recursive listing's entries, depth first or in name order, split into each directory's parts.

    All below a directory come together, so it keeps only the directories the
    listing is in, from the root down: those ``open``. A name not below the
    deepest one closes it, for none below it comes later. A directory's entry
    may come apart from what is below it, as "a/b-x" comes between "a/b" and
    "a/b/c" in name order: it waits as ``pending`` in its parent until a path
    below it opens it, or the listing passes it, empty. Depth first, its
    siblings may come in any order: what is below it comes right after it, so
    any other name tells it is empty. A sibling sorting before it only keeps
    it pending longer, never wrongly.

    A directory not listed, as an object store lists only files, gets an entry
    in its parent from the paths below it. A directory's entry comes in a part
    before any part of its own, as the strategies count what is in a directory
    by its parent's listing: before a part is put, those of the directories
    above it are. Once ``chunk_size`` entries are held in all, they are put.
    """

    def __init__(
        self, root: str, root_depth: int, chunk_size: int, entry: Entry | None = None
    ) -> None:
        self.root = root
        self.chunk_size = chunk_size
        self.open = [_OpenDirectory(root, root_depth, entry)]
        self.count = 0  # The entries held.

    @property
    def first(self) -> bool:
        """Whether no part of the root has been put yet."""
        return self.open[0].first if self.open else False

    def add(self, entry: dict[str, Any]) -> list[_Part]:
        """Hold an entry in its directory, and the parts it completes."""
        name = entry["name"].rstrip("/")
        if name == self.root:
            # A file lists itself. A directory may list its own marker.
            if entry["type"] != "directory":
                raise _not_a_directory(self.root, entry)
            return []
        if not _below(name, self.root):
            return []  # No storage should list it.

        parts = self.close(name)
        top = self.open[-1]
        for directory in _between(top.name, name):
            if directory not in top.pending:
                top.hold({"name": directory, "size": 0, "type": "directory"})
                self.count += 1
            # Its entry as listed, to list it by if the listing breaks off in it.
            listed = top.pending.pop(directory, None)
            top = _OpenDirectory(directory, top.depth + 1, listed)
            self.open.append(top)

        if entry["type"] == "directory":
            if name in top.seen:
                return parts  # Listed twice, e.g. as its marker too.
            top.pending[name] = {**entry, "name": name}
        top.hold(entry)
        self.count += 1

        if self.count >= self.chunk_size:
            parts += self.flush()
        return parts

    def close(self, name: str) -> list[_Part]:
        """The last parts of the directories that ``name`` comes after."""
        parts = []
        while len(self.open) > 1 and not _below(name, self.open[-1].name):
            parts += self.pop()

        top = self.open[-1]
        for directory in list(top.pending):
            # Name order: those after it are not passed either.
            if _below(name, directory) or name < _prefix(directory):
                break
            parts += self.empty(directory)
        return parts

    def end(self) -> list[_Part]:
        """The last parts of all still open, once the listing has ended."""
        parts = []
        while self.open:
            parts += self.pop()
        return parts

    def pop(self) -> list[_Part]:
        """The last parts of the deepest directory open, and of those pending in it."""
        top = self.open[-1]
        parts = [p for d in list(top.pending) for p in self.empty(d)]
        self.open.pop()
        parts += self.flush()
        parts.append(top.part(last=True))
        return parts

    def empty(self, directory: str) -> list[_Part]:
        """The one part of a directory pending in the deepest one open: nothing is below it."""
        top = self.open[-1]
        del top.pending[directory]
        return [*self.flush(), _Part(directory, top.depth + 1, [])]

    def flush(self) -> list[_Part]:
        """The parts of all held, parents first."""
        self.count = 0
        return [o.part(last=False) for o in self.open if o.entries]


class _ParentsFirstSplit:
    """A recursive listing's entries, parents first, split into each directory's parts.

    A directory's entry comes before what is below it, but the directories'
    contents interleave, e.g. breadth first: no entry tells that a directory
    is done. So every directory stays open, its entries held as they come,
    and put a part at a time, parents first, once ``chunk_size`` are held in
    all. Their last parts come once the listing ends, the deepest first.

    It answers a break as `_DepthFirstSplit` does: every directory seen is
    ``open``, with the names of all its entries, and none ``pending``. That
    holds every name of the tree, as only a parents-first listing needs.
    """

    def __init__(
        self, root: str, root_depth: int, chunk_size: int, entry: Entry | None = None
    ) -> None:
        self.root = root
        self.chunk_size = chunk_size
        # Every directory seen, by its path, each after its parent.
        self.directories = {root: _OpenDirectory(root, root_depth, entry)}
        self.count = 0  # The entries held.
        self.current = self.directories[root]  # The directory of the entry listed last.

    @property
    def open(self) -> list[_OpenDirectory]:
        """Every directory seen, parents first: all are open until the listing ends."""
        return list(self.directories.values())

    @property
    def first(self) -> bool:
        """Whether no part of the root has been put yet."""
        return self.directories[self.root].first

    def add(self, entry: dict[str, Any]) -> list[_Part]:
        """Hold an entry in its directory, and the parts it fills."""
        name = entry["name"].rstrip("/")
        if name == self.root:
            # A file lists itself. A directory may list its own marker.
            if entry["type"] != "directory":
                raise _not_a_directory(self.root, entry)
            return []
        if not _below(name, self.root):
            return []  # No storage should list it.

        # A directory not listed, as an object store lists only files, gets an
        # entry in its parent from the path below it.
        top = self.directories[self.root]
        for directory in _between(self.root, name):
            if (known := self.directories.get(directory)) is None:
                top.hold({"name": directory, "size": 0, "type": "directory"})
                self.count += 1
                known = self.directories[directory] = _OpenDirectory(
                    directory, top.depth + 1
                )
            top = known

        # The listing moved to another directory: what is held goes, so that the
        # strategies act on it while the listing goes on. Only last parts wait.
        parts = self.flush() if top is not self.current and self.count else []
        self.current = top

        if entry["type"] == "directory":
            if name in self.directories:
                return parts  # Listed twice, e.g. as its marker too.
            listed = {**entry, "name": name}
            self.directories[name] = _OpenDirectory(name, top.depth + 1, listed)
        top.hold(entry)
        self.count += 1

        return parts + self.flush() if self.count >= self.chunk_size else parts

    def flush(self) -> list[_Part]:
        """The parts of all held, parents first."""
        self.count = 0
        return [d.part(last=False) for d in self.directories.values() if d.entries]

    def end(self) -> list[_Part]:
        """The last parts of every directory, once the listing has ended: the deepest first."""
        parts = self.flush()
        parts += [d.part(last=True) for d in reversed(self.directories.values())]
        return parts


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
    """Note its directory on a listing's error, yielded among directories.

    In its note for people, and as an attribute for code, e.g. fsspec's walk.
    """
    error.add_note(f"Listing {directory!r} failed.")
    error.directory = directory  # type: ignore[attr-defined]
