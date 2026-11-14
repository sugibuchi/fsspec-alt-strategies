# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A concurrent walk of a directory tree, by recursive listings or a directory per listing.

`_tree_walk` yields each directory's ``(path, dirs, files)`` as its listing
arrives, as fsspec's ``walk`` gives them. A ``descend`` predicate prunes the
directories it lists, as glob needs.

Its engine, ``_chunks``, is the walk under the other tree strategies, such as
find, rm and du. It streams each directory's listing in parts, as ``ils``
yields it, so that they act on a directory of hundreds of thousands of files as
it is listed. ``_tree_walk`` gathers the parts, as fsspec's ``walk`` gives a
directory whole.

A tree that nothing prunes is listed by one ``ils_recursive``. Its entries are
split into the same parts. Each directory's last part comes as soon as the
listing has passed it where the order is depth first, which covers name order,
else once the listing ends: unknown, a listing is only taken to give each
directory before what is below it. A handler with no recursive listing of its
own answers it by default with `_tree_ils`, a directory per request, several at
a time, which is depth first whatever the profile says. Otherwise, the profile
declares the order. A listing broken off part way is resumed a directory per
request, from the directories it was in.

Where the walk prunes, it lists each directory it enters on its own, several at
a time, breadth first.

The tree strategies, and the engines under them, take the handler they
work on, and call its primitives, such as ``ils``, under its
``operation_limit``.

This module depends on `_visit` and `listing`, and `handler`, `jobs`, `operation`
and `paths` of `common`.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from fsspec_alt_strategies.states import WalkState

from ..common.jobs import _Queue
from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim, _triples
from ._visit import _Pruned, _TopDown, _Visit
from .listing import _DepthFirst, _tree_ils

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Collection,
    )

    from ..common.handler import Entry, StorageHandler, StoragePath
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
    to their ``ils`` entries. An empty directory yields empty ones. The
    directories come in the order their listings arrive, not in the tree's
    order. A directory that fails to list yields its error instead, and the
    walk goes on with the others. The error names its directory in its note,
    and as its ``directory`` attribute. ``on_error``, below, may handle those
    errors instead, as with fsspec's ``walk``.

    With ``topdown``, a directory comes before those below it, so each is
    listed on its own: a recursive listing passes a directory only after all
    below it, and the whole tree would be held. Without, a directory comes
    after all of those below it, as fsspec's ``walk`` orders them, and the tree
    is listed by one ``ils_recursive``, unless ``maxdepth`` or ``descend``
    prunes it. A listing broken off part way is resumed a directory per
    request, so that a directory failing yields its own error, and the walk
    goes on with the others. A walk bottom up that is cancelled leaves out the
    directories whose subtrees were not done.

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
        # Each directory's parts, held until its last, as fsspec's walk gives a
        # directory whole. Bottom up, a directory then waits until all below it
        # is listed.
        held: dict[str, list[dict[str, Any]]] = {}
        waiting: dict[str, _Listing] = {}
        events = _parts(
            _chunks(
                handler,
                path,
                visit,
                ils_kwargs=ils_kwargs,
                stopping=op.cancelling,
                subtrees=not topdown,
            )
        )

        async with aclosing(events):
            async for event in events:
                if (listing := _ready(event, held, waiting, topdown)) is None:
                    continue
                if listing.error is not None and on_error == "raise":
                    raise listing.error
                op.state = _walk_state(op.state, listing)
                if (item := _walked(listing, on_error)) is not None:
                    yield item

    return Operation(run, WalkState(directories_listed=0))


async def _parts(
    events: AsyncGenerator[_Event],
) -> AsyncGenerator[_Listing | _SubtreeListed]:
    """A walk's parts and subtrees, as fsspec's walk takes them: a root that is a file raises, and what is left is not walked."""
    async with aclosing(events):
        async for event in events:
            if isinstance(event, _FileRoot):
                raise event.error
            elif not isinstance(event, _Skipped):
                yield event


def _ready(
    event: _Listing | _SubtreeListed,
    held: dict[str, list[dict[str, Any]]],
    waiting: dict[str, _Listing],
    topdown: bool,
) -> _Listing | None:
    """The whole listing a walk yields next, if any, as an event makes it ready.

    Top down, a listing is ready once whole. Bottom up, a whole one waits in
    ``waiting`` until all below its directory is listed, though its error is
    ready at once to raise.
    """
    if isinstance(event, _SubtreeListed):
        return waiting.pop(event.directory)
    elif (listing := _whole(event, held)) is None or topdown:
        return listing
    else:
        waiting[listing.directory] = listing
        return None


def _whole(part: _Listing, held: dict[str, list[dict[str, Any]]]) -> _Listing | None:
    """A directory's whole listing once its last part comes, its parts held till then, or None.

    A failed listing comes whole with its error and no entries.
    """
    held.setdefault(part.directory, []).extend(part.entries)
    if not part.last:
        return None
    entries = held.pop(part.directory)
    if part.error is not None:
        return _Listing(part.directory, part.depth, [], part.error)
    else:
        return _Listing(part.directory, part.depth, entries)


def _walked(
    listing: _Listing,
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


def _walk_state(state: WalkState, listing: _Listing) -> WalkState:
    """The walk's state once a directory's listing is out."""
    return WalkState(
        errors=state.errors + (listing.error is not None),
        current=listing.directory,
        directories_listed=(state.directories_listed or 0) + 1,
        found=state.found + len(listing.entries),
    )


@dataclass(frozen=True, slots=True)
class _Listing:
    """A part of a directory's listing, as ``_chunks`` yields it, or all of it.

    ``depth`` is 1 for the directory the walk started from, 2 for its subdirectories,
    and so on. ``entries`` are as ``ils`` gives them, without the directory itself,
    and empty if the listing failed with ``error``.

    A directory's listing comes in parts, each of at most ``chunk_size`` entries,
    as ``ils`` yields them: ``first`` on the first part, ``last`` on the last.
    So an empty directory is one part, both first and last, with no entries. A
    failed listing ends with its error, as its last part. `_tree_walk` holds
    each directory whole: one part, both first and last.

    ``depth`` 0 is no listing, but the walk's root itself, as `_find_listings`
    yields it: ``entries`` hold only the root's ``info`` entry, under ``directory``.
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


_Event = _Listing | _Skipped | _FileRoot | _SubtreeListed
"""What a walk yields: the parts of each directory's listing, what it found but did not list, and when a subtree is done."""


_CHUNK_SIZE = 1000
"""The most entries of one part of a listing: a directory's are never all held."""


async def _chunks(
    handler: TreeFileOperationHandler,
    path: str | Entry | Collection[str | Entry],
    visit: _Visit | None = None,
    *,
    ils_kwargs: dict[str, Any] | None = None,
    stopping: Callable[[], object] | None = None,
    chunk_size: int = _CHUNK_SIZE,
    subtrees: bool = False,
) -> AsyncGenerator[_Event]:
    """Walk a directory tree, yielding each directory's listing in parts, as they come.

    It is the walk under the strategies acting on what they find, such as rm
    and find: a directory of hundreds of thousands of files is streamed, a part
    at a time, never held whole. A subdirectory is listed as soon as its entry
    comes, while its parent's listing goes on.

    ``path`` may be several directories, e.g. a glob's matches, walked at once,
    each as a root. None may be inside another. A root's listing failing before
    its first part raises, as fsspec's walk does, since nothing has been done
    yet, e.g. FileNotFoundError. A root that is a file is yielded as a
    `_FileRoot`, and the other roots walked. Any other failed listing is
    yielded as its last part, with its error, and its subdirectories not
    listed yet are not walked. A subdirectory the visit leaves is yielded as a
    `_Skipped`, before the part that found it. With ``subtrees``, a
    `_SubtreeListed` tells when all below a directory is listed, as a walk
    bottom up needs: it costs the names of the directories whose subtrees
    are not done.

    ``visit`` says which directories the walk lists: by default all below
    each root. A root starts at the depth it says, e.g. a glob's match one
    level down.

    The parts of a directory come in order, but those of several directories
    mix, as the listings run at once. Each listing, by ``handler.ils`` with
    ``ils_kwargs``, holds a permit of ``handler.operation_limit`` while it runs.

    Once ``stopping`` returns true, no new listing starts, and those in flight
    stop after their current part: a directory stopped so has no last part.

    A directory whose visit takes all below it is listed whole by one
    ``ils_recursive``, split into the same parts, as `_produce_tree` says. A
    failure in it is resumed a directory per request, as `_resume` says. Any
    other directory is listed on its own, and the visit asked of each
    subdirectory its listing finds.
    """
    # Each directory to list by the entry its listing gave, e.g. with the
    # storage's id, else by its path: a root given as a path, verified.
    roots, listed = _roots(path)
    plan = _Plan(visit or _Visit(), listed)
    done = _Subtrees(plan) if subtrees else None
    # The directories to list, each with its depth.
    directories: _Queue[tuple[str, int]] = _Queue(handler.operation_limit)
    # The parts listed, a few at a time: a listing waits while they are full.
    parts: asyncio.Queue[_Listing] = asyncio.Queue(maxsize=16)
    # The subdirectories found so far in each directory being listed: a part
    # yields each once, though a storage may list one twice.
    found: dict[str, set[str]] = {}

    def stopped() -> bool:
        return stopping is not None and bool(stopping())

    async def start() -> None:
        if stopped():
            directories.waiting.clear()
        await directories.start(
            lambda item: (
                _produce_tree(
                    handler,
                    ils_kwargs or {},
                    listed.pop(item[0], None),
                    *item,
                    parts,
                    chunk_size,
                    stopped,
                )
                if item[0] in plan.subtrees
                else _produce(
                    handler,
                    ils_kwargs or {},
                    listed.pop(item[0], None),
                    *item,
                    item[0] in roots,
                    parts,
                    chunk_size,
                    stopped,
                )
            )
        )

    directories.waiting.extend(plan.start(roots))
    await start()

    try:
        while directories.waiting or directories.running or not parts.empty():
            if (part := await _next_part(parts, directories)) is None:
                await start()  # A listing ended, or a permit came: start more.
                continue

            # A root's failure is raised here, before another listing starts.
            events, listing = _taken(part, roots, plan, found, done)
            directories.waiting.extend(listing)

            # Start the next listings before yielding, so that they run while
            # the caller handles this part.
            await start()
            for event in events:
                yield event
    finally:
        await directories.close()


def _taken(
    part: _Listing,
    roots: set[str],
    plan: _Plan,
    found: dict[str, set[str]],
    done: _Subtrees | None,
) -> tuple[list[_Event], list[tuple[str, int]]]:
    """What the walk yields of a part, and the subdirectories it adds to list.

    A root's failure before its first part raises, as nothing has been done
    yet, but a file's, which is a `_FileRoot`. Any other failure is the part's,
    noted with its directory. With ``done``, the subtrees the part ends follow
    it.
    """
    if part.error is None:
        part = _once(part, found)
    if part.last:
        found.pop(part.directory, None)

    if part.error is None:
        listing, skipped = plan.found(part)
        events: list[_Event] = [*skipped, part]
    elif part.directory in roots and part.first:
        if not isinstance(part.error, NotADirectoryError):
            raise part.error
        return [_FileRoot(part.directory, part.error)], []
    else:
        _note(part.error, part.directory)
        listing, events = [], [part]

    if done is not None:
        events += done.add(part, listing)
    return events, listing


def _roots(
    path: str | Entry | Collection[str | Entry],
) -> tuple[set[str], dict[str, Entry]]:
    """The roots of a walk, by path, and the entries of those given as entries, e.g. a glob's matches."""
    roots: set[str] = set()
    listed: dict[str, Entry] = {}
    for root in [path] if isinstance(path, (str, dict)) else path:
        if isinstance(root, dict):
            name = _trim(root["name"])
            listed[name] = {**root, "name": name}
        else:
            name = _trim(root)
        roots.add(name)
    return roots, listed


async def _next_part(
    parts: asyncio.Queue[_Listing], directories: _Queue[tuple[str, int]]
) -> _Listing | None:
    """The next part listed, or None once a listing ends, or a permit comes, first."""
    getter = asyncio.ensure_future(parts.get())
    try:
        done, _ = await asyncio.wait(
            [getter, *directories.pending()], return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if task is not getter:
                directories.pop(task)  # A listing's parts are all queued already.
        return getter.result() if getter.done() else None
    finally:
        # A part not got yet stays in the queue: cancelling a get takes none.
        if not getter.done():
            getter.cancel()


async def _produce(
    handler: StorageHandler,
    ils_kwargs: dict[str, Any],
    listed: Entry | None,
    directory: str,
    depth: int,
    root: bool,
    parts: asyncio.Queue[_Listing],
    chunk_size: int,
    stopped: Callable[[], bool],
) -> None:
    """List a directory, putting its parts in ``parts`` as they come, its error last.

    It lists by ``listed``, the entry its parent's listing gave, else by its
    path, verified: a path refused is the listing's error.
    """
    first = True
    chunk: list[dict[str, Any]] = []
    try:
        entries = handler.ils(listed or handler.entry(directory), **ils_kwargs)
        try:
            async for entry in entries:
                if entry["name"].rstrip("/") == directory:
                    # A file lists itself, as fsspec's _ls of a file does. Some
                    # storages list a directory within itself too, e.g. as its
                    # marker object.
                    if entry["type"] != "directory" and root:
                        raise _not_a_directory(directory, entry)
                    continue

                chunk.append(entry)
                if len(chunk) >= chunk_size:
                    await parts.put(
                        _Listing(directory, depth, chunk, None, first, False)
                    )
                    first, chunk = False, []
                    if stopped():
                        return
        finally:
            if (aclose := getattr(entries, "aclose", None)) is not None:
                await aclose()
    except Exception as e:
        await parts.put(_Listing(directory, depth, [], e, first, True))
        return

    await parts.put(_Listing(directory, depth, chunk, None, first, True))


async def _produce_tree(
    handler: TreeFileOperationHandler,
    ils_kwargs: dict[str, Any],
    listed: Entry | None,
    root: str,
    root_depth: int,
    parts: asyncio.Queue[_Listing],
    chunk_size: int,
    stopped: Callable[[], bool],
    *,
    own: bool = True,
) -> None:
    """List a tree by one recursive listing, putting each directory's parts as a walk would.

    The listing is the handler's ``ils_recursive``, or with ``own`` False
    `_tree_ils`, a directory per request. Depth first or in name order, all
    below a directory come together, so a directory is done once the listing
    has passed it: `_Split` puts its last part then. That is `_tree_ils`'s
    order, and a storage's own where the profile declares the
    ``recursive_order`` "depth_first". Otherwise, a listing is only known to
    give a directory before what is below it: `_ParentsFirst` puts every
    directory's last part once the listing ends.

    A listing failing before anything came puts its error as the root's first
    and last part, raised as such for a root of the walk. One broken off later
    is resumed, as `_resume` says: the tree is listed whole all the same, but
    for the directories that fail again. A recursive listing is long, e.g. a
    WebDAV body of minutes, and a connection dropped in it would otherwise
    leave an arbitrary part of the tree unwalked, which fsspec's ``find`` then
    omits with no error.
    """
    entries: AsyncIterator[Entry] | None = None
    try:
        entry = listed or handler.entry(root)
        entries = _recursive_listing(handler, own, root, entry, ils_kwargs)
        splitter = await _splitter(handler, entries)
    except Exception as e:
        # Nothing was listed, or the profile could not be told: the root fails.
        if entries is not None:
            await _close(entries)
        await parts.put(_Listing(root, root_depth, [], e, True, True))
        return
    tree = splitter(root, root_depth, chunk_size, listed)
    try:
        try:
            async for entry in entries:
                for part in tree.add(entry):
                    await parts.put(part)
                if stopped():
                    return
        finally:
            await _close(entries)
    except Exception as e:
        # What was listed first, so that only the root's own failure is a first part.
        for part in tree.flush():
            await parts.put(part)
        if tree.first:
            await parts.put(_Listing(root, root_depth, [], e, True, True))
        else:
            await _resume(handler, ils_kwargs, tree, parts, chunk_size, stopped)
        return

    for part in tree.end():
        await parts.put(part)


def _recursive_listing(
    handler: TreeFileOperationHandler,
    own: bool,
    root: str,
    entry: Entry,
    ils_kwargs: dict[str, Any],
) -> AsyncIterator[Entry]:
    """The recursive listing of a tree: the handler's own, or with ``own`` False, `_tree_ils`."""
    if own:
        return handler.ils_recursive(entry, **ils_kwargs)
    else:
        return _tree_ils(handler, entry, **ils_kwargs)


async def _close(entries: AsyncIterator[Entry]) -> None:
    """Close a listing, where it can be: an ``ils_recursive`` gives any async iterator."""
    if (aclose := getattr(entries, "aclose", None)) is not None:
        await aclose()


async def _splitter(
    handler: TreeFileOperationHandler, entries: AsyncIterator[Entry]
) -> type[_Split | _ParentsFirst]:
    """How a recursive listing is split: depth first where it is `_tree_ils`'s, or the profile declares it so.

    `_tree_ils` lists depth first, whatever the profile says, so a handler
    keeping it as its ``ils_recursive`` needs to declare nothing. A storage's
    own listing keeps the order its profile declares. Unknown, it is taken as
    parents first, which is right on any storage. The handler is opened either
    way, as the walk starts: its profile is loaded as every strategy loads it.
    """
    await handler.open()
    if (
        isinstance(entries, _DepthFirst)
        or handler.profile.recursive_order == "depth_first"
    ):
        return _Split
    else:
        return _ParentsFirst


async def _resume(
    handler: TreeFileOperationHandler,
    ils_kwargs: dict[str, Any],
    tree: _Split | _ParentsFirst,
    parts: asyncio.Queue[_Listing],
    chunk_size: int,
    stopped: Callable[[], bool],
) -> None:
    """Walk what a recursive listing broken off left, a directory per request.

    The directories done have had their last part. Those the listing was in,
    from the root down to the break, are listed again by ``ils``, each on its
    own, their entries put before left out: `_Open` keeps their names. The
    subdirectories none of whose entries came, those found so and those
    pending, are each listed whole by `_tree_ils`, which a failure may break
    off again. So each round resumes deeper in the tree, and ends.

    It runs under the permit of the listing it resumes, one directory or
    subtree after another: a failure's path, not a walk's.
    """
    for directory in list(tree.open):
        subtrees: list[tuple[str, Entry | None]] = [*directory.pending.items()]
        subtrees += await _relist(
            handler, ils_kwargs, directory, parts, chunk_size, stopped
        )
        for name, entry in subtrees:
            if stopped():
                return
            await _produce_tree(
                handler,
                ils_kwargs,
                entry,
                name,
                directory.depth + 1,
                parts,
                chunk_size,
                stopped,
                own=False,
            )
        if stopped():
            return


async def _relist(
    handler: StorageHandler,
    ils_kwargs: dict[str, Any],
    directory: _Open,
    parts: asyncio.Queue[_Listing],
    chunk_size: int,
    stopped: Callable[[], bool],
) -> list[tuple[str, Entry | None]]:
    """List a directory a recursive listing broke off in, putting the entries it had not put.

    It puts the rest of the directory's parts, its error last if it fails
    again, and gives the subdirectories it found, by name with their entries.
    """
    found: list[tuple[str, Entry | None]] = []
    chunk: list[dict[str, Any]] = []
    first = directory.first
    try:
        listed = directory.entry or handler.entry(directory.name)
        entries = handler.ils(listed, **ils_kwargs)
        try:
            async for entry in entries:
                name = entry["name"].rstrip("/")
                if name == directory.name or name in directory.seen:
                    continue
                directory.seen.add(name)
                chunk.append(entry)
                if entry["type"] == "directory":
                    found.append((name, {**entry, "name": name}))
                if len(chunk) >= chunk_size:
                    await parts.put(
                        _Listing(
                            directory.name, directory.depth, chunk, None, first, False
                        )
                    )
                    first, chunk = False, []
                    if stopped():
                        return found
        finally:
            if (aclose := getattr(entries, "aclose", None)) is not None:
                await aclose()
    except Exception as e:
        if chunk:
            await parts.put(
                _Listing(directory.name, directory.depth, chunk, None, first, False)
            )
            first = False
        await parts.put(_Listing(directory.name, directory.depth, [], e, first, True))
        return found

    await parts.put(_Listing(directory.name, directory.depth, chunk, None, first, True))
    return found


class _Split:
    """A recursive listing's entries, depth first or in name order, split into each directory's parts.

    All below a directory come together, so the walk keeps only the directories
    it is in, from the root down: those ``open``. A name not below the deepest
    one closes it, for none below it comes later. A directory's entry may come
    apart from what is below it, as "a/b-x" comes between "a/b" and "a/b/c" in
    name order: it waits as ``pending`` in its parent until a path below it
    opens it, or the listing passes it, empty. Depth first, its siblings may
    come in any order: what is below it comes right after it, so any other
    name tells it is empty. A sibling sorting before it only keeps it pending
    longer, never wrongly.

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
        self.open = [_Open(root, root_depth, entry)]
        self.count = 0  # The entries held.

    @property
    def first(self) -> bool:
        """Whether no part of the root has been put yet."""
        return self.open[0].first if self.open else False

    def add(self, entry: dict[str, Any]) -> list[_Listing]:
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
            top = _Open(directory, top.depth + 1, listed)
            self.open.append(top)

        if entry["type"] == "directory":
            if name in top.pending:
                return parts  # Listed twice, e.g. as its marker too.
            top.pending[name] = {**entry, "name": name}
        top.hold(entry)
        self.count += 1

        if self.count >= self.chunk_size:
            parts += self.flush()
        return parts

    def close(self, name: str) -> list[_Listing]:
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

    def end(self) -> list[_Listing]:
        """The last parts of all still open, once the listing has ended."""
        parts = []
        while self.open:
            parts += self.pop()
        return parts

    def pop(self) -> list[_Listing]:
        """The last parts of the deepest directory open, and of those pending in it."""
        top = self.open[-1]
        parts = [p for d in list(top.pending) for p in self.empty(d)]
        self.open.pop()
        parts += self.flush()
        parts.append(top.part(last=True))
        return parts

    def empty(self, directory: str) -> list[_Listing]:
        """The one part of a directory pending in the deepest one open: nothing is below it."""
        top = self.open[-1]
        del top.pending[directory]
        return [*self.flush(), _Listing(directory, top.depth + 1, [])]

    def flush(self) -> list[_Listing]:
        """The parts of all held, parents first."""
        self.count = 0
        return [o.part(last=False) for o in self.open if o.entries]


class _ParentsFirst:
    """A recursive listing's entries, parents first, split into each directory's parts.

    A directory's entry comes before what is below it, but the directories'
    contents interleave, e.g. breadth first: no entry tells that a directory
    is done. So every directory stays open, its entries held as they come,
    and put a part at a time, parents first, once ``chunk_size`` are held in
    all. Their last parts come once the listing ends, the deepest first.

    It answers `_resume` as `_Split` does: every directory seen is ``open``,
    with the names of all its entries, and none ``pending``. That holds every
    name of the tree, as only a parents-first listing needs.
    """

    def __init__(
        self, root: str, root_depth: int, chunk_size: int, entry: Entry | None = None
    ) -> None:
        self.root = root
        self.chunk_size = chunk_size
        # Every directory seen, by its path, each after its parent.
        self.directories = {root: _Open(root, root_depth, entry)}
        self.count = 0  # The entries held.
        self.current = self.directories[root]  # The directory of the entry listed last.

    @property
    def open(self) -> list[_Open]:
        """Every directory seen, parents first: all are open until the listing ends."""
        return list(self.directories.values())

    @property
    def first(self) -> bool:
        """Whether no part of the root has been put yet."""
        return self.directories[self.root].first

    def add(self, entry: dict[str, Any]) -> list[_Listing]:
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
                known = self.directories[directory] = _Open(directory, top.depth + 1)
            top = known

        # The listing moved to another directory: what is held goes, so that the
        # strategies act on it while the listing goes on. Only last parts wait.
        parts = self.flush() if top is not self.current and self.count else []
        self.current = top

        if entry["type"] == "directory":
            if name in self.directories:
                return parts  # Listed twice, e.g. as its marker too.
            listed = {**entry, "name": name}
            self.directories[name] = _Open(name, top.depth + 1, listed)
        top.hold(entry)
        self.count += 1

        return parts + self.flush() if self.count >= self.chunk_size else parts

    def flush(self) -> list[_Listing]:
        """The parts of all held, parents first."""
        self.count = 0
        return [d.part(last=False) for d in self.directories.values() if d.entries]

    def end(self) -> list[_Listing]:
        """The last parts of every directory, once the listing has ended: the deepest first."""
        parts = self.flush()
        parts += [d.part(last=True) for d in reversed(self.directories.values())]
        return parts


class _Open:
    """A directory the recursive listing is in, with its entries held.

    It keeps the names of all its entries come so far, put or held, so that a
    listing broken off in it leaves them out as `_relist` lists it again. That
    holds a directory's names whole, but only of the directories open, and
    only their names.
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

    def part(self, *, last: bool) -> _Listing:
        part = _Listing(self.name, self.depth, self.entries, None, self.first, last)
        self.entries, self.first = [], False
        return part


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


def _once(part: _Listing, found: dict[str, set[str]]) -> _Listing:
    """A part without the subdirectories its directory listed before.

    A storage may list a directory twice, e.g. as a prefix and as its marker
    object, maybe in two parts: each comes once, so that what consumes the
    parts counts it once. The names kept are a directory's subdirectories, not
    its files, until its last part.
    """
    seen = found.setdefault(part.directory, set())
    kept: list[dict[str, Any]] = []
    for entry in part.entries:
        if entry["type"] == "directory":
            name = entry["name"].rstrip("/")
            if name in seen:
                continue
            seen.add(name)
        kept.append(entry)
    return part if len(kept) == len(part.entries) else replace(part, entries=kept)


class _Plan:
    """The directories a walk lists, and how, as its visit says of each one found.

    A directory whose visit takes all below it is listed whole, by one
    ``ils_recursive``: the directories below are then in its listing, never
    listed on their own, nor asked of. Any other is listed alone, and the visit
    asked of each subdirectory its listing finds.
    """

    def __init__(self, visit: _Visit, listed: dict[str, Entry]) -> None:
        self.visit = visit
        # The entry each directory to list is listed by: the walk's, shared.
        self.listed = listed
        self.subtrees: set[str] = set()

    def start(self, roots: Collection[str]) -> list[tuple[str, int]]:
        """The roots to list, each with its depth: a root is never left."""
        directories = []
        for root in sorted(roots):
            depth = self.visit.depth(root)
            entry = self.listed.get(root) or {"name": root, "type": "directory"}
            if self.visit.root(entry, depth) == "subtree":
                self.subtrees.add(root)
            directories.append((root, depth))
        return directories

    def found(self, part: _Listing) -> tuple[list[tuple[str, int]], list[_Skipped]]:
        """The subdirectories a part adds to list, as `_once` left them, each with its depth, and those the visit leaves.

        Each is listed by its entry, as the part gave it, e.g. with the
        storage's id. A part from a recursive listing adds none, and leaves
        none: all below its root is listed.
        """
        if self.in_subtree(part.directory):
            return [], []
        directories, skipped = [], []
        for entry in part.entries:
            if entry["type"] == "directory":
                depth = part.depth + 1
                need = self.visit.enter(entry, depth)
                name = entry["name"].rstrip("/")
                if need == "skip":
                    skipped.append(_Skipped({**entry, "name": name}, depth))
                    continue
                if need == "subtree":
                    self.subtrees.add(name)
                self.listed[name] = {**entry, "name": name}
                directories.append((name, depth))
        return directories, skipped

    def in_subtree(self, directory: str) -> bool:
        """Whether a directory is in a recursive listing: it, or a directory above it, is listed whole."""
        if "" in self.subtrees:
            return True
        path = directory
        while path:
            if path in self.subtrees:
                return True
            path = path.rpartition("/")[0]
        return False


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


class _Subtrees:
    """Tells when all below a directory is listed: its own listing, and those of its subdirectories the walk lists.

    A subdirectory's listing may end before its parent's: a recursive listing
    passes a directory only after all below it, and a long listing streamed
    ends after the short ones it found. Such a subdirectory, done already, is
    not waited for.
    """

    def __init__(self, plan: _Plan) -> None:
        self.plan = plan
        # The subdirectories each directory being listed has shown so far.
        self.subdirs: dict[str, list[str]] = {}
        # Each directory listed whose subtree is not done, with how many of its
        # subdirectories are not, and the directory each of those is in.
        self.waiting: dict[str, int] = {}
        self.parents: dict[str, str] = {}
        # The directories done before their parent's listing ended.
        self.done: set[str] = set()

    def add(
        self, part: _Listing, listing: list[tuple[str, int]]
    ) -> list[_SubtreeListed]:
        """Note a part's subdirectories, and once it is the last, the subtrees it ends, deepest first.

        A part from a recursive listing shows every subdirectory it holds, each
        listed in the same listing. Any other shows those the walk lists.
        """
        if self.plan.in_subtree(part.directory):
            names = [
                e["name"].rstrip("/") for e in part.entries if e["type"] == "directory"
            ]
        else:
            names = [name for name, _ in listing]
        self.subdirs.setdefault(part.directory, []).extend(names)
        if not part.last:
            return []
        else:
            return [_SubtreeListed(d) for d in self._ended(part.directory)]

    def _ended(self, directory: str) -> list[str]:
        """The directories whose subtree is done once this one's listing ended, deepest first."""
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
