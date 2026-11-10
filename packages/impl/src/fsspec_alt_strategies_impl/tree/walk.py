# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A concurrent walk of a directory tree, a directory per listing request.

`_tree_walk` lists directories breadth first, several at a time, and yields each
directory's ``(path, dirs, files)`` as its listing arrives, as fsspec's ``walk``
gives them. A ``descend`` predicate prunes the directories it lists, as glob needs.

Its engine, ``_chunks``, is the walk under the other tree strategies, such as
find, rm and du. It streams each directory's listing in parts, as ``_ils`` yields
it, so that they act on a directory of hundreds of thousands of files as it is
listed. ``_tree_walk`` gathers the parts, as fsspec's ``walk`` gives a directory
whole.

It suits storages that list one directory per request, and those whose directories
can be pruned. A storage whose ``_ils`` lists a whole tree, as the profile's
``lists_recursively`` declares, is listed so, a root per request, when nothing is
pruned: its listing is split into the same parts.

The tree strategies, and the engines under them, take the filesystem they
work on, and call its primitives, such as ``_ils``, under its
``operation_limit``.

This module depends on `jobs`, `operation` and `paths` of `common`.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from fsspec_alt_strategies.states import WalkState

from ..common.jobs import _Queue
from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim, _triples

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Collection, Iterable

__all__: list[str] = []


def _tree_walk(
    fs: Any,
    path: str,
    *,
    maxdepth: int | None = None,
    topdown: bool = True,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    on_error: Literal["omit", "raise"] | Callable[[Exception], object] | None = None,
    ils_kwargs: dict[str, Any] | None = None,
) -> Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState]:
    """Walk a directory tree, listing several directories at a time.

    It yields each directory listed as ``(path, dirs, files)``, as fsspec's
    ``walk`` does with ``detail=True``: ``path`` itself first, then each directory
    below it. ``dirs`` and ``files`` map the names in the directory to their
    ``_ils`` entries. An empty directory yields empty ones. The directories come in the
    order their listings arrive, not in the tree's order. A directory that fails
    to list yields its error instead, and the walk goes on with the others.
    The error names its directory in its note, and as its ``directory`` attribute.
    ``on_error`` handles those errors as with fsspec's ``walk``, otherwise.

    With ``topdown``, a directory comes before those below it. Without, it comes
    after all of those below it, as fsspec's ``walk`` orders them. The
    listings run as concurrently either way, but bottom up holds back each
    directory until its subtree is done. So a walk bottom up that is cancelled
    leaves out the directories whose subtrees were not done.

    Once cancelled, it lists no new directory, and yields the directories of the
    listings in flight. Iterating it raises these errors, which end it:

    - FileNotFoundError, if ``path`` does not exist;
    - NotADirectoryError, if ``path`` is a file;
    - with ``on_error="raise"``, the error of the first directory that failed to
      list.

    Requests: one ``_ils`` per directory walked, as the filesystem's
    ``operation_limit`` allows.

    ``path`` is the directory the walk starts from, listed by the ``_ils`` of
    ``fs``. ``maxdepth`` of 1 lists only ``path``. ``topdown`` puts a directory
    before those below it. ``descend``, called with the entry of each
    subdirectory found, lets the walk list it only if it returns True.
    ``ils_kwargs`` go to each ``_ils``, as fsspec's ``walk`` passes its own on.

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

    async def run(
        op: Operation[tuple[str, dict[str, Any], dict[str, Any]], WalkState],
    ) -> AsyncGenerator[tuple[str, dict[str, Any], dict[str, Any]] | Exception]:
        listings = _listings(
            fs,
            path,
            maxdepth=maxdepth,
            topdown=topdown,
            descend=descend,
            raise_on_error=on_error == "raise",
            ils_kwargs=ils_kwargs,
            stopping=op.cancelling,
        )
        state = op.state

        async with aclosing(listings):
            async for listing in listings:
                op.state = state = WalkState(
                    errors=state.errors + (listing.error is not None),
                    current=listing.directory,
                    directories_listed=state.directories_listed + 1,
                    found=state.found + len(listing.entries),
                )

                if listing.error is None:
                    yield _triples(listing.directory, listing.entries, detail=True)
                elif on_error is None:
                    yield listing.error
                elif callable(on_error):
                    on_error(listing.error)
                # Else "omit": left out.

    return Operation(run, WalkState(directories_listed=0))


@dataclass(frozen=True, slots=True)
class _Listing:
    """A part of a directory's listing, as ``_chunks`` yields it, or all of it.

    ``depth`` is 1 for the directory the walk started from, 2 for its subdirectories,
    and so on. ``entries`` are as ``_ils`` gives them, without the directory itself,
    and empty if the listing failed with ``error``.

    A directory's listing comes in parts, each of at most ``chunk_size`` entries,
    as ``_ils`` yields them: ``first`` on the first part, ``last`` on the last.
    So an empty directory is one part, both first and last, with no entries. A
    failed listing ends with its error, as its last part. ``_listings`` yields
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


_CHUNK_SIZE = 1000
"""The most entries of one part of a listing: a directory's are never all held."""


async def _chunks(
    fs: Any,
    path: str | Collection[str],
    *,
    maxdepth: int | None = None,
    matched: Collection[str] = (),
    descend: Callable[[dict[str, Any]], bool] | None = None,
    whole: Callable[[dict[str, Any]], bool] | None = None,
    raise_on_error: bool = False,
    ils_kwargs: dict[str, Any] | None = None,
    stopping: Callable[[], object] | None = None,
    chunk_size: int = _CHUNK_SIZE,
) -> AsyncGenerator[_Listing]:
    """Walk a directory tree, yielding each directory's listing in parts, as they come.

    It is the walk under the strategies acting on what they find, such as rm
    and find: a directory of hundreds of thousands of files is streamed, a part
    at a time, never held whole. A subdirectory is listed as soon as its entry
    comes, while its parent's listing goes on.

    ``path`` may be several directories, e.g. a glob's matches, walked at once, each
    as a root. None may be inside another. A root's listing failing before its
    first part raises, as fsspec's walk does, since nothing has been done yet:
    FileNotFoundError, or NotADirectoryError for a file. Any other failed
    listing is yielded as its last part, with its error, and its subdirectories not
    listed yet are not walked. With ``raise_on_error``, it raises instead.

    The roots in ``matched`` start one level down, at depth 2: a pattern's
    matches are walked one level less deep than ``maxdepth``, as fsspec's
    ``_expand_path`` walks them.

    The parts of a directory come in order, but those of several directories mix, as
    the listings run at once. Each listing, by ``fs._ils`` with
    ``ils_kwargs``, holds a permit of ``fs.operation_limit`` while it runs.

    Once ``stopping`` returns true, no new listing starts, and those in flight
    stop after their current part: a directory stopped so has no last part.

    With the profile's ``lists_recursively=True``, a tree that nothing prunes
    is listed whole by one recursive ``_ils``, split into the same parts, as
    `_produce_tree` says. Without ``descend``, that is each root. With it,
    ``whole`` tells of a directory's entry whether ``descend`` holds for every
    directory below it, e.g. below a glob's ``**``: such a directory is listed
    so, and the walk prunes only above it.
    """
    _check_maxdepth(maxdepth)

    roots = {_trim(p) for p in ([path] if isinstance(path, str) else path)}
    trees = _Trees(await _lists_recursively(fs), descend, whole)
    # The directories to list, each with its depth.
    directories: _Queue[tuple[str, int]] = _Queue(fs.operation_limit)
    # The parts listed, a few at a time: a listing waits while they are full.
    parts: asyncio.Queue[_Listing] = asyncio.Queue(maxsize=16)
    # The subdirectories found so far in each directory being listed, each listed once.
    found: dict[str, set[str]] = {}

    def stopped() -> bool:
        return stopping is not None and bool(stopping())

    async def start() -> None:
        if stopped():
            directories.waiting.clear()
        await directories.start(
            lambda item: (
                _produce_tree(
                    fs, ils_kwargs or {}, *item, maxdepth, parts, chunk_size, stopped
                )
                if item[0] in trees.roots
                else _produce(
                    fs,
                    ils_kwargs or {},
                    *item,
                    item[0] in roots,
                    parts,
                    chunk_size,
                    stopped,
                )
            )
        )

    trees.add(roots)
    directories.waiting.extend(
        (root, 2 if root in matched else 1) for root in sorted(roots)
    )
    await start()

    try:
        while directories.waiting or directories.running or not parts.empty():
            if (part := await _next_part(parts, directories)) is None:
                await start()  # A listing ended, or a permit came: start more.
                continue

            if part.error is not None:
                # Raised here, it stops the walk before another listing starts.
                if (part.directory in roots and part.first) or raise_on_error:
                    raise part.error
                _note(part.error, part.directory)
            elif not trees.listed(part.directory) and (
                maxdepth is None or part.depth < maxdepth
            ):
                subdirs = _new_subdirs(part, descend, found)
                trees.add(s for s, _ in subdirs)
                directories.waiting.extend(subdirs)

            if part.last:
                found.pop(part.directory, None)

            # Start the next listings before yielding, so that they run while
            # the caller handles this part.
            await start()
            yield part
    finally:
        await directories.close()


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
    fs: Any,
    ils_kwargs: dict[str, Any],
    directory: str,
    depth: int,
    root: bool,
    parts: asyncio.Queue[_Listing],
    chunk_size: int,
    stopped: Callable[[], bool],
) -> None:
    """List a directory, putting its parts in ``parts`` as they come, its error last."""
    first = True
    chunk: list[dict[str, Any]] = []
    try:
        entries = fs._ils(directory, **ils_kwargs)
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


async def _lists_recursively(fs: Any) -> bool:
    """Whether the profile of ``fs`` declares a recursive ``_ils``."""
    get = getattr(fs, "_get_storage_profile", None)
    return get is not None and (await get()).lists_recursively is True


async def _produce_tree(
    fs: Any,
    ils_kwargs: dict[str, Any],
    root: str,
    root_depth: int,
    maxdepth: int | None,
    parts: asyncio.Queue[_Listing],
    chunk_size: int,
    stopped: Callable[[], bool],
) -> None:
    """List a tree by one recursive ``_ils``, putting each directory's parts as a walk would.

    The entries come in any order, so each is held under its directory, and a
    directory's last part comes only once the listing ends. A directory not
    listed, as an object store lists only files, gets an entry in its parent
    from the paths below it. A directory's entry always comes in a part before
    any part of its own, as the strategies count what is in a directory by its
    parent's listing.

    Its parts hold at most ``chunk_size`` entries in all: once that many are
    held, every directory's are put, parents first. What is deeper than
    ``maxdepth`` is left out, as a walk would not list it.
    """
    tree = _Split(root, root_depth, maxdepth)
    try:
        entries = fs._ils(root, recursive=True, **ils_kwargs)
        try:
            async for entry in entries:
                tree.add(entry)
                if tree.count >= chunk_size:
                    for part in tree.parts(last=False):
                        await parts.put(part)
                    if stopped():
                        return
        finally:
            if (aclose := getattr(entries, "aclose", None)) is not None:
                await aclose()
    except Exception as e:
        await parts.put(_Listing(root, root_depth, [], e, root in tree.first, True))
        return

    for part in tree.parts(last=True):
        await parts.put(part)


class _Split:
    """A recursive listing's entries, held under their directories until put as parts."""

    def __init__(self, root: str, root_depth: int, maxdepth: int | None) -> None:
        self.root = root
        self.maxdepth = maxdepth
        self.held: dict[str, list[dict[str, Any]]] = {root: []}
        self.depths: dict[str, int] = {root: root_depth}
        self.first: set[str] = {root}  # The directories without a part put yet.
        self.count = 0  # The entries held.

    def add(self, entry: dict[str, Any]) -> None:
        """Hold an entry under its directory, noting the directories it tells of."""
        name = entry["name"].rstrip("/")
        if name == self.root:
            # A file lists itself. A directory may list its own marker.
            if entry["type"] != "directory":
                raise _not_a_directory(self.root, entry)
            return
        if not name.startswith(f"{self.root}/" if self.root else ""):
            return  # Not below the root: no storage should list it.

        parent = name.rpartition("/")[0]
        self.known(parent)
        if entry["type"] == "directory":
            if name in self.depths:
                return  # Known already, from a path below it.
            self.note(name, parent)
        self.hold(parent, entry)

    def known(self, directory: str) -> None:
        """Note a directory not listed, with an entry in its parent, parents first."""
        if directory in self.depths:
            return
        parent = directory.rpartition("/")[0]
        self.known(parent)
        self.hold(parent, {"name": directory, "size": 0, "type": "directory"})
        self.note(directory, parent)

    def note(self, directory: str, parent: str) -> None:
        self.depths[directory] = self.depths[parent] + 1
        self.held[directory] = []
        self.first.add(directory)

    def deeper(self, directory: str) -> bool:
        return self.maxdepth is not None and self.depths[directory] > self.maxdepth

    def hold(self, directory: str, entry: dict[str, Any]) -> None:
        if not self.deeper(directory):
            self.held[directory].append(entry)
            self.count += 1

    def parts(self, *, last: bool) -> list[_Listing]:
        """The parts of all held, parents first: their entries come before their own parts."""
        parts = []
        for directory in sorted(self.held, key=self.depths.__getitem__):
            entries = self.held[directory]
            if not (entries or last) or self.deeper(directory):
                continue
            first = directory in self.first
            parts.append(
                _Listing(directory, self.depths[directory], entries, None, first, last)
            )
            self.first.discard(directory)
            self.held[directory] = []
        self.count = 0
        return parts


def _new_subdirs(
    part: _Listing,
    descend: Callable[[dict[str, Any]], bool] | None,
    found: dict[str, set[str]],
) -> list[tuple[str, int]]:
    """The subdirectories a part adds to list, with their depth, those found before left out.

    Each comes once, though a storage may list a directory twice, e.g. with and
    without its marker object, in two parts.
    """
    seen = found.setdefault(part.directory, set())
    new = [s for s in dict.fromkeys(_subdirs(part.entries, descend)) if s not in seen]
    seen.update(new)
    return [(s, part.depth + 1) for s in new]


class _Trees:
    """The directories a walk lists whole, by one recursive ``_ils`` each.

    A directory is so listed when the storage lists recursively, and the walk
    prunes nothing below it: it has no ``descend``, or ``whole`` holds for it.
    The directories below are then in its listing, never listed on their own.
    """

    def __init__(
        self,
        recursive: bool,
        descend: Callable[[dict[str, Any]], bool] | None,
        whole: Callable[[dict[str, Any]], bool] | None,
    ) -> None:
        self.recursive = recursive
        self.descend = descend
        self.whole = whole
        self.roots: set[str] = set()

    def add(self, directories: Iterable[str]) -> None:
        """Note those of the directories to list that are listed whole."""
        if self.recursive:
            self.roots.update(d for d in directories if self._whole(d))

    def _whole(self, directory: str) -> bool:
        if self.descend is None:
            return True
        return self.whole is not None and self.whole(
            {"name": directory, "type": "directory"}
        )

    def listed(self, directory: str) -> bool:
        """Whether the directory's subdirectories are in a recursive listing."""
        return any(
            r in (directory, "") or directory.startswith(r + "/") for r in self.roots
        )


def _not_a_directory(root: str, entry: dict[str, Any]) -> NotADirectoryError:
    """The error of a root that is a file, with its entry as the listing gave it.

    A copy or move of the root then hands the entry on, with no ``_info``.
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


async def _listings(
    fs: Any,
    path: str,
    *,
    maxdepth: int | None = None,
    topdown: bool = True,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    raise_on_error: bool = False,
    ils_kwargs: dict[str, Any] | None = None,
    stopping: Callable[[], object] | None = None,
) -> AsyncGenerator[_Listing]:
    """Walk a directory tree as `_tree_walk` does, yielding each directory's listing whole.

    It gathers the parts of `_chunks` into one listing per directory, as fsspec's
    walk gives a directory at a time, so it holds each directory until it is done.
    A failed listing comes as one, with its error and no entries. A directory
    stopped before its last part is left out.

    Bottom up, each directory waits for its subdirectories, which are only all known
    once its listing ends.
    """
    held: dict[str, list[dict[str, Any]]] = {}
    bottom_up = None if topdown else _BottomUp()
    chunks = _chunks(
        fs,
        path,
        maxdepth=maxdepth,
        descend=descend,
        raise_on_error=raise_on_error,
        ils_kwargs=ils_kwargs,
        stopping=stopping,
    )

    async with aclosing(chunks):
        async for part in chunks:
            entries = held.setdefault(part.directory, [])
            entries.extend(part.entries)
            if not part.last:
                continue

            del held[part.directory]
            if part.error is not None:
                listing = _Listing(part.directory, part.depth, [], part.error)
            else:
                listing = _Listing(part.directory, part.depth, entries)

            if bottom_up is None:
                yield listing
                continue

            subdirs = []
            if listing.error is None and (maxdepth is None or listing.depth < maxdepth):
                subdirs = list(dict.fromkeys(_subdirs(listing.entries, descend)))
            for ready in bottom_up.add(listing, subdirs):
                yield ready


def _subdirs(
    entries: list[dict[str, Any]],
    descend: Callable[[dict[str, Any]], bool] | None,
) -> list[str]:
    """The subdirectories among entries to list next."""
    return [
        e["name"].rstrip("/")
        for e in entries
        if e["type"] == "directory" and (descend is None or descend(e))
    ]


class _BottomUp:
    """Holds each listing back until the listings of its subdirectories are out."""

    def __init__(self) -> None:
        # Each listing held back, with how many of its subdirectories are not out.
        self.held: dict[str, tuple[_Listing, int]] = {}
        self.parents: dict[str, str] = {}

    def add(self, listing: _Listing, subdirs: list[str]) -> list[_Listing]:
        """The listings ready once this one is in, deepest first."""
        if subdirs:
            self.held[listing.directory] = listing, len(subdirs)
            self.parents.update(dict.fromkeys(subdirs, listing.directory))
            return []

        ready = [listing]
        directory = listing.directory

        # A subtree done may complete its parent's, and so on up.
        while (parent := self.parents.pop(directory, None)) is not None:
            held, remaining = self.held[parent]
            if remaining > 1:
                self.held[parent] = held, remaining - 1
                break
            del self.held[parent]
            ready.append(held)
            directory = parent

        return ready
