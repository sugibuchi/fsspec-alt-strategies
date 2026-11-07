# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A concurrent walk of a folder tree, a folder per listing request.

`tree_walk` lists folders breadth first, several at a time, and yields each entry
as its folder's listing arrives. A ``descend`` predicate prunes the folders it
lists, as glob needs. Its engine, ``_listings``, is also the engine under the
other tree strategies, such as find, rm and du.

It suits storages that list one folder per request, and those whose folders
can be pruned. On object stores, one recursive listing usually costs fewer
requests: `fsspec.asyn.AsyncFileSystem._find` there.

The tree strategies take the storage's operations as functions, such as
``ls``, rather than a filesystem: see `common.operations`.

This module depends on `control`, and `jobs` and `paths` of `common`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..common.jobs import _Permits
from ..common.paths import _check_maxdepth, _trim
from ..control import Operation, State

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Collection

    from ..common.operations import _Info, _Ls

__all__ = ["WalkState", "tree_walk"]


@dataclass(frozen=True, kw_only=True)
class WalkState(State):
    """The progress of a `tree_walk`.

    Attributes
    ----------
    folders_listed : int
        The folders listed so far, those that failed included.
    found : int
        The entries found so far.
    """

    folders_listed: int = 0
    found: int = 0


def tree_walk(
    path: str,
    *,
    ls: _Ls,
    maxdepth: int | None = None,
    topdown: bool = True,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    raise_on_error: bool = False,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[dict[str, Any], WalkState]:
    """Walk a folder tree, listing several folders at a time.

    It yields the ``ls`` entry of each file and folder below ``path``, but not
    ``path`` itself. The folders' listings come in the order they arrive, not
    in the tree's order. A folder that fails to list yields its error, and the
    walk goes on with the other folders.

    With ``topdown``, a folder's entries come before those below it. Without,
    they come after all of those below it, as fsspec's ``walk`` orders them.
    The listings run as concurrently either way, but bottom up holds back each
    folder's entries until its subtree is done. So a walk bottom up that is
    cancelled leaves out the folders whose subtrees were not done.

    Once cancelled, it lists no new folder, and yields the entries of the
    listings in flight. Iterating it raises these errors, which end it:

    - FileNotFoundError, if ``path`` does not exist;
    - NotADirectoryError, if ``path`` is a file;
    - with ``raise_on_error``, the error of the first folder that failed to
      list.

    Requests: one ``ls`` per folder walked, as ``concurrency_limit`` allows.

    Parameters
    ----------
    path : str
        The folder to start from, as ``ls`` takes it.
    ls : callable
        Lists a folder, called as ``ls(path, detail=True)``: e.g. the
        filesystem's ``_ls``.
    maxdepth : int, optional
        The deepest level to list: 1 lists only ``path``. No limit by default.
    topdown : bool
        Whether a folder's entries come before those below it, rather than
        after.
    descend : callable, optional
        Called with the ``ls`` entry of each subfolder found. The walk lists it
        only if this returns True. All by default.
    raise_on_error : bool
        Whether a folder that fails to list raises its error, ending the walk,
        rather than yielding it.
    concurrency_limit : asyncio.Semaphore
        Held by each request while it runs. Share one to limit several
        operations, or files, together.

    Returns
    -------
    Operation
        Yielding the ``ls`` entry of each file and folder found, or the error
        of a folder that failed to list. Its state is a `WalkState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1.

    See Also
    --------
    fsspec_alt_strategies.control.collect : All the entries, at once.

    Examples
    --------
    >>> limit = asyncio.Semaphore(16)  # Shared, e.g. by a filesystem's operations.
    >>> walk = tree_walk("data", ls=fs._ls, concurrency_limit=limit)
    >>> async with walk as entries:
    ...     async for entry in entries:
    ...         if isinstance(entry, Exception):
    ...             raise entry
    ...         print(entry["name"], entry["type"])
    """
    _check_maxdepth(maxdepth)

    async def run(
        op: Operation[dict[str, Any], WalkState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        listings = _listings(
            ls,
            path,
            maxdepth=maxdepth,
            topdown=topdown,
            descend=descend,
            raise_on_error=raise_on_error,
            concurrency_limit=concurrency_limit,
            stopping=op.cancelling,
        )
        state = op.state
        async with aclosing(listings):
            async for listing in listings:
                op.state = state = WalkState(
                    errors=state.errors + (listing.error is not None),
                    current=listing.folder,
                    folders_listed=state.folders_listed + 1,
                    found=state.found + len(listing.entries),
                )
                if listing.error is not None:
                    yield listing.error
                for entry in listing.entries:
                    yield entry

    return Operation(run, WalkState())


@dataclass(frozen=True, slots=True)
class _Listing:
    """A folder's listing, as ``_listings`` yields it.

    ``depth`` is 1 for the folder the walk started from, 2 for its subfolders,
    and so on. ``entries`` are as ``ls`` gives them, without
    the folder itself, and empty if the listing failed with ``error``.
    """

    folder: str
    depth: int
    entries: list[dict[str, Any]]
    error: Exception | None = None


async def _listings(
    ls: _Ls,
    path: str | Collection[str],
    *,
    maxdepth: int | None = None,
    topdown: bool = True,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    raise_on_error: bool = False,
    concurrency_limit: asyncio.Semaphore,
    stopping: Callable[[], object] | None = None,
) -> AsyncGenerator[_Listing]:
    """Walk a folder tree as `tree_walk` does, yielding each folder's listing.

    The tree strategies need a listing's folder, which `tree_walk`'s entries do
    not carry. A failed listing is yielded with its error, and its subfolders
    are not walked. ``path``'s own listing fails by raising, as fsspec's walk
    does, since nothing has been done yet.

    ``path`` may be several folders, e.g. a glob's matches, walked at once, each
    as a root. None may be inside another.

    With ``raise_on_error``, any failed listing raises as soon as it is seen:
    before another listing starts, or another one done is yielded.

    Once ``stopping`` returns true, no new listing starts, and those in flight
    are yielded as they arrive. It is an operation's ``cancelling``.

    Each listing holds a permit of ``concurrency_limit`` while it runs.
    """
    _check_maxdepth(maxdepth)
    roots = {_trim(p) for p in ([path] if isinstance(path, str) else path)}
    permits = _Permits(concurrency_limit)
    waiting: deque[tuple[str, int]] = deque()
    running: dict[asyncio.Task[list[dict[str, Any]]], tuple[str, int]] = {}
    bottom_up = None if topdown else _BottomUp()

    async def start() -> None:
        if stopping is not None and stopping():
            waiting.clear()
        while waiting and await permits.take():
            folder, depth = waiting.popleft()
            running[permits.start(ls(folder, detail=True))] = folder, depth

    waiting.extend((root, 1) for root in sorted(roots))
    await start()
    try:
        while running or waiting:
            pending = [*running, *([permits.waiter()] if waiting else [])]
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task not in running:
                    continue  # The waiter's permit, which start takes.
                folder, depth = running.pop(task)
                listing = _listing(task, folder, depth, roots, raise_on_error)
                subfolders = _subfolders(listing, maxdepth, descend)
                waiting.extend((s, depth + 1) for s in subfolders)
                # Start the next listings before yielding, so that they run
                # while the caller handles this one.
                await start()
                if bottom_up is None:
                    yield listing
                else:
                    for ready in bottom_up.add(listing, subfolders):
                        yield ready
            await start()
    finally:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        await permits.close()


async def _scan(
    ls: _Ls,
    info: _Info,
    path: str,
    *,
    maxdepth: int | None = None,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    concurrency_limit: asyncio.Semaphore,
    stopping: Callable[[], object] | None = None,
    file_root: bool = False,
    folder_root: bool = False,
) -> AsyncGenerator[_Listing | dict[str, Any]]:
    """Walk as the strategies that only read do, such as find, du and glob.

    It yields each folder's listing, as ``_listings`` does, but treats ``path``
    itself as fsspec's ``_find`` and ``_glob`` do:

    - missing, it yields nothing;
    - a file, it yields the file's ``info`` entry if ``file_root``, else
      nothing;
    - a folder, it yields the folder's own ``info`` entry after its listing,
      if ``folder_root`` and the walk is not stopping.
    """
    root = _trim(path)
    listings = _listings(
        ls,
        root,
        maxdepth=maxdepth,
        descend=descend,
        concurrency_limit=concurrency_limit,
        stopping=stopping,
    )
    async with aclosing(listings):
        try:
            root_listing = await anext(listings)
        except StopAsyncIteration:
            return  # Stopping before the root was listed.
        except (FileNotFoundError, NotADirectoryError) as e:
            # Only the root's own listing raises them: any later one is yielded.
            if file_root and isinstance(e, NotADirectoryError):
                async with concurrency_limit:
                    entry = await info(root)
                yield entry
            return
        yield root_listing
        if folder_root and not (stopping and stopping()):
            async with concurrency_limit:
                entry = await info(root)
            yield entry
        async for listing in listings:
            yield listing


def _listing(
    task: asyncio.Task[list[dict[str, Any]]],
    folder: str,
    depth: int,
    roots: set[str],
    raise_on_error: bool,
) -> _Listing:
    """The listing a finished ``ls`` task gives, or its error."""
    try:
        entries = task.result()
    except Exception as e:
        # A root's error is raised as fsspec's own walk raises it: nothing has
        # been done yet. Raised here, any other stops the walk before it starts
        # another listing.
        if folder in roots or raise_on_error:
            raise
        # Yielded among entries, the error must tell its folder itself.
        e.add_note(f"Listing {folder!r} failed.")
        return _Listing(folder, depth, [], e)
    if folder in roots and any(
        e["name"].rstrip("/") == folder and e["type"] != "directory" for e in entries
    ):
        # fsspec's _ls of a file lists the file itself.
        raise NotADirectoryError(folder)
    # Some storages list a folder within itself, e.g. as its marker object.
    return _Listing(
        folder, depth, [e for e in entries if e["name"].rstrip("/") != folder]
    )


def _subfolders(
    listing: _Listing,
    maxdepth: int | None,
    descend: Callable[[dict[str, Any]], bool] | None,
) -> list[str]:
    """The subfolders of a listing to list next."""
    if listing.error is not None or (
        maxdepth is not None and listing.depth >= maxdepth
    ):
        return []
    # Each once, though a storage may list a folder twice, e.g. with and without
    # its marker object: bottom up counts on it.
    return list(
        dict.fromkeys(
            e["name"].rstrip("/")
            for e in listing.entries
            if e["type"] == "directory" and (descend is None or descend(e))
        )
    )


class _BottomUp:
    """Holds each listing back until the listings of its subfolders are out."""

    def __init__(self) -> None:
        # Each listing held back, with how many of its subfolders are not out.
        self.held: dict[str, tuple[_Listing, int]] = {}
        self.parents: dict[str, str] = {}

    def add(self, listing: _Listing, subfolders: list[str]) -> list[_Listing]:
        """The listings ready once this one is in, deepest first."""
        if subfolders:
            self.held[listing.folder] = listing, len(subfolders)
            self.parents.update(dict.fromkeys(subfolders, listing.folder))
            return []
        ready = [listing]
        folder = listing.folder
        # A subtree done may complete its parent's, and so on up.
        while (parent := self.parents.pop(folder, None)) is not None:
            held, remaining = self.held[parent]
            if remaining > 1:
                self.held[parent] = held, remaining - 1
                break
            del self.held[parent]
            ready.append(held)
            folder = parent
        return ready
