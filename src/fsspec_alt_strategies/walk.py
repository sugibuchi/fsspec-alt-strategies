# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A concurrent walk of a folder tree, a folder per listing request.

`iwalk` lists folders breadth first, several at a time, and yields each entry
as its folder's listing arrives. A ``descend`` predicate prunes the folders it
lists, as glob needs. Its engine, ``_listings``, is also the engine under the
tree strategies, such as find, rm and du.

It suits storages that list one folder per request, and those whose folders
can be pruned. On object stores, one recursive listing usually costs fewer
requests: `fsspec.asyn.AsyncFileSystem._find` there.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsspec.asyn import _get_batch_size

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Callable

    from fsspec.asyn import AsyncFileSystem

__all__ = ["iwalk"]


async def iwalk(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    topdown: bool = True,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    raise_on_error: bool = False,
    batch_size: int | None = None,
) -> AsyncIterator[dict[str, Any] | Exception]:
    """Walk a folder tree, listing several folders at a time.

    It yields the ``_ls`` entry of each file and folder below ``path``, but not
    ``path`` itself. The folders' listings come in the order they arrive, not
    in the tree's order. A folder that fails to list yields its error, and the
    walk goes on with the other folders.

    With ``topdown``, a folder's entries come before those below it. Without,
    they come after all of those below it, as fsspec's ``walk`` orders them.
    The listings run as concurrently either way, but bottom up holds back each
    folder's entries until its subtree is done.

    Unlike most strategies, `iwalk` takes no callback and returns no result:
    the caller drives it. Breaking out of the loop stops it. Close it with
    `contextlib.aclosing`, so that its listings in flight are cancelled at once.
    New listings start only as the caller asks for entries, so a slow loop body
    slows the walk.

    Requests: one ``_ls`` per folder walked, at most ``batch_size`` at a time.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path : str
        The folder to start from.
    maxdepth : int, optional
        The deepest level to list: 1 lists only ``path``. No limit by default.
    topdown : bool
        Whether a folder's entries come before those below it, rather than
        after.
    descend : callable, optional
        Called with the ``_ls`` entry of each subfolder found. The walk lists it
        only if this returns True. All by default.
    raise_on_error : bool
        Whether a folder that fails to list raises its error, ending the walk,
        rather than yielding it.
    batch_size : int, optional
        The most listings at a time. fsspec's default if None.

    Yields
    ------
    dict or Exception
        The ``_ls`` entry of each file and folder found, or the error of a
        folder that failed to list.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    NotADirectoryError
        If ``path`` is a file.
    ValueError
        If ``maxdepth`` is less than 1.
    Exception
        With ``raise_on_error``, the error of the first folder that failed to
        list.

    Examples
    --------
    >>> from contextlib import aclosing
    >>> async with aclosing(iwalk(fs, "data", batch_size=16)) as entries:
    ...     async for entry in entries:
    ...         if isinstance(entry, Exception):
    ...             raise entry
    ...         print(entry["name"], entry["type"])
    """
    async with aclosing(
        _listings(
            fs,
            path,
            maxdepth=maxdepth,
            topdown=topdown,
            descend=descend,
            batch_size=batch_size,
        )
    ) as listings:
        async for listing in listings:
            if listing.error is not None:
                if raise_on_error:
                    raise listing.error
                yield listing.error
            for entry in listing.entries:
                yield entry


@dataclass(frozen=True, slots=True)
class _Listing:
    """A folder's listing, as ``_listings`` yields it.

    ``depth`` is 1 for the folder the walk started from, 2 for its subfolders,
    and so on. ``entries`` are as the filesystem's ``_ls`` gives them, without
    the folder itself, and empty if the listing failed with ``error``.
    """

    folder: str
    depth: int
    entries: list[dict[str, Any]]
    error: Exception | None = None


async def _listings(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    topdown: bool = True,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    batch_size: int | None = None,
) -> AsyncGenerator[_Listing]:
    """Walk a folder tree as `iwalk` does, yielding each folder's listing.

    The tree strategies need a listing's folder, which `iwalk`'s entries do
    not carry. A failed listing is yielded with its error, and its subfolders
    are not walked. ``path``'s own listing fails by raising, as fsspec's walk
    does, since nothing has been done yet.
    """
    if maxdepth is not None and maxdepth < 1:
        raise ValueError(f"maxdepth must be at least 1: {maxdepth}")
    root = fs._strip_protocol(path)
    batch_size = batch_size or fs.batch_size or _get_batch_size(nofiles=True)
    waiting: deque[tuple[str, int]] = deque()
    running: dict[asyncio.Task[list[dict[str, Any]]], tuple[str, int]] = {}
    bottom_up = None if topdown else _BottomUp()

    def start() -> None:
        while waiting and len(running) < batch_size:
            folder, depth = waiting.popleft()
            running[asyncio.ensure_future(fs._ls(folder, detail=True))] = folder, depth

    waiting.append((root, 1))
    start()
    try:
        while running:
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                folder, depth = running.pop(task)
                listing = _listing(task, folder, depth, root)
                subfolders = _subfolders(listing, maxdepth, descend)
                waiting.extend((s, depth + 1) for s in subfolders)
                # Start the next listings before yielding, so that they run
                # while the caller handles this one.
                start()
                if bottom_up is None:
                    yield listing
                else:
                    for ready in bottom_up.add(listing, subfolders):
                        yield ready
    finally:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)


def _listing(
    task: asyncio.Task[list[dict[str, Any]]], folder: str, depth: int, root: str
) -> _Listing:
    """The listing a finished ``_ls`` task gives, or its error."""
    try:
        entries = task.result()
    except Exception as e:
        if folder == root:
            # Nothing has been done yet: fail as fsspec's own walk does.
            raise
        return _Listing(folder, depth, [], e)
    if folder == root and any(
        e["name"].rstrip("/") == root and e["type"] != "directory" for e in entries
    ):
        # fsspec's _ls of a file lists the file itself.
        raise NotADirectoryError(root)
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
