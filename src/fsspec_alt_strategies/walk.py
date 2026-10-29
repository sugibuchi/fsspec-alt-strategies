# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A concurrent walk of a folder tree, a folder per listing request.

`walk` is the engine under the tree strategies, such as find, rm and du. It
lists folders breadth first, several at a time, and yields each listing as it
arrives. A ``descend`` predicate prunes the folders it lists, as glob needs.

It suits storages that list one folder per request, and those whose folders
can be pruned. On object stores, one recursive listing usually costs fewer
requests: `fsspec.asyn.AsyncFileSystem._find` there.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsspec.asyn import _get_batch_size

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from fsspec.asyn import AsyncFileSystem

__all__ = ["Listing", "walk"]


@dataclass(frozen=True, slots=True)
class Listing:
    """A folder's listing, as `walk` yields it.

    Attributes
    ----------
    folder : str
        The folder's path, without the protocol.
    depth : int
        1 for the folder the walk started from, 2 for its subfolders, and so on.
    entries : list of dict
        The folder's entries, as the filesystem's ``_ls`` gives them, without the
        folder itself. Empty if the listing failed.
    error : Exception, optional
        Why the listing failed, if it did. Its subfolders are then not walked.
    """

    folder: str
    depth: int
    entries: list[dict[str, Any]]
    error: Exception | None = None


async def walk(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    descend: Callable[[dict[str, Any]], bool] | None = None,
    batch_size: int | None = None,
) -> AsyncIterator[Listing]:
    """Walk a folder tree, listing several folders at a time.

    The listings come in the order they arrive, not in the tree's order. A
    listing that fails is yielded with its error, and the walk goes on with the
    other folders.

    Unlike most strategies, `walk` takes no callback and returns no result: the
    caller drives it. Breaking out of the loop stops it. Close it with
    `contextlib.aclosing`, so that its listings in flight are cancelled at once.

    Requests: one ``_ls`` per folder walked, at most ``batch_size`` at a time.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path : str
        The folder to start from.
    maxdepth : int, optional
        The deepest level to list: 1 lists only ``path``. No limit by default.
    descend : callable, optional
        Called with the ``_ls`` entry of each subfolder found. The walk lists it
        only if this returns True. All by default.
    batch_size : int, optional
        The most listings at a time. fsspec's default if None.

    Yields
    ------
    Listing
        Each folder's listing, ``path``'s first.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    NotADirectoryError
        If ``path`` is a file.
    ValueError
        If ``maxdepth`` is less than 1.

    Examples
    --------
    >>> from contextlib import aclosing
    >>> async with aclosing(walk(fs, "data", batch_size=16)) as listings:
    ...     async for listing in listings:
    ...         print(listing.folder, len(listing.entries))
    """
    if maxdepth is not None and maxdepth < 1:
        raise ValueError(f"maxdepth must be at least 1: {maxdepth}")
    root = fs._strip_protocol(path)
    batch_size = batch_size or fs.batch_size or _get_batch_size(nofiles=True)
    waiting: deque[tuple[str, int]] = deque()
    running: dict[asyncio.Task[list[dict[str, Any]]], tuple[str, int]] = {}

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
                if listing.error is None and (maxdepth is None or depth < maxdepth):
                    waiting.extend(
                        (e["name"].rstrip("/"), depth + 1)
                        for e in listing.entries
                        if e["type"] == "directory" and (descend is None or descend(e))
                    )
                # Start the next listings before yielding, so that they run
                # while the caller handles this one.
                start()
                yield listing
    finally:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)


def _listing(
    task: asyncio.Task[list[dict[str, Any]]], folder: str, depth: int, root: str
) -> Listing:
    """The listing a finished ``_ls`` task gives, or its error."""
    try:
        entries = task.result()
    except Exception as e:
        if folder == root:
            # Nothing has been done yet: fail as fsspec's own walk does.
            raise
        return Listing(folder, depth, [], e)
    if folder == root and any(
        e["name"].rstrip("/") == root and e["type"] != "directory" for e in entries
    ):
        # fsspec's _ls of a file lists the file itself.
        raise NotADirectoryError(root)
    # Some storages list a folder within itself, e.g. as its marker object.
    return Listing(
        folder, depth, [e for e in entries if e["name"].rstrip("/") != folder]
    )
