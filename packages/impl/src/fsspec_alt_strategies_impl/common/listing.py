# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Recursive listings as the strategies take them: a tree, depth first, every directory in it.

`implied_directories` is for a handler's ``ils_recursive`` on a storage
listing keys, such as an object store: it makes up the directories that the
keys imply, and turns markers into their directories' entries. A key listing
in name order then gives a tree in preorder, as ``ils_recursive`` must.

`ils_by_directory` lists a tree by one ``ils`` per directory: for a caller
asking all below a directory of a handler with no ``ils_recursive``, and for
a handler's ``ils_recursive`` falling back on it, e.g. where its storage
refuses its own listing.

This module depends on `handler`.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable

    from .handler import Entry, StorageHandler

__all__ = ["ils_by_directory", "implied_directories"]


async def implied_directories(
    keys: AsyncIterable[Entry], top: str
) -> AsyncGenerator[Entry]:
    """Keys listed in name order, as a tree: each directory they imply before its first key, each once.

    A storage listing keys, e.g. S3's ListObjectsV2 with no delimiter, lists
    files alone, sorted by their whole names. All keys below a directory then
    come together, as they share its name and "/". So making up each
    directory before the first key below it gives a tree in preorder, as
    ``ils_recursive`` must list it.

    A key ending with "/" is a directory's marker: it is yielded as the
    directory's entry, its name without the "/", with ``"marker": True``, so
    that a deletion knows an object is there to delete. So is an entry of type
    "directory", as a hierarchical namespace lists one. Named without the
    "/", it may come apart from what is below it, as "a" comes before
    "a.txt", then "a/b": it is held until the first path below it comes, or
    else yielded empty before its parent's last path. A directory made up
    has the entry fsspec's object stores give one: its ``name``, ``size`` 0
    and ``type``. A name listed twice, e.g. across pages, is yielded once.

    Parameters
    ----------
    keys : AsyncIterable of dict
        The entries of the keys, in name order, a "/" between segments.
    top : str
        The directory all of them are below, which is not yielded, its own
        marker neither: "" for the storage's root. For a listing narrowed by
        a prefix, e.g. "data/2024-", it is the prefix's directory, "data".

    Yields
    ------
    dict
        The entries of the directories and the files, depth first.

    Examples
    --------
    >>> def ils_recursive(self, entry, prefix=False):
    ...     top = entry["name"].rpartition("/")[0] if prefix else entry["name"]
    ...     return implied_directories(self._list_keys(entry["name"]), top)
    """
    tree = _Tree(top)
    last: str | None = None
    async for key in keys:
        if key["name"] != last:  # Else listed again, e.g. on the next page.
            last = key["name"]
            for entry in tree.take(key):
                yield entry
    for entry in tree.leave(""):
        yield entry
    for entry in tree.release(""):
        yield entry


class _Tree:
    """The directories `implied_directories` has open, and those it holds, as the keys come."""

    def __init__(self, top: str) -> None:
        self.top = top.rstrip("/")
        self.above = f"{self.top}/" if self.top else ""
        self.opened: list[str] = []  # The directories yielded holding the last key.
        # The directories listed, but not yet entered: "a" comes before "a.txt",
        # what is below it after. Each is entered as the first path below it
        # comes, or else yielded empty before its parent is left.
        self.held: dict[str, Entry] = {}
        self.done: set[str] = set()  # The directories yielded, opened or not.

    def take(self, key: Entry) -> list[Entry]:
        """The entries a key listed gives: the directories it opens, then itself, unless held."""
        name = key["name"]
        is_dir = name.endswith("/") or key.get("type") == "directory"
        trimmed = name.rstrip("/")
        if trimmed == self.top:
            return [] if is_dir else [key]  # The top's own marker, or a file itself.
        if not trimmed.startswith(self.above):
            return []  # Nothing below the top.
        out = self.leave(trimmed)
        parts = trimmed[len(self.above) :].split("/")
        for depth in range(len(self.opened), len(parts) - 1):
            directory = self.above + "/".join(parts[: depth + 1])
            self.opened.append(directory)
            entry = self.held.pop(directory, None)
            if directory not in self.done:
                self.done.add(directory)
                out.append(entry or {"name": directory, "size": 0, "type": "directory"})
        if not is_dir:
            out.append(key)
        elif trimmed not in self.done and trimmed not in self.held:
            entry = {**key, "name": trimmed, "type": "directory"}
            if name.endswith("/"):
                entry["marker"] = True  # An object of its own, which rmdir deletes.
            self.held[trimmed] = entry
        return out

    def leave(self, name: str) -> list[Entry]:
        """Leave the directories open not holding ``name``: the empty ones held in them."""
        empty: list[Entry] = []
        while self.opened and not name.startswith(f"{self.opened[-1]}/"):
            empty += self.release(f"{self.opened.pop()}/")
        return empty

    def release(self, start: str) -> list[Entry]:
        """The directories held starting with ``start``, empty: no longer held."""
        names = sorted(n for n in self.held if n.startswith(start))
        self.done.update(names)
        return [self.held.pop(n) for n in names]


async def ils_by_directory(
    handler: StorageHandler, entry: Entry, **ils_kwargs: Any
) -> AsyncGenerator[Entry]:
    """List all below a directory depth first, by one ``ils`` per directory, in turn.

    It yields what ``ils_recursive`` yields: each entry below ``entry``, at
    any depth, depth first in preorder. A directory's entries come in the
    order its ``ils`` gives them, each subdirectory's followed by all below
    it. A file lists itself, as ``ils`` lists it. A directory listed twice,
    e.g. as a prefix and as its marker, is yielded twice, but listed once.

    The strategies never call it: on a handler without ``recursive_ls``,
    they list a directory per request themselves, many at once.

    Parameters
    ----------
    handler : StorageHandler
        The handler whose ``ils`` lists each directory.
    entry : dict
        The directory's entry, as a listing gave it or ``handler.entry``
        made it.
    **ils_kwargs
        Given to each ``ils``.

    Yields
    ------
    dict
        The entries below ``entry``, depth first.

    Raises
    ------
    Exception
        The error of a directory that fails to list, as the storage raised
        it, which ends the listing.

    Examples
    --------
    >>> def ils_recursive(self, entry, prefix=False):
    ...     return ils_by_directory(self, entry)  # Where the storage refuses its own.
    """
    directory = entry["name"].rstrip("/")
    listed: set[str] = set()
    async for found in handler.ils(entry, **ils_kwargs):
        name = found["name"].rstrip("/")
        if name == directory:
            # A file lists itself. A directory may list its own marker: it is
            # no entry below it.
            if found["type"] != "directory":
                yield found
            continue
        yield found
        if found["type"] == "directory" and name not in listed:
            listed.add(name)
            subtree = ils_by_directory(handler, {**found, "name": name}, **ils_kwargs)
            async with aclosing(subtree):  # Closed with this one.
                async for below in subtree:
                    yield below
