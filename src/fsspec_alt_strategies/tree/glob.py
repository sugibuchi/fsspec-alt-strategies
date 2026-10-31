# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A glob that lists only the folders its pattern can match in.

`tree_glob` is an alternative to fsspec's ``_glob``, with the same matches. fsspec's
finds every path under the pattern's literal root, as deep as the pattern goes,
then filters them with a regular expression. `tree_glob` walks the tree as `tree_walk`
does, and lists a folder only if the pattern can match in it.

To decide, a folder's path is matched with the pattern's segments, one by one.
A ``**`` matches any number of segments, so only those before the first ``**``
can rule a folder out. Without a ``**``, nor can a folder as deep as the
pattern, as nothing below it can match.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

import re
from contextlib import aclosing
from dataclasses import dataclass
from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec.utils import glob_translate

from ..control import Operation, State
from .walk import _listings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from fsspec.asyn import AsyncFileSystem

    from .walk import _Listing

__all__ = ["GlobState", "tree_glob"]


@dataclass(frozen=True, kw_only=True)
class GlobState(State):
    """The progress of a `tree_glob`.

    Attributes
    ----------
    folders_listed : int
        The folders listed so far, those that failed included.
    matched : int
        The paths matched so far.
    """

    folders_listed: int = 0
    matched: int = 0


def tree_glob(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    withdirs: bool = True,
    batch_size: int | None = None,
) -> Operation[dict[str, Any], GlobState]:
    """Find the paths matching a glob pattern, listing only folders it can match in.

    It matches what fsspec's ``_glob`` matches, with ``*``, ``?``, ``[...]`` and
    ``**``. The paths found are filtered with fsspec's own regular expression,
    so the folders pruned change only the requests, never the matches. A folder
    that fails to list yields its error, and the rest is still searched.

    It yields each match as its folder's listing arrives, so the matches are
    not in order. fsspec's ``_glob`` sorts them, by name. A storage listing a
    folder twice, e.g. as a prefix and as its marker object, yields it twice.

    It wins on storages that list one folder per request, when the pattern
    leaves folders out after its first wildcard, e.g. ``data/*/2024/*.parquet``.
    fsspec's ``_glob`` then lists every folder under ``data`` as deep as the
    pattern goes. It loses on object stores, which list a whole tree in one
    paginated request, and gains nothing over `tree_find` with patterns such as
    ``data/**/*.parquet``, which can match anywhere.

    Once cancelled, it lists no new folder, and yields the matches of the
    listings in flight.

    Requests: one ``_ls`` per folder the pattern can match in, at most
    ``batch_size`` at a time. One ``_info`` for a pattern without wildcards, or
    one matching its literal root folder itself, e.g. ``data/**``.

    Parameters
    ----------
    fs : fsspec.asyn.AsyncFileSystem
        The filesystem.
    path : str
        The pattern. A trailing "/" matches only folders.
    maxdepth : int, optional
        How deep a ``**`` goes, as with fsspec's ``_glob``. No limit by default.
    withdirs : bool
        Whether folders can match, as with fsspec's ``_glob``.
    batch_size : int, optional
        The most listings at a time. fsspec's default if None.

    Returns
    -------
    Operation
        Yielding the ``_ls`` entry of each path matched, or the error of a
        folder that failed to list. Its state is a `GlobState`.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1, or a ``**`` is not a whole segment.

    See Also
    --------
    tree_find : Every path in a tree.
    tree_walk : The walk under it.

    Examples
    --------
    >>> async with tree_glob(fs, "data/*/2024/*.parquet") as matches:
    ...     async for match in matches:
    ...         if isinstance(match, Exception):
    ...             raise match
    ...         print(match["name"])
    data/sales/2024/part-0.parquet
    """
    if maxdepth is not None and maxdepth < 1:
        raise ValueError("maxdepth must be at least 1")
    ends_with_sep = path.endswith("/")
    path = fs._strip_protocol(path)
    if not has_magic(path):
        return Operation(_info_of(fs, path), GlobState())
    search = _Search(fs, path, ends_with_sep, withdirs, maxdepth)

    async def run(
        op: Operation[dict[str, Any], GlobState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        root = search.root
        listings = _listings(
            fs,
            root,
            maxdepth=search.depth,
            descend=search.descend,
            batch_size=batch_size,
            stopping=op.cancelling,
        )
        async with aclosing(listings):
            try:
                first = await anext(listings)
            except (FileNotFoundError, NotADirectoryError):
                # As fsspec's _glob: a pattern under a missing folder matches nothing.
                return
            # fsspec's find includes the root folder itself, which few patterns match.
            if root and search.matched({"name": root, "type": "directory"}):
                op.state = GlobState(current=root, matched=1)
                yield await fs._info(root)
            for match in search.matches_in(first, op):
                yield match
            async for listing in listings:
                for match in search.matches_in(listing, op):
                    yield match

    return Operation(run, GlobState())


class _Prefix:
    """A pattern's segments before its first ``**``, which decide the folders to list."""

    def __init__(self, segments: list[str]) -> None:
        # Without a **, the pattern is as deep as its segments. With one, it can
        # match at any depth: only the segments before it can rule a folder out.
        self.bounded = "**" not in segments
        if not self.bounded:
            segments = segments[: segments.index("**")]
        self.segments = [re.compile(glob_translate(s)) for s in segments]

    def alive(self, relative: list[str]) -> bool:
        """Whether the pattern can still match below a folder, at that path."""
        if self.bounded and len(relative) >= len(self.segments):
            return False
        return all(s.match(n) for s, n in zip(self.segments, relative, strict=False))


def _info_of(
    fs: AsyncFileSystem, path: str
) -> Callable[
    [Operation[dict[str, Any], GlobState]], AsyncGenerator[dict[str, Any] | Exception]
]:
    """What a `tree_glob` runs for a pattern without wildcards: the path's ``_info``."""

    async def run(
        op: Operation[dict[str, Any], GlobState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        try:
            entry = await fs._info(path)
        except FileNotFoundError:
            return
        op.state = GlobState(current=path, matched=1)
        yield entry

    return run


class _Search:
    """One `tree_glob` call: the pattern, split and compiled."""

    def __init__(
        self,
        fs: AsyncFileSystem,
        path: str,
        ends_with_sep: bool,
        withdirs: bool,
        maxdepth: int | None,
    ) -> None:
        self.withdirs = withdirs
        # Split as fsspec does: the literal folder before the first wildcard,
        # and the pattern below it.
        first_magic = min(i for i in (path.find(c) for c in "*?[") if i >= 0)
        slash = path.rfind("/", 0, first_magic)
        relative = path[slash + 1 :]
        self.root = fs._strip_protocol(path[: slash + 1])
        self.start = len(self.root) + 1 if self.root else 0
        self.depth = _depth(path, relative, maxdepth)
        self.prefix = _Prefix(relative.split("/"))
        # The matches come from fsspec's own regular expression, and the prefix
        # only prunes. So pruning can never change them.
        regex = glob_translate(path + ("/" if ends_with_sep else ""))
        self.matches = re.compile(regex).match
        self.slash_dirs = ends_with_sep or path.endswith("/**")

    def matched(self, entry: dict[str, Any]) -> bool:
        is_dir = entry["type"] == "directory"
        if is_dir and not self.withdirs:
            return False
        name = entry["name"].rstrip("/")
        return (
            self.matches(name + "/" if is_dir and self.slash_dirs else name) is not None
        )

    def descend(self, entry: dict[str, Any]) -> bool:
        return self.prefix.alive(entry["name"].rstrip("/")[self.start :].split("/"))

    def matches_in(
        self, listing: _Listing, op: Operation[dict[str, Any], GlobState]
    ) -> list[dict[str, Any] | Exception]:
        """A listing's matches, or its error, counted in the operation's state."""
        state = op.state
        if listing.error is not None:
            found: list[dict[str, Any] | Exception] = [listing.error]
        else:
            found = [e for e in listing.entries if self.matched(e)]
        op.state = GlobState(
            errors=state.errors + (listing.error is not None),
            current=listing.folder,
            folders_listed=state.folders_listed + 1,
            matched=state.matched + len(found) - (listing.error is not None),
        )
        return found


def _depth(path: str, relative: str, maxdepth: int | None) -> int | None:
    """How deep to walk below the root, as fsspec's ``_glob`` computes it."""
    depth = relative.count("/") + 1
    if "**" not in path:
        return depth
    if maxdepth is None:
        return None
    return depth - path[path.find("**") :].count("/") - 1 + maxdepth
