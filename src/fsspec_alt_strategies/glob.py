# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A glob that lists only the folders its pattern can match in.

`tree_glob` is an alternative to fsspec's ``_glob``, with the same answers.
fsspec's finds every path under the pattern's literal root, as deep as the
pattern goes, then filters them with a regular expression. `tree_glob` walks
the tree with `walk`, and lists a folder only if the pattern can match in it.

To decide, a folder's path is matched with the pattern's segments, one by one.
A ``**`` matches any number of segments, so only those before the first ``**``
can rule a folder out. Without a ``**``, nor can a folder as deep as the
pattern, as nothing below it can match.

`itree_glob` is the same search as an async iterator, so that the matches can
be worked on as they arrive. `tree_glob` collects them.

This module depends on `walk`.
"""

from __future__ import annotations

import re
from contextlib import aclosing
from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec.utils import glob_translate

from .walk import walk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fsspec.asyn import AsyncFileSystem

    from .walk import Listing

__all__ = ["itree_glob", "tree_glob"]


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


async def tree_glob(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    withdirs: bool = True,
    batch_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[Exception]]:
    """Find the paths matching a glob pattern, listing only folders it can match in.

    It answers as fsspec's ``_glob`` does, with ``*``, ``?``, ``[...]`` and
    ``**``. The paths found are filtered with fsspec's own regular expression,
    so the folders pruned change only the requests, never the answer. A folder
    that fails to list adds its error, and the rest is still searched.

    It wins on storages that list one folder per request, when the pattern
    leaves folders out after its first wildcard, e.g. ``data/*/2024/*.parquet``.
    fsspec's ``_glob`` then lists every folder under ``data`` as deep as the
    pattern goes. It loses on object stores, which list a whole tree in one
    paginated request, and gains nothing over `tree_find` with patterns such as
    ``data/**/*.parquet``, which can match anywhere.

    Requests: one ``_ls`` per folder the pattern can match in, at most
    ``batch_size`` at a time. One ``_info`` for a pattern without wildcards, or
    one matching its literal root folder itself, e.g. ``data/**``.

    It takes no callback: `itree_glob` serves callers wanting to watch or stop
    the search.

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
    entries : list of dict
        The ``_ls`` entry of each path matched, sorted by name. Their names are
        what fsspec's ``_glob`` returns.
    errors : list of Exception
        The error of each folder that failed to list, in the order they failed.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1, or a ``**`` is not a whole segment.

    See Also
    --------
    itree_glob : The same search, as an async iterator.
    tree_find : Every path in a tree.

    Examples
    --------
    >>> entries, errors = await tree_glob(fs, "data/*/2024/*.parquet")
    >>> [e["name"] for e in entries], errors
    (['data/sales/2024/part-0.parquet'], [])
    """
    found: dict[str, dict[str, Any]] = {}
    errors: list[Exception] = []
    async for match in itree_glob(
        fs, path, maxdepth=maxdepth, withdirs=withdirs, batch_size=batch_size
    ):
        if isinstance(match, Exception):
            errors.append(match)
        else:
            # Keyed by name as fsspec's _glob keys them, so a path listed twice,
            # e.g. a folder and its marker object, is answered once.
            found[match["name"].rstrip("/")] = match
    return [found[name] for name in sorted(found)], errors


async def itree_glob(
    fs: AsyncFileSystem,
    path: str,
    *,
    maxdepth: int | None = None,
    withdirs: bool = True,
    batch_size: int | None = None,
) -> AsyncIterator[dict[str, Any] | Exception]:
    """Yield the paths matching a glob pattern, as `tree_glob` finds them.

    It searches as `tree_glob` does, with the same requests, and yields each
    match as its folder's listing arrives. So the matches are not in order. A
    folder that fails to list yields its error, and the rest is still searched.

    Like `walk`, it takes no callback and returns no result: the caller drives
    it. Breaking out of the loop stops it. Close it with `contextlib.aclosing`,
    so that its listings in flight are cancelled at once.

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

    Yields
    ------
    dict or Exception
        The ``_ls`` entry of each path matched, or the error of a folder that
        failed to list.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1, or a ``**`` is not a whole segment.

    See Also
    --------
    tree_glob : The same search, collected and sorted.
    walk : The walk under it.

    Examples
    --------
    >>> from contextlib import aclosing
    >>> async with aclosing(itree_glob(fs, "data/*/2024/*.parquet")) as matches:
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
        try:
            entry = await fs._info(path)
        except FileNotFoundError:
            return
        yield entry
        return
    search = _Search(fs, path, ends_with_sep, withdirs, maxdepth)
    root = search.root
    listings = walk(
        fs, root, maxdepth=search.depth, descend=search.descend, batch_size=batch_size
    )
    async with aclosing(listings):
        try:
            first = await anext(listings)
        except (FileNotFoundError, NotADirectoryError):
            # As fsspec's _glob: a pattern under a missing folder matches nothing.
            return
        # fsspec's find includes the root folder itself, which few patterns match.
        if root and search.matched({"name": root, "type": "directory"}):
            yield await fs._info(root)
        for match in search.matches_in(first):
            yield match
        async for listing in listings:
            for match in search.matches_in(listing):
                yield match


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
        # The answer comes from fsspec's own regular expression, and the prefix
        # only prunes. So pruning can never change the answer.
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

    def matches_in(self, listing: Listing) -> list[dict[str, Any] | Exception]:
        if listing.error is not None:
            return [listing.error]
        return [e for e in listing.entries if self.matched(e)]


def _depth(path: str, relative: str, maxdepth: int | None) -> int | None:
    """How deep to walk below the root, as fsspec's ``_glob`` computes it."""
    depth = relative.count("/") + 1
    if "**" not in path:
        return depth
    if maxdepth is None:
        return None
    return depth - path[path.find("**") :].count("/") - 1 + maxdepth
