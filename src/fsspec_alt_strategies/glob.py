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

This module depends on `control` and `walk`.
"""

from __future__ import annotations

import re
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec.utils import glob_translate

from .control import Result, State, _finish, _report
from .walk import walk

if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem

    from .control import Callback

__all__ = ["GlobResult", "GlobState", "tree_glob"]


@dataclass(frozen=True, kw_only=True)
class GlobState(State):
    """The state of a `tree_glob`, as its callback gets it.

    Attributes
    ----------
    folders_listed : int
        The folders listed so far.
    folders_pruned : int
        The folders found but not listed, as the pattern cannot match in them.
    found : int
        The paths matched so far.
    """

    folders_listed: int = 0
    folders_pruned: int = 0
    found: int = 0


@dataclass(kw_only=True)
class GlobResult(Result):
    """The paths a `tree_glob` matched.

    Attributes
    ----------
    entries : dict of str to dict
        Each path matched, sorted, with its entry from ``_ls``. ``list(entries)``
        is what fsspec's ``_glob`` returns, and ``entries`` is what it returns
        with ``detail=True``.
    folders_pruned : int
        The folders found but not listed.
    """

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    folders_pruned: int = 0


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
    callback: Callback[GlobState] | None = None,
) -> GlobResult:
    """Find the paths matching a glob pattern, listing only folders it can match in.

    It answers as fsspec's ``_glob`` does, with ``*``, ``?``, ``[...]`` and
    ``**``. The paths found are filtered with fsspec's own regular expression,
    so the folders pruned change only the requests, never the answer. A folder
    that fails to list is in ``errors``, and the rest is still searched.

    It wins on storages that list one folder per request, when the pattern
    leaves folders out after its first wildcard, e.g. ``data/*/2024/*.parquet``.
    fsspec's ``_glob`` then lists every folder under ``data`` as deep as the
    pattern goes. It loses on object stores, which list a whole tree in one
    paginated request, and gains nothing over `tree_find` with patterns such as
    ``data/**/*.parquet``, which can match anywhere.

    Requests: one ``_ls`` per folder the pattern can match in, at most
    ``batch_size`` at a time. One ``_info`` for a pattern without wildcards, or
    one matching its literal root folder itself, e.g. ``data/**``.

    The callback is called after each folder listed. A stop returns the paths
    matched so far. ``Stop(cleanup=True)`` undoes nothing, as nothing is changed.

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
    callback : callable, optional
        Called with a `GlobState` after each step. It may return a `Stop`.

    Returns
    -------
    GlobResult
        The paths matched, and the folders that failed.

    Raises
    ------
    ValueError
        If ``maxdepth`` is less than 1, or a ``**`` is not a whole segment.
    OperationStopped
        If the callback returned ``Stop(raise_error=True)``.

    See Also
    --------
    walk : The walk under it.
    tree_find : Every path in a tree.

    Examples
    --------
    >>> result = await tree_glob(fs, "data/*/2024/*.parquet", batch_size=32)
    >>> list(result.entries), result.folders_pruned
    (['data/sales/2024/part-0.parquet'], 41)
    """
    if maxdepth is not None and maxdepth < 1:
        raise ValueError("maxdepth must be at least 1")
    ends_with_sep = path.endswith("/")
    path = fs._strip_protocol(path)
    result = GlobResult()
    if not has_magic(path):
        with suppress(FileNotFoundError):
            result.entries[path] = await fs._info(path)
        return result
    search = _Search(fs, path, ends_with_sep, withdirs, maxdepth, result)
    try:
        await search.run(batch_size, callback)
    except (FileNotFoundError, NotADirectoryError):
        if search.listed:
            raise
        # As fsspec's _glob: a pattern under a missing folder matches nothing.
        return result
    root = search.root
    # fsspec's find includes the root folder itself, which few patterns match.
    if withdirs and root and search.matched({"name": root, "type": "directory"}):
        result.entries[root] = await fs._info(root)
    result.entries = dict(sorted(result.entries.items()))
    return _finish(result)


class _Search:
    """One `tree_glob` call: the pattern, split and compiled, and its walk."""

    def __init__(
        self,
        fs: AsyncFileSystem,
        path: str,
        ends_with_sep: bool,
        withdirs: bool,
        maxdepth: int | None,
        result: GlobResult,
    ) -> None:
        self.fs = fs
        self.withdirs = withdirs
        self.result = result
        self.listed = 0
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
        if self.prefix.alive(entry["name"].rstrip("/")[self.start :].split("/")):
            return True
        self.result.folders_pruned += 1
        return False

    async def run(
        self, batch_size: int | None, callback: Callback[GlobState] | None
    ) -> None:
        result = self.result
        listings = walk(
            self.fs,
            self.root,
            maxdepth=self.depth,
            descend=self.descend,
            batch_size=batch_size,
        )
        async with aclosing(listings):
            async for listing in listings:
                self.listed += 1
                if listing.error is not None:
                    result.errors[listing.folder] = listing.error
                result.entries.update(
                    (e["name"].rstrip("/"), e)
                    for e in listing.entries
                    if self.matched(e)
                )
                state = GlobState(
                    errors=len(result.errors),
                    current=listing.folder,
                    folders_listed=self.listed,
                    folders_pruned=result.folders_pruned,
                    found=len(result.entries),
                )
                if (stop := await _report(callback, state)) is not None:
                    result.stop = stop
                    return


def _depth(path: str, relative: str, maxdepth: int | None) -> int | None:
    """How deep to walk below the root, as fsspec's ``_glob`` computes it."""
    depth = relative.count("/") + 1
    if "**" not in path:
        return depth
    if maxdepth is None:
        return None
    return depth - path[path.find("**") :].count("/") - 1 + maxdepth
