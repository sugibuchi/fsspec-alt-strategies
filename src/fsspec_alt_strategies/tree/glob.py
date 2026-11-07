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

Several patterns are matched in one walk per tree: a folder is listed once if
any of them can match in it, and each path is matched with all of them.

This module depends on `control` and `walk`.
"""

from __future__ import annotations

import re
from contextlib import aclosing
from dataclasses import dataclass, replace
from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec.utils import glob_translate

from ..control import Operation, State
from .walk import _scan, _trim

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator, Sequence

    from .walk import _Info, _Listing, _Ls

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
    path: str | Sequence[str],
    *,
    ls: _Ls,
    info: _Info,
    maxdepth: int | None = None,
    withdirs: bool = True,
    concurrency_limit: asyncio.Semaphore,
) -> Operation[dict[str, Any], GlobState]:
    """Find the paths matching glob patterns, listing only folders they can match in.

    It matches what fsspec's ``_glob`` matches, with ``*``, ``?``, ``[...]`` and
    ``**``. The paths found are filtered with fsspec's own regular expression,
    so the folders pruned change only the requests, never the matches. A folder
    that fails to list yields its error, and the rest is still searched.

    It yields each match as its folder's listing arrives, so the matches are
    not in order. fsspec's ``_glob`` sorts them, by name. A storage listing a
    folder twice, e.g. as a prefix and as its marker object, yields it twice.

    Several patterns match what fsspec's globs of each match together, each
    path once. They are matched in one walk per tree, rather than one per
    pattern: a folder that several can match in is listed once. Their trees
    are walked one after another. A pattern under another's literal root is
    walked from that root, which may list the folders between them.

    It wins on storages that list one folder per request, when the pattern
    leaves folders out after its first wildcard, e.g. ``data/*/2024/*.parquet``.
    fsspec's ``_glob`` then lists every folder under ``data`` as deep as the
    pattern goes. Several patterns win again where they share folders, e.g.
    ``data/*/2024/*.csv`` and ``data/*/2024/*.json``. It loses on object
    stores, which list a whole tree in one paginated request, and gains nothing
    over `tree_find` with patterns such as ``data/**/*.parquet``, which can
    match anywhere.

    Once cancelled, it lists no new folder, and yields the matches of the
    listings in flight.

    Requests: one ``ls`` per folder a pattern can match in, as
    ``concurrency_limit`` allows. One ``info`` per pattern without wildcards,
    and per literal root folder walked that a pattern matches, e.g. that of
    ``data/**``.

    Parameters
    ----------
    path : str or sequence of str
        The pattern, or patterns, of paths as ``ls`` takes them. A trailing
        "/" matches only folders.
    ls : callable
        Lists a folder, called as ``ls(path, detail=True)``: e.g. the
        filesystem's ``_ls``.
    info : callable
        Describes a path, called as ``info(path)``: e.g. the filesystem's
        ``_info``.
    maxdepth : int, optional
        How deep a ``**`` goes, as with fsspec's ``_glob``. No limit by default.
    withdirs : bool
        Whether folders can match, as with fsspec's ``_glob``.
    concurrency_limit : asyncio.Semaphore
        Held by each request while it runs. Share one to limit several
        operations, or files, together.

    Returns
    -------
    Operation
        Yielding the ``ls`` entry of each path matched, or the error of a
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
    >>> limit = asyncio.Semaphore(32)  # Shared, e.g. by a filesystem's operations.
    >>> glob = tree_glob(
    ...     "data/*/2024/*.parquet", ls=fs._ls, info=fs._info, concurrency_limit=limit
    ... )
    >>> async with glob as matches:
    ...     async for match in matches:
    ...         if isinstance(match, Exception):
    ...             raise match
    ...         print(match["name"])
    data/sales/2024/part-0.parquet
    """
    if maxdepth is not None and maxdepth < 1:
        raise ValueError("maxdepth must be at least 1")
    literals: list[str] = []
    searches: list[_Search] = []
    for pattern in [path] if isinstance(path, str) else path:
        trimmed = _trim(pattern)
        if has_magic(trimmed):
            searches.append(_Search(trimmed, pattern.endswith("/"), withdirs, maxdepth))
        else:
            literals.append(trimmed)

    async def run(
        op: Operation[dict[str, Any], GlobState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        found: set[str] = set()
        async for entry in _infos(info, literals, concurrency_limit, op, found):
            yield entry
        for root, group in _walks(searches):
            if op.cancelling():
                return
            walk = _walk(ls, info, root, group, concurrency_limit, op, found)
            async for match in walk:
                yield match

    return Operation(run, GlobState())


async def _infos(
    info: _Info,
    paths: list[str],
    concurrency_limit: asyncio.Semaphore,
    op: Operation[dict[str, Any], GlobState],
    found: set[str],
) -> AsyncGenerator[dict[str, Any]]:
    """The ``info`` of each pattern without wildcards that exists, noted in ``found``.

    Each ``info`` holds a permit of ``concurrency_limit`` while it runs.
    """
    for path in dict.fromkeys(paths):
        if op.cancelling():
            return
        try:
            async with concurrency_limit:
                entry = await info(path)
        except FileNotFoundError:
            continue
        found.add(entry["name"].rstrip("/"))
        op.state = replace(op.state, current=path, matched=op.state.matched + 1)
        yield entry


def _walks(searches: list[_Search]) -> list[tuple[str, list[_Search]]]:
    """The roots to walk, each with the patterns it serves.

    A pattern whose literal root is inside another's is walked from that root:
    their trees are one.
    """
    roots = sorted({s.root for s in searches})
    tops = [r for r in roots if not any(_inside(r, o) for o in roots if o != r)]
    return [
        (top, [s for s in searches if s.root == top or _inside(s.root, top)])
        for top in tops
    ]


def _inside(path: str, folder: str) -> bool:
    """Whether ``path`` is below ``folder``, "" being the top of the storage."""
    return folder == "" or path.startswith(folder + "/")


async def _walk(
    ls: _Ls,
    info: _Info,
    root: str,
    searches: list[_Search],
    concurrency_limit: asyncio.Semaphore,
    op: Operation[dict[str, Any], GlobState],
    found: set[str],
) -> AsyncGenerator[dict[str, Any] | Exception]:
    """The matches of patterns in one tree, walked from ``root``."""
    depths = [s.depth_from(root) for s in searches]
    # As fsspec's _glob: a pattern under a missing folder, or a file, matches
    # nothing. Its find includes the root folder itself, which few patterns
    # match.
    steps = _scan(
        ls,
        info,
        root,
        maxdepth=None if None in depths else max(d for d in depths if d is not None),
        descend=lambda e: any(s.descend(e) for s in searches),
        concurrency_limit=concurrency_limit,
        stopping=op.cancelling,
        folder_root=bool(root)
        and any(s.matched({"name": root, "type": "directory"}) for s in searches),
    )
    async with aclosing(steps):
        async for step in steps:
            if isinstance(step, dict):
                # The root's own entry, as it matches.
                if step["name"].rstrip("/") not in found:
                    op.state = replace(op.state, matched=op.state.matched + 1)
                    yield step
                continue
            for match in _matches_in(step, searches, op, found):
                yield match


def _matches_in(
    listing: _Listing,
    searches: list[_Search],
    op: Operation[dict[str, Any], GlobState],
    found: set[str],
) -> list[dict[str, Any] | Exception]:
    """A listing's matches, or its error, counted in the operation's state."""
    state = op.state
    if listing.error is not None:
        matches: list[dict[str, Any] | Exception] = [listing.error]
    else:
        matches = [
            e
            for e in listing.entries
            if any(s.matched(e) for s in searches)
            # A match of a pattern without wildcards, found already.
            and e["name"].rstrip("/") not in found
        ]
    op.state = GlobState(
        errors=state.errors + (listing.error is not None),
        current=listing.folder,
        folders_listed=state.folders_listed + 1,
        matched=state.matched + len(matches) - (listing.error is not None),
    )
    return matches


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


class _Search:
    """One pattern of a `tree_glob`, split and compiled.

    It answers as a walk from its own literal root would: it matches only
    below that root, or the root itself, as deep as the pattern goes. So a walk
    from a folder above serves it too.
    """

    def __init__(
        self,
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
        self.root = _trim(path[: slash + 1])
        self.start = len(self.root) + 1 if self.root else 0
        self.depth = _depth(path, relative, maxdepth)
        self.prefix = _Prefix(relative.split("/"))
        # The matches come from fsspec's own regular expression, and the prefix
        # only prunes. So pruning can never change them.
        regex = glob_translate(path + ("/" if ends_with_sep else ""))
        self.matches = re.compile(regex).match
        self.slash_dirs = ends_with_sep or path.endswith("/**")

    def depth_from(self, folder: str) -> int | None:
        """How deep to walk from ``folder``, at or above the root, for this pattern."""
        if self.depth is None:
            return None
        between = self.root.count("/") + 1 if self.root else 0
        above = folder.count("/") + 1 if folder else 0
        return self.depth + between - above

    def _level(self, name: str) -> int | None:
        """How far below the root a path is, 0 for the root, or None if not below."""
        if name == self.root:
            return 0
        if not _inside(name, self.root):
            return None
        return name[self.start :].count("/") + 1

    def matched(self, entry: dict[str, Any]) -> bool:
        is_dir = entry["type"] == "directory"
        if is_dir and not self.withdirs:
            return False
        name = entry["name"].rstrip("/")
        level = self._level(name)
        if (
            level is None
            or (level == 0 and not is_dir)
            or (self.depth is not None and level > self.depth)
        ):
            return False
        return (
            self.matches(name + "/" if is_dir and self.slash_dirs else name) is not None
        )

    def descend(self, entry: dict[str, Any]) -> bool:
        name = entry["name"].rstrip("/")
        if self.root.startswith(name + "/"):
            return True  # On the way down to the root.
        level = self._level(name)
        if level is None or (self.depth is not None and level >= self.depth):
            return False
        return level == 0 or self.prefix.alive(name[self.start :].split("/"))


def _depth(path: str, relative: str, maxdepth: int | None) -> int | None:
    """How deep to walk below the root, as fsspec's ``_glob`` computes it."""
    depth = relative.count("/") + 1
    if "**" not in path:
        return depth
    if maxdepth is None:
        return None
    return depth - path[path.find("**") :].count("/") - 1 + maxdepth
