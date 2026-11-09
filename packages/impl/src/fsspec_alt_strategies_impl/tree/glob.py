# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A glob that lists only the folders its pattern can match in.

`_tree_glob` is an alternative to fsspec's ``_glob``, with the same matches. fsspec's
finds every path under the pattern's literal root, as deep as the pattern goes,
then filters them with a regular expression. `_tree_glob` walks the tree as `_tree_walk`
does, and lists a folder only if the pattern can match in it.

To decide, a folder's path is matched with the pattern's segments, one by one.
A ``**`` matches any number of segments, so only those before the first ``**``
can rule a folder out. Without a ``**``, nor can a folder as deep as the
pattern, as nothing below it can match.

Several patterns are matched in one walk per tree: a folder is listed once if
any of them can match in it, and each path is matched with all of them.

This module depends on `find`, and `operation`, `paths` and `patterns` of
`common`.
"""

from __future__ import annotations

from contextlib import aclosing
from glob import has_magic
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import GlobState

from ..common.operation import Operation
from ..common.paths import _check_maxdepth, _trim
from ..common.patterns import _infos, _inside, _Search
from .find import _find_listings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from .walk import _Listing

__all__: list[str] = []


def _tree_glob(
    fs: Any,
    path: str | Sequence[str],
    *,
    maxdepth: int | None = None,
    withdirs: bool = True,
    ils_kwargs: dict[str, Any] | None = None,
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
    over `_tree_find` with patterns such as ``data/**/*.parquet``, which can
    match anywhere.

    Once cancelled, it lists no new folder, and yields the matches of the
    listings in flight.

    Requests: one ``_ils`` per folder a pattern can match in, as the filesystem's ``operation_limit`` allows. One ``info`` per pattern without wildcards,
    and per literal root folder walked that a pattern matches, e.g. that of
    ``data/**``.

    ``path`` is a pattern, or patterns, of paths on ``fs``, as its ``_ils``
    takes them: a trailing "/" matches only folders. ``maxdepth`` is how deep
    a ``**`` goes, and ``withdirs`` whether folders can match, as with
    fsspec's ``_glob``. ``ils_kwargs`` go to each ``_ils``, as fsspec's
    ``glob`` passes its own on.

    The operation yields the ``_ils`` entry of each path matched, or the error
    of a folder that failed to list, and its state is a `GlobState`. The call
    raises ValueError if ``maxdepth`` is less than 1, or a ``**`` is not a
    whole segment.
    """
    _check_maxdepth(maxdepth)
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
        infos = _infos(fs, literals, op, found)
        async for entry in infos:
            yield entry

        for root, group in _walks(searches):
            if op.cancelling():
                return
            walk = _walk(fs, root, group, op, found, ils_kwargs)
            async for match in walk:
                yield match

    return Operation(run, GlobState(folders_listed=0))


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


async def _walk(
    fs: Any,
    root: str,
    searches: list[_Search],
    op: Operation[dict[str, Any], GlobState],
    found: set[str],
    ils_kwargs: dict[str, Any] | None,
) -> AsyncGenerator[dict[str, Any] | Exception]:
    """The matches of patterns in one tree, walked from ``root``."""
    depths = [s.depth_from(root) for s in searches]

    # As fsspec's _glob: a pattern under a missing folder, or a file, matches
    # nothing. Its find includes the root folder itself, which few patterns
    # match.
    steps = _find_listings(
        fs,
        root,
        maxdepth=None if None in depths else max(d for d in depths if d is not None),
        descend=lambda e: any(s.descend(e) for s in searches),
        ils_kwargs=ils_kwargs,
        stopping=op.cancelling,
        folder_root=bool(root)
        and any(s.matched({"name": root, "type": "directory"}) for s in searches),
    )

    async with aclosing(steps):
        async for listing in steps:
            for match in _matches_in(listing, searches, op, found):
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
        # Depth 0 is the root's own entry: no folder listed.
        folders_listed=state.folders_listed + (listing.depth > 0 and listing.last),
        matched=state.matched + len(matches) - (listing.error is not None),
    )

    return matches
