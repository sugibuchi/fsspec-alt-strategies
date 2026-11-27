# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A glob that lists only the directories its pattern can match in.

`_tree_glob` is an alternative to fsspec's ``_glob``, with the same matches.
fsspec's finds every path under the pattern's literal root, as deep as the
pattern goes, then filters them with a regular expression. `_tree_glob` walks
the tree as `_tree_walk` does, and lists a directory only if the pattern can
match in it.

To decide, a directory's path is matched with the pattern's segments, one by
one. A ``**`` matches any number of segments, so only those before the first
``**`` can rule a directory out. Without a ``**``, a directory as deep as the
pattern is ruled out too, as nothing below it can match.

Below a ``**``, nothing is pruned. So a directory below which a pattern prunes
nothing, e.g. ``data`` for ``data/**/*.csv``, is listed by one recursive
``ils``.

Several patterns are matched in one walk per tree: a directory is listed once if
any of them can match in it, and each path is matched with all of them.

This module depends on `find` and `walk`, and `operation`, `paths` and `patterns` of
`common`.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import GlobState

from ..common.operation import Operation
from ..common.paths import _check_maxdepth
from ..common.patterns import _inside, _Search
from .find import _Found, _read
from .walk import _MAX_RUNNING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from ..common.handler import Entry, StoragePath
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


def _tree_glob(
    handler: TreeFileOperationHandler,
    path: StoragePath | Sequence[StoragePath],
    *,
    maxdepth: int | None = None,
    withdirs: bool = True,
    ils_kwargs: dict[str, Any] | None = None,
    max_running: int = _MAX_RUNNING,
) -> Operation[dict[str, Any], GlobState]:
    """Find the paths matching glob patterns, listing only directories they can match in.

    It matches what fsspec's ``_glob`` matches, with ``*``, ``?``, ``[...]`` and
    ``**``. The paths found are filtered with fsspec's own regular expression,
    so the directories pruned change only the requests, never the matches. A
    directory that fails to list yields its error, and the rest is still
    searched.

    It yields each match as its directory's listing arrives, so the matches are
    not in order. fsspec's ``_glob`` sorts them, by name. A storage listing a
    directory twice, e.g. as a prefix and as its marker object, yields it once.

    Several patterns match what fsspec's globs of each match together, each
    path once. They are matched in one walk per tree, rather than one per
    pattern: a directory that several can match in is listed once. Their trees
    are walked one after another. A pattern under another's literal root is
    walked from that root, which may list the directories between them.

    It wins on storages that list one directory per request, when the pattern
    leaves directories out after its first wildcard, e.g.
    ``data/*/2024/*.parquet``. fsspec's ``_glob`` then lists every directory
    under ``data`` as deep as the pattern goes. Several patterns win again where they share directories, e.g.
    ``data/*/2024/*.csv`` and ``data/*/2024/*.json``. It loses on object
    stores, which list a whole tree in one paginated request, and gains nothing
    over `_tree_find` with patterns such as ``data/**/*.parquet``, which can
    match anywhere.

    It prunes a level at a time, so it lists every directory a wildcard
    matches, each by a request, at once. A pattern selective deep in a small
    tree costs more requests than one recursive listing of that tree would,
    though no more rounds: ``t007/year=2024/*/part-0.parquet`` lists
    ``year=2024``, then each of its 12 months, where adlfs lists all below
    ``year=2024`` once, then filters. Measured on Azure, that was 13 requests
    against 2, and 0.09 s against 0.06 s. The tree's size is not known, so
    it never lists a subtree whole by itself: listing all below the
    directories of ``*/year=2024/month=03`` that way took 3.4 s against
    0.16 s. A caller who knows the subtree is small writes ``**`` there, e.g.
    ``t007/year=2024/**/part-0.parquet``, which lists it whole.

    Once cancelled, it lists no new directory, and yields the matches of the
    listings in flight.

    Where the profile declares ``recursive_ls_with_prefix``, as an object
    store lists keys, a tree is listed by one ``ils_recursive``, narrowed to
    the patterns' literal prefix: ``data/2024-*/*.csv`` lists all starting
    with ``data/2024-``, in pages, and prunes as they stream.

    Requests: as above, or else one ``ils`` per directory a pattern can match
    in, as the handler's ``operation_limit`` allows. Without ``maxdepth``, a
    directory at the pattern's first ``**`` is listed by one
    ``ils_recursive``, with all below it, where the profile declares
    ``recursive_ls``: ``data/*/**/*.csv`` lists ``data``, then each
    directory in it.
    One ``info`` per literal root walked that a pattern matches, e.g. that of
    ``data/**``. A path without wildcards is such a root, its ``info`` alone,
    unless it is below another pattern's root: that pattern's listing then
    finds it.

    ``path`` is a pattern, or patterns, of paths on ``handler``, as its ``ils``
    takes them: a trailing "/" matches only directories. ``maxdepth`` is how
    deep a ``**`` goes, and ``withdirs`` whether directories can match, as with
    fsspec's ``_glob``. ``ils_kwargs`` go to each ``ils``, as fsspec's
    ``glob`` passes its own on.

    The operation yields the ``ils`` entry of each path matched, or the error
    of a directory that failed to list, and its state is a `GlobState`. The call
    raises ValueError if ``maxdepth`` is less than 1, or a ``**`` is not a
    whole segment.
    """
    _check_maxdepth(maxdepth)
    searches = [
        _Search(handler.trim(pattern), pattern.endswith("/"), withdirs, maxdepth)
        for pattern in ([path] if isinstance(path, str) else path)
    ]

    async def run(
        op: Operation[dict[str, Any], GlobState],
    ) -> AsyncGenerator[dict[str, Any] | Exception]:
        walks = _walks(searches)
        alone = [(root, group) for root, group in walks if _alone(root, group)]
        infos = _infos(handler, alone, op)
        async with aclosing(infos):
            async for item in infos:
                yield item

        for root, group in walks:
            if op.cancelling():
                return
            if _alone(root, group):
                continue
            walk = _walk(handler, root, group, op, ils_kwargs, max_running)
            async with aclosing(walk):
                async for match in walk:
                    yield match

    return Operation(run, GlobState(directories_listed=0))


def _alone(root: str, searches: list[_Search]) -> bool:
    """Whether no pattern can match below a root, e.g. a path without wildcards alone: its ``info`` tells it all."""
    entry: Entry = {"name": root, "type": "directory"}
    return bool(root) and not any(s.descend(entry) for s in searches)


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


def _shared_start(literals: list[str]) -> str:
    """The longest string all of `literals` start with."""
    first, last = min(literals), max(literals)
    size = next(
        (i for i, (a, b) in enumerate(zip(first, last, strict=False)) if a != b),
        len(first),
    )
    return first[:size]


def _walk(
    handler: TreeFileOperationHandler,
    root: str,
    searches: list[_Search],
    op: Operation[dict[str, Any], GlobState],
    ils_kwargs: dict[str, Any] | None,
    max_running: int,
) -> AsyncGenerator[dict[str, Any] | Exception]:
    """The matches of patterns in one tree, traversed from ``root``, the patterns pruning it.

    As fsspec's ``_glob``: a pattern under a missing directory, or a file,
    matches nothing. Its find includes the root itself, which few patterns
    match: a path without wildcards does, whether a file or a directory.
    """

    def matched(state: GlobState, entry: Entry) -> GlobState:
        return replace(state, matched=state.matched + 1)

    return _read(
        handler,
        root,
        op,
        lambda emit, root_entry: _Found(handler, emit, op, root_entry, withdirs=True),
        count=matched,
        max_running=max_running,
        patterns=searches,
        # What every match starts with, e.g. "data/2024-" for "data/2024-*/*.csv".
        prefix=_shared_start([s.literal for s in searches]),
        ils_kwargs=ils_kwargs,
        file_root=any(s.matched({"name": root, "type": "file"}) for s in searches),
        dir_root=bool(root)
        and any(s.matched({"name": root, "type": "directory"}) for s in searches),
    )


async def _infos(
    handler: TreeFileOperationHandler,
    roots: list[tuple[str, list[_Search]]],
    op: Operation[dict[str, Any], GlobState],
) -> AsyncGenerator[dict[str, Any] | Exception]:
    """The ``info`` of each root alone that a pattern matches, all sent at once, as they arrive.

    Each holds a permit of the handler's ``operation_limit``: a list of plain
    paths may give many. A missing root matches nothing, as with fsspec's
    ``_glob``. Once the operation is cancelled, none starts.
    """

    async def info(
        root: str, searches: list[_Search]
    ) -> tuple[str, list[_Search], Entry | Exception | None]:
        async with handler.operation_limit:
            if op.cancelling():
                return root, searches, None
            try:
                return root, searches, await handler.info(handler.entry(root))
            except FileNotFoundError:
                return root, searches, None
            except Exception as error:
                return root, searches, error

    tasks = [asyncio.ensure_future(info(root, group)) for root, group in roots]
    try:
        for next_done in asyncio.as_completed(tasks):
            root, searches, found = await next_done
            if isinstance(found, Exception):
                op.state = replace(op.state, current=root, errors=op.state.errors + 1)
                yield found
            elif found is not None and any(s.matched(found) for s in searches):
                op.state = replace(op.state, current=root, matched=op.state.matched + 1)
                yield found
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
