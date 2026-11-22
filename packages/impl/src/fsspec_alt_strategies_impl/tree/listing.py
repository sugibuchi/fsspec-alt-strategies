# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A recursive listing made of one listing per directory.

A `TreeFileOperationHandler`'s recursive ``ils`` lists all below a directory,
depth first in preorder, which covers name order, where the profile's
``recursive_ls_depth_first_preorder`` declares it, else in any order. A
storage with a recursive listing of its own implements it so. One without
has `_tree_ils`, its default, which lists a directory per request by ``ils``.
One whose own may be refused, as only asking tells, falls back on it.

The tree strategies list a directory per request themselves, many at once,
unless the profile declares the order: so `_tree_ils` serves the other
callers, e.g. a filesystem's recursive listing, one directory at a time.

This module depends on `handler` and `paths` of `common`.
"""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from ..common.paths import _trim

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ..common.handler import Entry
    from .handler import TreeFileOperationHandler

__all__: list[str] = []


async def _tree_ils(
    handler: TreeFileOperationHandler, entry: Entry, **ils_kwargs: Any
) -> AsyncGenerator[dict[str, Any]]:
    """List all below a directory depth first, by one ``ils`` per directory, in turn.

    It yields what a recursive ``ils`` yields: each entry below ``entry``, at
    any depth, depth first in preorder. A directory's entries come in the
    order its ``ils`` gives them, each subdirectory's followed by all below
    it. A file lists itself, as ``ils`` lists it. A directory listed twice,
    e.g. as a prefix and as its marker object, is yielded twice, but listed
    once.

    A directory below ``entry`` that fails to list raises its error, as the
    storage raised it, and ends the listing: a recursive listing cannot go on
    past a failure. A missing ``entry`` raises FileNotFoundError, as its own
    ``ils`` raises it.

    ``entry`` is the directory's entry, as its parent's listing gave it, or as
    the handler's ``entry`` makes it from a path: it is listed by it whole, so
    that a storage addressing items by id asks nothing again. ``ils_kwargs`` go
    to each ``ils``. The caller holds the permits, as for any ``ils``.
    """
    directory = _trim(entry["name"])
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
            subtree = _tree_ils(handler, {**found, "name": name}, **ils_kwargs)
            async with aclosing(subtree):  # Closed with this one.
                async for below in subtree:
                    yield below
