# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The handler of a storage with a hierarchical namespace, as the tree strategies take it.

`TreeHandler` adds to `StorageHandler` a recursive listing, optional, and the
requests on a directory whole, each optional and declared by the profile.

This module depends on `handler` of `common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from ..common.handler import StorageHandler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..common.handler import Entry, StoragePath

__all__: list[str] = []


class TreeHandler(StorageHandler):
    """A storage's primitives, as the tree strategies call them.

    Beside `StorageHandler`'s, a subclass may implement:

    - ``ils_recursive``, all below a directory in one listing, declaring its
      order in ``recursive_order``. Without it, the strategies list a tree a
      directory per request, by ``ils``, several at a time;
    - ``rm_tree``, a storage's recursive delete, declared by the profile's
      ``deletes_directories=True``;
    - ``mv_tree``, a storage's rename of a directory, declared by
      ``renames_directories=True``;
    - ``cp_tree``, a storage's copy of a directory, declared by
      ``copies_directories=True``.

    Each takes a file too, as a root may be either, and a missing root raises
    FileNotFoundError.
    """

    recursive_order: ClassVar[Literal["depth first", "name"] | None] = None
    """The order of ``ils_recursive``'s entries, or None where the handler has none.

    - "depth first": a directory's entry, then all below it, before any other
      entry. Its entries among themselves may come in any order.
    - "name": sorted by the whole name, as object stores list keys. A
      directory's entry may then come apart from what is below it, as "a/b-x"
      comes between "a/b" and "a/b/c".
    """

    def ils_recursive(self, entry: Entry, **kwargs: Any) -> AsyncIterator[Entry]:
        """List all below a directory, at any depth, in ``recursive_order``.

        A directory may be left out, as object stores list only files: the
        strategies tell it from the paths below it. A file lists itself. A
        missing directory raises FileNotFoundError, when the iteration starts.
        """
        raise NotImplementedError(f"{type(self).__name__} lists no tree at once")

    async def rm_tree(self, entry: Entry) -> None:
        """Delete a directory with everything in it, in one request."""
        raise NotImplementedError(f"{type(self).__name__} deletes no directory whole")

    async def mv_tree(self, entry: Entry, path2: StoragePath) -> None:
        """Move a directory with everything in it to ``path2``, in one request.

        The strategies call it only with ``path2`` missing and its parent
        existing, having made it.
        """
        raise NotImplementedError(f"{type(self).__name__} moves no directory whole")

    async def cp_tree(self, entry: Entry, path2: StoragePath) -> None:
        """Copy a directory with everything in it to ``path2``, in one request, as `mv_tree` moves one."""
        raise NotImplementedError(f"{type(self).__name__} copies no directory whole")
