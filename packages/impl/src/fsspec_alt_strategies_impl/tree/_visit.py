# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What a walk visits: the directories an operation needs, as the walk finds them.

A walk's plan has two sides. Which directories to visit is the operation's,
as its arguments say: a ``maxdepth``, a glob's pattern, a ``descend``. It
decides what the operation answers. How each is listed, one directory per
request or a whole subtree by one, is the storage's, and decides only what the
listing costs.

A `_Visit` is the operation's side. The walk asks it of each root, then of each
subdirectory a listing finds, whether to leave it, list it alone, or take all
below it. Only all below a directory may be listed by one recursive listing.

This module depends on `handler` and `paths` of `common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ..common.paths import _check_maxdepth

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from ..common.handler import Entry

__all__: list[str] = []

_Need = Literal["skip", "directory", "subtree"]
"""What a walk does with a directory: leave it, list it alone, or take all below it."""


class _Visit:
    """Every directory, all below each root: what find, du, rm, cp and mv need without ``maxdepth``.

    Subclasses narrow it. Each method is asked once per directory, as the walk
    finds it, so it never holds a directory's listing.
    """

    def depth(self, root: str) -> int:
        """The depth of a root of the walk: 1, as the directory walked from."""
        return 1

    def root(self, entry: Entry, depth: int) -> Literal["directory", "subtree"]:
        """Whether a root is listed alone, or with all below it: a root is never left."""
        return "subtree"

    def enter(self, entry: dict[str, Any], depth: int) -> _Need:
        """What becomes of a subdirectory a listing found, by its entry as listed, at ``depth``."""
        return "subtree"


class _Pruned(_Visit):
    """The directories within ``maxdepth`` that ``descend`` enters, as fsspec's walk and glob visit them.

    - ``maxdepth`` lists no directory deeper than it, as fsspec's walk lists
      none. A directory one deeper is found, by its parent's listing, but
      left.
    - ``descend``, called with the entry of each subdirectory found, leaves it
      unless it returns True. A root is always listed.
    - ``whole`` tells of a directory whether ``descend`` holds for all below
      it, e.g. below a glob's ``**``: all below it is then taken, and
      ``descend`` asked no more. Without ``descend``, all below each root is.
      With ``maxdepth``, never: what is deeper is left.
    - ``matched`` are the roots a glob matched, one level down: depth 2, as
      fsspec's ``_expand_path`` walks them.
    """

    def __init__(
        self,
        *,
        maxdepth: int | None = None,
        descend: Callable[[dict[str, Any]], bool] | None = None,
        whole: Callable[[dict[str, Any]], bool] | None = None,
        matched: Collection[str] = (),
    ) -> None:
        _check_maxdepth(maxdepth)
        self.maxdepth = maxdepth
        self.descend = descend
        self.whole = whole
        self.matched = matched

    def depth(self, root: str) -> int:
        return 2 if root in self.matched else 1

    def root(self, entry: Entry, depth: int) -> Literal["directory", "subtree"]:
        return "subtree" if self._all_below(entry) else "directory"

    def enter(self, entry: dict[str, Any], depth: int) -> _Need:
        if self._left(entry, depth):
            return "skip"
        elif self._all_below(entry):
            return "subtree"
        else:
            return "directory"

    def _left(self, entry: dict[str, Any], depth: int) -> bool:
        """Whether a subdirectory is left: deeper than ``maxdepth``, or one ``descend`` refuses."""
        too_deep = self.maxdepth is not None and depth > self.maxdepth
        return too_deep or (self.descend is not None and not self.descend(entry))

    def _all_below(self, entry: dict[str, Any]) -> bool:
        """Whether all below a directory is visited: nothing deeper is left, nor pruned."""
        if self.maxdepth is not None:
            return False
        elif self.descend is None:
            return True
        else:
            # By its name alone, as a pattern tells of a path.
            name = entry["name"].rstrip("/")
            return self.whole is not None and self.whole(
                {"name": name, "type": "directory"}
            )


class _TopDown(_Visit):
    """Another visit, each directory listed alone: a walk top down gives a directory before those below it.

    A recursive listing passes a directory only after all below it, so taking
    all below one would hold the whole subtree.
    """

    def __init__(self, visit: _Visit) -> None:
        self.visit = visit

    def depth(self, root: str) -> int:
        return self.visit.depth(root)

    def root(self, entry: Entry, depth: int) -> Literal["directory", "subtree"]:
        return "directory"

    def enter(self, entry: dict[str, Any], depth: int) -> _Need:
        need = self.visit.enter(entry, depth)
        return "directory" if need == "subtree" else need
