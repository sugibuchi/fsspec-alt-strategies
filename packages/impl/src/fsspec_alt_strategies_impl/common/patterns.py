# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Glob patterns, matched as fsspec's ``_glob`` matches them.

`_tree_glob` matches with `_Search`, as fsspec's `_glob` matches. A `_Search` also tells which directories the pattern can still
match in, as `_tree_glob` prunes by directory.

To decide, a directory's path is matched with the pattern's segments, one by one.
A ``**`` matches any number of segments, so only those before the first ``**``
can rule a directory out. Without a ``**``, a directory as deep as the pattern
is ruled out too, as nothing below it can match.

A path without wildcards is a pattern too, matching only itself: its root, at
depth 0, below which nothing is listed.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import re
from typing import Any

from fsspec.utils import glob_translate


def _inside(path: str, directory: str) -> bool:
    """Whether ``path`` is below ``directory``, "" being the top of the storage."""
    return directory == "" or path.startswith(directory + "/")


class _Prefix:
    """A pattern's segments before its first ``**``, which decide the directories to list."""

    def __init__(self, segments: list[str]) -> None:
        # Without a **, the pattern is as deep as its segments. With one, it can
        # match at any depth: only the segments before it can rule a directory out.
        self.bounded = "**" not in segments
        if not self.bounded:
            segments = segments[: segments.index("**")]
        self.segments = [re.compile(glob_translate(s)) for s in segments]

    def everything(self, relative: list[str]) -> bool:
        """Whether the pattern prunes nothing below a directory, at that path."""
        return (
            not self.bounded
            and len(relative) >= len(self.segments)
            and self.alive(relative)
        )

    def alive(self, relative: list[str]) -> bool:
        """Whether the pattern can still match below a directory, at that path."""
        if self.bounded and len(relative) >= len(self.segments):
            return False
        else:
            return all(
                s.match(n) for s, n in zip(self.segments, relative, strict=False)
            )


class _Search:
    """One glob pattern, split and compiled.

    It answers as a walk from its own literal root would: it matches only
    below that root, or the root itself, as deep as the pattern goes. So a walk
    from a directory above serves it too.
    """

    def __init__(
        self,
        path: str,
        ends_with_sep: bool,
        withdirs: bool,
        maxdepth: int | None,
    ) -> None:
        self.withdirs = withdirs

        magic = [i for i in (path.find(c) for c in "*?[") if i >= 0]
        # A path without wildcards: as with fsspec's _glob, it matches itself,
        # a file or a directory, whatever withdirs says.
        self.plain = not magic
        if magic:
            # Split as fsspec does: the literal directory before the first
            # wildcard, and the pattern below it.
            first_magic = min(magic)
            slash = path.rfind("/", 0, first_magic)
            relative = path[slash + 1 :]
            # The directory before the slash, or the root "/" itself.
            self.root = path[:slash] if slash > 0 else path[: slash + 1]
            self.depth = _depth(path, relative, maxdepth)
            self.prefix = _Prefix(relative.split("/"))
        else:
            # A path without wildcards matches only itself: its root, at depth
            # 0, so that nothing below it is listed.
            first_magic = len(path)
            self.root = path
            self.depth = 0
            self.prefix = _Prefix([])
        # What every path matched starts with: the pattern up to its first wildcard.
        self.literal = path[:first_magic]
        self.start = len(self.root) + 1 if self.root else 0

        # The matches come from fsspec's own regular expression, and the prefix
        # only prunes. So pruning can never change them.
        regex = glob_translate(path + ("/" if ends_with_sep else ""))
        self.matches = re.compile(regex).match
        self.slash_dirs = ends_with_sep or path.endswith("/**")

    def _level(self, name: str) -> int | None:
        """How far below the root a path is, 0 for the root, or None if not below."""
        if name == self.root:
            return 0
        if not _inside(name, self.root):
            return None
        else:
            return name[self.start :].count("/") + 1

    def matched(self, entry: dict[str, Any]) -> bool:
        is_dir = entry["type"] == "directory"
        if is_dir and not self.withdirs and not self.plain:
            return False

        name = entry["name"].rstrip("/")
        level = self._level(name)
        if (
            level is None
            or (level == 0 and not is_dir and not self.plain)
            or (self.depth is not None and level > self.depth)
        ):
            return False
        else:
            return (
                self.matches(name + "/" if is_dir and self.slash_dirs else name)
                is not None
            )

    def descend(self, entry: dict[str, Any]) -> bool:
        name = entry["name"].rstrip("/")
        if self.root.startswith(name + "/"):
            return True  # On the way down to the root.

        level = self._level(name)
        if level is None or (self.depth is not None and level >= self.depth):
            return False
        else:
            return level == 0 or self.prefix.alive(name[self.start :].split("/"))

    def whole(self, entry: dict[str, Any]) -> bool:
        """Whether every directory below a directory's entry is one to descend into.

        So it is at or below the pattern's first ``**``, when no depth limits
        it: a recursive listing of the directory lists what a walk would.
        """
        name = entry["name"].rstrip("/")
        level = self._level(name)
        if level is None or self.depth is not None:
            return False
        else:
            return self.prefix.everything(
                name[self.start :].split("/") if level else []
            )


def _depth(path: str, relative: str, maxdepth: int | None) -> int | None:
    """How deep to walk below the root, as fsspec's ``_glob`` computes it."""
    depth = relative.count("/") + 1
    if "**" not in path:
        return depth
    if maxdepth is None:
        return None
    else:
        return depth - path[path.find("**") :].count("/") - 1 + maxdepth
