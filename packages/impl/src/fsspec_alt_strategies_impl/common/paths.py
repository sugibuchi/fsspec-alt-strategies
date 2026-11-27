# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Paths and depths, as the strategies take them, and directories as fsspec's walk gives them.

`fixed_prefix` is for a handler's ``verify_path``, which sees glob patterns
too: it gives the part of a pattern before its first wildcard.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["fixed_prefix"]

_WILDCARD = re.compile(r"[*?[]")
"""A character of a glob pattern, as fsspec's ``has_magic`` finds them."""


def fixed_prefix(path: str) -> str:
    """The part of a path before its first segment holding a wildcard: the path itself without one.

    A handler's ``verify_path`` sees a glob pattern as its caller gave it,
    e.g. ``abfs://data/2026-*/x.csv``. One whose names refuse ``*``, ``?`` or
    ``[`` verifies the pattern's fixed prefix instead, e.g. ``abfs://data``,
    the directory the glob starts from: the names it matches below are the
    storage's own, listed. A URL's protocol, and what it names beyond the
    path, stay in the prefix, so that the handler still checks them.

    Parameters
    ----------
    path : str
        A path or URL, maybe a glob pattern.

    Returns
    -------
    str
        The prefix, without a trailing "/" unless it is a root, e.g. "/" or
        ``abfs://``. A pattern whose first segment holds a wildcard gives
        that root, or "" for a relative path.

    Examples
    --------
    >>> fixed_prefix("abfs://data/2026-*/x.csv")
    'abfs://data'
    >>> fixed_prefix("abfs://data-*/x.csv")
    'abfs://'
    >>> fixed_prefix("/home/me/*.txt")
    '/home/me'
    >>> fixed_prefix("data/x.csv")
    'data/x.csv'
    """
    wildcard = _WILDCARD.search(path)
    if wildcard is None:
        return path
    head = path[: wildcard.start()]
    prefix = head[: head.rfind("/") + 1]  # Up to the last "/" before it, or "".
    trimmed = prefix.rstrip("/")
    if trimmed.endswith(":") or not trimmed:
        # A root, which keeps its "/": a URL's, e.g. "file:///", or "/".
        return prefix
    else:
        return trimmed


def _check_maxdepth(maxdepth: int | None) -> None:
    """Raise ValueError for a ``maxdepth`` below 1, as fsspec's walk does."""
    if maxdepth is not None and maxdepth < 1:
        raise ValueError(f"maxdepth must be at least 1: {maxdepth}")


def _triples(
    directory: str, entries: list[dict[str, Any]], detail: bool
) -> tuple[str, Any, Any]:
    """A directory's ``(path, dirs, files)``, as fsspec's ``walk`` yields them."""
    dirs: dict[str, Any] = {}
    files: dict[str, Any] = {}
    for entry in entries:
        name = entry["name"].rstrip("/").rsplit("/", 1)[-1]
        (dirs if entry["type"] == "directory" else files)[name] = entry

    if detail:
        return directory, dirs, files
    else:
        return directory, list(dirs), list(files)
