# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The states of the operations several families have, such as a find's.

An operation of each family, e.g. the ``find`` of a ``TreeFileSystem`` and of
a ``FlatFileSystem``, reports its progress with one state class, which a
monitor reads. A field that only some of them can count is None in the
others: e.g. ``folders_listed`` of a ``FlatFileSystem``'s ``find``, which lists
keys, not folders.

This module depends on `monitor`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .monitor import State

__all__ = [
    "CpState",
    "DuState",
    "FindState",
    "GlobState",
    "MvState",
    "RmState",
    "WalkState",
]


@dataclass(frozen=True, kw_only=True)
class WalkState(State):
    """The progress of a walk, such as a ``TreeFileSystem``'s or a
    ``FlatFileSystem``'s.

    Attributes
    ----------
    folders_listed : int or None
        The folders listed so far, those that failed included. None for a
        ``FlatFileSystem``, which lists no folders.
    found : int
        The entries found so far.
    """

    folders_listed: int | None = None
    found: int = 0


@dataclass(frozen=True, kw_only=True)
class FindState(State):
    """The progress of a find, such as a ``TreeFileSystem``'s or a
    ``FlatFileSystem``'s.

    Attributes
    ----------
    folders_listed : int or None
        The folders listed so far, those that failed included. None for a
        ``FlatFileSystem``, which lists no folders.
    found : int
        The paths found so far.
    """

    folders_listed: int | None = None
    found: int = 0


@dataclass(frozen=True, kw_only=True)
class DuState(State):
    """The progress of a du, such as a ``TreeFileSystem``'s or a
    ``FlatFileSystem``'s.

    Attributes
    ----------
    folders_listed : int or None
        The folders listed so far, those that failed included. None for a
        ``FlatFileSystem``, which lists no folders.
    files : int
        The files counted so far.
    size : int
        Their size so far, in bytes. Once the operation has run to its end, it
        is what fsspec's ``_du`` returns.
    """

    folders_listed: int | None = None
    files: int = 0
    size: int = 0


@dataclass(frozen=True, kw_only=True)
class GlobState(State):
    """The progress of a glob, such as a ``TreeFileSystem``'s or a
    ``FlatFileSystem``'s.

    Attributes
    ----------
    folders_listed : int or None
        The folders listed so far, those that failed included. None for a
        ``FlatFileSystem``, which lists no folders.
    matched : int
        The paths matched so far.
    """

    folders_listed: int | None = None
    matched: int = 0


@dataclass(frozen=True, kw_only=True)
class RmState(State):
    """The progress of a recursive delete, such as a ``TreeFileSystem``'s or a
    ``FlatFileSystem``'s.

    Attributes
    ----------
    files_found : int
        The files found so far. It grows as the walk or the listing goes on.
    files_deleted : int
        The files deleted so far.
    folders_deleted : int or None
        The folders deleted so far. None for a ``FlatFileSystem``,
        which deletes no folders.
    """

    files_found: int = 0
    files_deleted: int = 0
    folders_deleted: int | None = None


@dataclass(frozen=True, kw_only=True)
class CpState(State):
    """The progress of a recursive copy, such as a ``TreeFileSystem``'s or a
    ``FlatFileSystem``'s.

    Attributes
    ----------
    files_found : int
        The files found so far. It grows as the walk or the listing goes on.
    files_copied : int
        The files copied so far.
    """

    files_found: int = 0
    files_copied: int = 0


@dataclass(frozen=True, kw_only=True)
class MvState(State):
    """The progress of a recursive move, such as a ``TreeFileSystem``'s or a
    ``FlatFileSystem``'s.

    Attributes
    ----------
    files_found : int
        The files found so far. It grows as the walk or the listing goes on.
    files_moved : int
        The files moved so far: copied, and their source deleted.
    folders_moved : int or None
        The source folders deleted so far, once all in them was moved. None
        for a ``FlatFileSystem``, which deletes no folders.
    roots_renamed : int or None
        The folders moved whole, each by one ``_mv_file``, what is in them
        neither found nor counted. None unless the storage moves folders so.
    """

    files_found: int = 0
    files_moved: int = 0
    folders_moved: int | None = None
    roots_renamed: int | None = None
