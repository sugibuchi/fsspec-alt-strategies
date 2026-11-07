# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The states of the operations several families have, such as a find's.

A family's strategy and its counterpart in another, e.g. `tree_find` and
`flat_find`, report their progress with one state class. A field that only
some of them can count is None in the others: e.g. ``folders_listed`` of a
`flat_find`, which lists keys, not folders. Each strategy's module exports its
state class, so that it is imported from there.

This module depends on `control`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..control import State

__all__ = ["DuState", "FindState", "GlobState"]


@dataclass(frozen=True, kw_only=True)
class FindState(State):
    """The progress of a find, such as `tree_find` or `flat_find`.

    Attributes
    ----------
    folders_listed : int or None
        The folders listed so far, those that failed included. None for a
        strategy listing no folders, such as `flat_find`.
    found : int
        The paths found so far.
    """

    folders_listed: int | None = None
    found: int = 0


@dataclass(frozen=True, kw_only=True)
class DuState(State):
    """The progress of a du, such as `tree_du` or `flat_du`.

    Attributes
    ----------
    folders_listed : int or None
        The folders listed so far, those that failed included. None for a
        strategy listing no folders, such as `flat_du`.
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
    """The progress of a glob, such as `tree_glob` or `flat_glob`.

    Attributes
    ----------
    folders_listed : int or None
        The folders listed so far, those that failed included. None for a
        strategy listing no folders, such as `flat_glob`.
    matched : int
        The paths matched so far.
    """

    folders_listed: int | None = None
    matched: int = 0
