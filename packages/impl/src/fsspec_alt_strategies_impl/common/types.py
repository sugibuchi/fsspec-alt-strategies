# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The requests a strategy takes from its caller, as types.

A strategy takes the filesystems it works on, and calls their primitives
itself. A copy or move takes one request too, as a hook: how a file goes to
the destination, e.g. a router's copy between two storages.

This module depends on nothing else in the package. Its types exist only for
type checkers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import TypeAlias

    _Request: TypeAlias = Callable[..., Awaitable[Any]]  # noqa: PYI047
    """A request changing the storage, e.g. fsspec's ``_cp_file``."""
