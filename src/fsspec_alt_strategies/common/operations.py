# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The storage's operations the strategies take, as types.

A strategy takes the operations it uses as functions, such as ``ls``, rather
than a filesystem: e.g. a filesystem's ``_ls``. It calls each as fsspec calls
its own, and takes paths as those functions do, without a protocol.

This module depends on nothing else in the package. Its types exist only for
type checkers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import TypeAlias

    # For the strategies' signatures: none here takes one.
    _Ls: TypeAlias = Callable[..., Awaitable[list[dict[str, Any]]]]  # noqa: PYI047
    """Lists a folder, called as ``ls(path, detail=True)``, as fsspec's ``_ls``."""

    _Info: TypeAlias = Callable[[str], Awaitable[dict[str, Any]]]  # noqa: PYI047
    """Describes a path, called as ``info(path)``, as fsspec's ``_info``."""
