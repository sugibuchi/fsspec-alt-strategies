# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Paths and depths, as the strategies take them.

This module depends on nothing else in the package.
"""

from __future__ import annotations


def _trim(path: str) -> str:
    """A path without its trailing "/", as fsspec's ``_strip_protocol`` leaves it.

    A storage's root stays as it is, e.g. "/" on a local filesystem.
    """
    return path.rstrip("/") or path


def _check_maxdepth(maxdepth: int | None) -> None:
    if maxdepth is not None and maxdepth < 1:
        raise ValueError(f"maxdepth must be at least 1: {maxdepth}")
