# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Alternative, filesystem-independent strategies for fsspec.

fsspec's defaults assume a flat key space, as object stores have. The strategies
here are written for filesystems with a hierarchical namespace too, and work on
any fsspec filesystem, as functions or as bases of a filesystem class.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # A source tree never built, or a copy of one: no version file.
    __version__ = "0.0.0"

__all__ = ["__version__"]
