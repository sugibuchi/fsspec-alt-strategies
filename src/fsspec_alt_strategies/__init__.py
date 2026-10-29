# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Alternative, filesystem-independent strategies for fsspec.

fsspec's defaults assume a flat key space, as object stores have. The strategies
here are written for filesystems with a hierarchical namespace too, and work on
any fsspec filesystem, as functions or as bases of a filesystem class.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fsspec-alt-strategies")
except PackageNotFoundError:  # Not installed, e.g. run from a source tree.
    __version__ = "0.0.0"

__all__ = ["__version__"]
