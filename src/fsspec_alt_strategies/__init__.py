# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""Alternative, filesystem-independent strategies for fsspec.

fsspec's defaults assume a flat key space, as object stores have. The strategies
here are written for filesystems with a hierarchical namespace too, and work on
any fsspec filesystem, as functions or as bases of a filesystem class.

It re-exports the interface every strategy shares: `Operation` and `State`,
the monitors of `control`, and `StorageProfile`. Both modules depend on
nothing else, and the strategies use them, so a copy keeps them all the same.
The strategies, the filesystem classes and the files are imported from their
modules, so that a copy may leave out those it does not use.
"""

from __future__ import annotations

from .control import Call, CallCancelled, Monitor, Operation, State, monitoring
from .profile import StorageProfile

try:
    from ._version import __version__
except ImportError:  # A source tree never built, or a copy of one: no version file.
    __version__ = "0.0.0"

__all__ = [
    "Call",
    "CallCancelled",
    "Monitor",
    "Operation",
    "State",
    "StorageProfile",
    "__version__",
    "monitoring",
]
