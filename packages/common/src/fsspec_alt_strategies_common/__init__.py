# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The interface the strategies of fsspec-alt-strategies share with their callers.

The strategies are copied into a project. This package is not: it is installed
once, so that every copy in a project, of any version, shares one interface.

- `control`: `State`, an operation's progress, `OperationHandler`, its control,
  and the monitors watching the operations behind a filesystem's fsspec
  methods.
- `profile`: `StorageProfile`, what a storage does that fsspec leaves open.
- `states`: the states the strategies report, such as `FindState`.

Its monitors are one context variable for the whole project: a `monitoring`
block watches the calls of every copy.
"""

from __future__ import annotations

from .control import (
    Call,
    CallCancelled,
    Monitor,
    OperationHandler,
    State,
    monitoring,
)
from .profile import StorageProfile
from .states import CpState, DuState, FindState, GlobState, MvState, RmState, WalkState

try:
    from ._version import __version__
except ImportError:  # A source tree never built: no version file.
    __version__ = "0.0.0"

__all__ = [
    "Call",
    "CallCancelled",
    "CpState",
    "DuState",
    "FindState",
    "GlobState",
    "Monitor",
    "MvState",
    "OperationHandler",
    "RmState",
    "State",
    "StorageProfile",
    "WalkState",
    "__version__",
    "monitoring",
]
