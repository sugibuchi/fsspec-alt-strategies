# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The filesystem on a handler wrapping another filesystem, serving that one's paths.

`FileSystemWrapper` is a `BaseFileSystem` on a handler of `tree` or `flat`,
e.g. `FsspecTreeHandler`, which wraps an existing fsspec filesystem. Its
recursive methods are the handler's strategies, its paths the wrapped one's.

This module depends on `_base`, and `filesystem` of `common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..common.filesystem import BaseFileSystem
from ._base import _unwrapped

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

    from ._base import _FsspecHandler
    from .flat import FsspecFlatHandler
    from .tree import FsspecTreeHandler

__all__ = ["FileSystemWrapper"]


class FileSystemWrapper(BaseFileSystem):
    """A `BaseFileSystem` on a handler wrapping another filesystem, serving that one's paths.

    As it is made, it takes the handler's `protocol`, `root` and
    `strip_protocol`, the wrapped filesystem's: it takes the URLs that one
    takes, and gives its own. A storage's own class names its paths instead.
    Its recursive methods are the handler's strategies: the tree strategies
    on an `FsspecTreeHandler`, and the flat ones on an `FsspecFlatHandler`.

    It is not cached: each is new, and may be dropped once used.

    Parameters
    ----------
    handler : FsspecTreeHandler or FsspecFlatHandler
        The handler of the filesystem wrapped.
    **kwargs
        fsspec's own options, as `BaseFileSystem` takes them, e.g. ``asynchronous=True``.

    Examples
    --------
    >>> fs = FileSystemWrapper(FsspecTreeHandler(fsspec.filesystem("file"), ...))
    >>> fs.rm("data/2024", recursive=True)
    """

    cachable = False
    """Not cached: fsspec would key it by its handler's repr, which two wrappers of one filesystem share."""

    _handler: _FsspecHandler  # type: ignore[assignment]

    def __init__(
        self, handler: FsspecTreeHandler | FsspecFlatHandler, **kwargs: Any
    ) -> None:
        super().__init__(handler, **kwargs)

        # The wrapped filesystem's paths, which its handler holds.
        self.protocol = handler.protocol
        self.root_marker = handler.root
        self._strip_protocol = handler.strip_protocol

    def _unwrapped(self) -> AbstractFileSystem:
        """The filesystem wrapped, as a monitor's `Call` names it: the one its user made."""
        return _unwrapped(self._handler.fs)
