# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The filesystem on a handler wrapping another filesystem, serving that one's paths.

`FileSystemWrapper` is a `BaseFileSystem` on a handler of `tree` or `flat`,
e.g. `FsspecTreeHandler`, which wraps an existing fsspec filesystem. Its
recursive methods are the handler's strategies, its paths the wrapped one's.

This module depends on `_base`, and `filesystem` of `common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..common.filesystem import BaseFileSystem
from ._base import _unwrapped

if TYPE_CHECKING:
    import asyncio

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
    asynchronous, loop, use_listings_cache, listings_expiry_time, max_paths
        fsspec's own options, as `BaseFileSystem` takes them.

    Examples
    --------
    >>> fs = FileSystemWrapper(FsspecTreeHandler(fsspec.filesystem("file"), ...))
    >>> fs.rm("data/2024", recursive=True)
    """

    cachable = False
    """Not cached: fsspec would key it by its handler's repr, which two wrappers of one filesystem share."""

    _handler: _FsspecHandler  # type: ignore[assignment]

    def __init__(
        self,
        handler: FsspecTreeHandler | FsspecFlatHandler,
        *,
        asynchronous: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        use_listings_cache: bool = True,
        listings_expiry_time: float | None = None,
        max_paths: int | None = None,
    ) -> None:
        super().__init__(
            handler,
            asynchronous=asynchronous,
            loop=loop,
            use_listings_cache=use_listings_cache,
            listings_expiry_time=listings_expiry_time,
            max_paths=max_paths,
        )

        # The wrapped filesystem's paths, which its handler holds.
        self.protocol = handler.protocol
        self.root_marker = handler.root
        self._strip_protocol = handler.strip_protocol

    def _unwrapped(self) -> AbstractFileSystem:
        """The filesystem wrapped, as a monitor's `Call` names it: the one its user made."""
        return _unwrapped(self._handler.fs)
