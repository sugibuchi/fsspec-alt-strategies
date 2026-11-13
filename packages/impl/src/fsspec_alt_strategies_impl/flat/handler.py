# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The handler of a flat storage, such as an object store, as the flat strategies take it.

`FlatHandler` adds to `StorageHandler` the recursive listing by prefix, which
the flat strategies list with, and the deletion of keys in batches.

This module depends on `handler` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ..common.handler import StorageHandler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from ..common.handler import Entry

__all__ = ["FlatHandler"]


class FlatHandler(StorageHandler):
    """A storage's primitives, as the flat strategies call them.

    Beside `StorageHandler`'s, a subclass implements ``ils_recursive``, and
    ``rm_files``, which deletes keys in batches of ``rm_batch_size``.
    """

    family = "flat"

    rm_batch_size: int = 1000
    """The most keys `rm_files` takes at once, e.g. 1000 as S3's DeleteObjects takes."""

    @abstractmethod
    async def rm_files(self, entries: list[Entry]) -> Mapping[str, Exception] | None:
        """Delete keys, by one request, at most ``rm_batch_size`` of them.

        It returns the keys it could not delete, each with its error, and
        raises only if the request failed whole. A batch may span the storage's
        partitions, e.g. buckets: a storage whose request takes one partition
        sends one per partition.
        """

    @abstractmethod
    def ils_recursive(
        self, entry: Entry, *, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[Entry]:
        """List the keys below a directory whose names below it start with ``prefix``, in name order.

        It yields the keys' entries, sorted by the whole name, a page at a
        time. A directory's own marker, if the storage keeps one, comes as an
        entry whose name ends with "/".
        """
