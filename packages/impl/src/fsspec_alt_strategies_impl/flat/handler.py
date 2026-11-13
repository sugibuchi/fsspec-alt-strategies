# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The handler of a flat storage, such as an object store, as the flat strategies take it.

`FlatHandler` adds to `StorageHandler` the recursive listing by prefix, which
the flat strategies list with.

This module depends on `handler` of `common`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ..common.handler import StorageHandler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..common.handler import Entry

__all__: list[str] = []


class FlatHandler(StorageHandler):
    """A storage's primitives, as the flat strategies call them.

    Beside `StorageHandler`'s, a subclass implements ``ils_recursive``. Its
    profile's ``rm_files``, where declared, deletes in batches of
    ``rm_batch_size``.
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
