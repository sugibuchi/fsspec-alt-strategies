# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""gcsfs's ``GCSFileSystem``, wrapped as a `FlatFileSystem`, listing by prefix.

The generic `AsyncFlatWrapper` lists a prefix by gcsfs's ``_find`` of its
directory, then keeps the names starting with the prefix. `GcsfsWrapper` gives
gcsfs's ``_find`` the prefix, which lists only the objects starting with it:
for the glob "data/2024-*", the objects of "data/2024-", not all of "data/".
gcsfs still gathers every page before answering, so the listing is held
whole, not streamed.

The rest is the generic one's, as gcsfs does it best already:

- its copies, as gcsfs's ``objects.rewrite`` asks nothing first;
- its deletes, as gcsfs's ``_rm`` sends batches of 100, GCS's most.

It imports nothing of gcsfs: it calls the wrapped filesystem's own methods.

This module depends on `flat`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .flat import AsyncFlatWrapper

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["GcsfsWrapper"]


class GcsfsWrapper(AsyncFlatWrapper):
    """A gcsfs filesystem as a `FlatFileSystem`, listing by prefix as gcsfs does.

    It takes what `AsyncFlatWrapper` takes. ``rm_batch_size`` is best a
    multiple of 100, the most calls gcsfs sends in one batch request.

    Examples
    --------
    >>> flat = GcsfsWrapper(
    ...     gcsfs.GCSFileSystem(),
    ...     rm_batch_size=1000,
    ...     operation_concurrency=32,
    ...     io_concurrency=8,
    ...     storage_profile=known_implementations["gcsfs.core.GCSFileSystem"]["storage_profile"],
    ... )
    """

    def _ils(
        self, path: str, recursive: bool = False, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        if recursive and path:
            return self._objects(path, prefix)
        else:
            # The buckets, or one directory: as the generic wrapper lists them.
            return super()._ils(path, recursive=recursive, prefix=prefix, **kwargs)

    async def _objects(self, path: str, prefix: str) -> AsyncIterator[dict[str, Any]]:
        """Every object starting with ``path/prefix``, in name order, listed by that prefix."""
        try:
            found = await self.fs._find(path, detail=True, prefix=prefix)
        except FileNotFoundError:
            return  # A missing bucket lists nothing, as a missing prefix does.
        for entry in sorted(found.values(), key=lambda e: e["name"]):
            yield entry
