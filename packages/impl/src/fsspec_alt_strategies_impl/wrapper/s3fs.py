# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""s3fs's ``S3FileSystem``, wrapped as a `FlatHandler`, with its own listing and copies.

The generic `FsspecFlatHandler` lists a prefix by s3fs's ``_find`` of its
directory, and copies by its ``_cp_file``. `S3fsHandler` uses what s3fs has
beside them:

- its ``ils_recursive`` is s3fs's ``_iterdir`` with no delimiter: a page of
  ListObjectsV2 at a time, as it comes, narrowed to the prefix in the request.
  The generic one waits for the whole directory, holds it, then filters.
- its ``cp_file`` copies by the size its listing found, with s3fs's
  ``_copy_basic``, or ``_copy_managed`` beyond 5 GiB. s3fs's ``_cp_file``
  asks the size first, by a HeadObject per file.

It imports nothing of s3fs: it calls the wrapped filesystem's own methods.
Those are private, as of s3fs 2026.x: where one is missing, it lists or copies
as the generic handler does.

This module depends on `flat`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .flat import FsspecFlatHandler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..common.handler import Entry, StoragePath

__all__ = ["S3fsHandler"]

_COPY_LIMIT = 5 * 1024**3
"""The most CopyObject copies in one request, as s3fs's own ``_cp_file`` tells them apart."""


class S3fsHandler(FsspecFlatHandler):
    """An s3fs filesystem as a flat handler, listing and copying as s3fs does best.

    It takes what `FsspecFlatHandler` takes. A copy by the listed size never
    preserves a multipart ETag, as s3fs's ``preserve_etag`` would.

    Examples
    --------
    >>> flat = FlatFileSystem(
    ...     S3fsHandler(
    ...         s3fs.S3FileSystem(),
    ...         rm_batch_size=1000,
    ...         operation_concurrency=64,
    ...         io_concurrency=8,
    ...         storage_profile=known_implementations["s3fs.core.S3FileSystem"]["storage_profile"],
    ...     )
    ... )
    """

    def ils_recursive(  # type: ignore[override]
        self, entry: Entry, *, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[Entry]:
        path = entry["name"]
        if path and hasattr(self.fs, "_iterdir"):
            return self._keys(path, prefix)
        else:
            # The buckets: as the generic handler lists them.
            return super().ils_recursive(entry, prefix=prefix, **kwargs)

    async def _keys(self, path: str, prefix: str) -> AsyncIterator[Entry]:
        """Every key starting with ``path/prefix``, in name order, a page at a time."""
        bucket, key, _ = self.fs.split_path(path)
        start = f"{key.rstrip('/')}/{prefix}" if key else prefix
        try:
            async for entry in self.fs._iterdir(bucket, delimiter="", prefix=start):
                yield entry
        except FileNotFoundError:
            return  # A missing bucket lists nothing, as a missing prefix does.

    async def cp_file(self, entry: Entry, path2: StoragePath) -> None:
        size = entry.get("size")
        own = hasattr(self.fs, "_copy_basic") and hasattr(self.fs, "_copy_managed")
        if size is None or not own:
            await super().cp_file(entry, path2)
        elif size <= _COPY_LIMIT:
            await self.fs._copy_basic(entry["name"], path2)
        else:
            await self.fs._copy_managed(entry["name"], path2, size)

    async def mv_file(self, entry: Entry, path2: StoragePath) -> None:
        # S3 renames nothing: a copy by the listed size, then a delete.
        await self.cp_file(entry, path2)
        await self.fs._rm_file(entry["name"])
