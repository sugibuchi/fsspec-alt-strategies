# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What the filesystem classes of the families share, e.g. `TreeFileSystem`'s.

`_FileSystem` holds the two concurrency limits, and moves data through file
objects: its ``_cat_file``, ``_pipe_file``, ``_get_file`` and ``_put_file``
open the file with ``open_async``, which a subclass implements.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

from fsspec.asyn import AsyncFileSystem

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_MiB = 1024 * 1024


class _FileSystem(AsyncFileSystem):
    """An async filesystem with two concurrency limits, moving data through files.

    ``operation_limit`` bounds the requests of the strategies, such as
    listings, deletions and copies. ``io_limit`` bounds the reads and writes of
    data: ``open_async`` gives it to the files it opens, as their
    ``concurrency_limit``. Each is an `asyncio.Semaphore`, its own per
    instance.
    """

    transfer_size: int = 8 * _MiB
    """The bytes read or written at a time by ``_get_file`` and ``_put_file``."""

    def __init__(
        self,
        *args: Any,
        operation_concurrency: int,
        io_concurrency: int,
        **kwargs: Any,
    ) -> None:
        for name, value in (
            ("operation_concurrency", operation_concurrency),
            ("io_concurrency", io_concurrency),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1: {value}")
        super().__init__(*args, **kwargs)
        self.operation_concurrency = operation_concurrency
        self.io_concurrency = io_concurrency
        self.operation_limit = asyncio.Semaphore(operation_concurrency)
        self.io_limit = asyncio.Semaphore(io_concurrency)

    def _plain(self, path: str | Sequence[str]) -> Any:
        """A path, or paths, without the protocol, keeping a trailing "/".

        The strategies take paths without a protocol. A trailing "/" means
        something to some, e.g. a glob of folders only, or a copy into a folder.
        """
        if not isinstance(path, str):
            return [self._plain(p) for p in path]
        plain = self._strip_protocol(path)
        return plain + "/" if path.endswith("/") and not plain.endswith("/") else plain

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        f = await self.open_async(path, "rb", **kwargs)
        async with f:
            if start is not None:
                f.seek(start if start >= 0 else max(0, f.size + start))
            if end is not None:
                if end < 0:
                    end = f.size + end
                return await f.read(end - f.tell())
            return await f.read()

    async def _pipe_file(
        self, path: str, value: bytes, mode: str = "overwrite", **kwargs: Any
    ) -> None:
        f = await self.open_async(path, "xb" if mode == "create" else "wb", **kwargs)
        async with f:
            await f.write(value)

    async def _get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        if os.path.isdir(lpath):  # noqa: ASYNC240, as the local open below.
            return  # A folder, which fsspec's get makes itself.
        f = await self.open_async(rpath, "rb", **kwargs)
        async with f:
            # Local writes are short next to a storage's requests, as in fsspec's
            # own filesystems.
            with open(lpath, "wb") as local:  # noqa: ASYNC230
                while chunk := await f.read(self.transfer_size):
                    local.write(chunk)

    async def _put_file(
        self, lpath: str, rpath: str, mode: str = "overwrite", **kwargs: Any
    ) -> None:
        if os.path.isdir(lpath):  # noqa: ASYNC240, as the local open below.
            return  # A folder: a storage needs none for the files put in it.
        f = await self.open_async(rpath, "xb" if mode == "create" else "wb", **kwargs)
        async with f:
            with open(lpath, "rb") as local:  # noqa: ASYNC230
                while chunk := local.read(self.transfer_size):
                    await f.write(chunk)


def _raise(errors: list[Exception], doing: str) -> None:
    """Raise the errors an operation yielded: one alone, several in a group."""
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup(f"{doing} failed for {len(errors)} paths", errors)


def _failed_root(
    root: str,
    error: Exception,
    on_error: str | Callable[[Exception], Any],
    detail: bool,
) -> tuple[str, Any, Any]:
    """What fsspec's ``walk`` yields for a root that failed to list."""
    if on_error == "raise":
        raise error
    if callable(on_error):
        on_error(error)
    return (root, {}, {}) if detail else (root, [], [])
