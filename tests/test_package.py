"""Tests of the package itself, and of the shared fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fsspec_alt_strategies

if TYPE_CHECKING:
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper


def test_version() -> None:
    assert fsspec_alt_strategies.__version__


async def test_async_memfs(async_memfs: AsyncFileSystemWrapper) -> None:
    await async_memfs._pipe_file("/a/b.txt", b"x")
    assert await async_memfs._cat_file("/a/b.txt") == b"x"
