"""Fixtures shared by the tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.implementations.memory import MemoryFileSystem

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def memfs() -> Iterator[MemoryFileSystem]:
    """A cleared memory filesystem, cleared again after the test.

    Memory filesystems share their store across instances, so a test must not
    see another's files.
    """
    fs = MemoryFileSystem()
    fs.store.clear()
    fs.pseudo_dirs.clear()
    fs.pseudo_dirs.append("")
    yield fs
    fs.store.clear()
    fs.pseudo_dirs.clear()
    fs.pseudo_dirs.append("")


@pytest.fixture
def async_memfs(memfs: MemoryFileSystem) -> AsyncFileSystemWrapper:
    """``memfs`` as an async filesystem, as the strategies mainly target."""
    return AsyncFileSystemWrapper(memfs, asynchronous=True)
