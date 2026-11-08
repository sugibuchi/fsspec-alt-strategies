"""Tests of what the wrappers of both families pass on to the filesystem wrapped."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fsspec.implementations.local import LocalFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.flat.filesystem import FlatFileSystem
from fsspec_alt_strategies_impl.router import AsyncRoutingFileSystem
from fsspec_alt_strategies_impl.tree.filesystem import TreeFileSystem
from fsspec_alt_strategies_impl.wrapper import (
    AsyncTreeWrapper,
    SyncTreeWrapper,
    _AsyncFileSystemWrapper,
    _SyncFileSystemWrapper,
)
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from pathlib import Path

_LIMITS: dict[str, Any] = {
    "operation_concurrency": 4,
    "io_concurrency": 2,
    "storage_profile": StorageProfile(),
}

# What each family's wrapper defines itself: its listing, and the flat deletions.
_FAMILYS_OWN = {"_ils", "_iexpand", "_rm_files"}
# What a router passes on to its storages, each a wrapper maybe.
_ROUTED = {
    "_info", "_ls", "_cat_file", "_pipe_file", "_get_file", "_put_file",
    "_rm_file", "_rmdir", "_mkdir", "_makedirs", "_cp_file", "_mv_file",
    "open_async", "_open", "created", "modified", "sign", "invalidate_cache",
}  # fmt: skip


@pytest.mark.parametrize("wrapper", [_AsyncFileSystemWrapper, _SyncFileSystemWrapper])
def test_every_primitive_is_passed_on(wrapper: type) -> None:
    declared = TreeFileSystem.__abstractmethods__ | FlatFileSystem.__abstractmethods__
    assert set(vars(AsyncRoutingFileSystem)) >= _ROUTED
    passed_on = {n for c in wrapper.__mro__ for n in vars(c)}
    assert (declared - _FAMILYS_OWN) | _ROUTED <= passed_on


async def test_an_async_filesystem_makes_a_folder() -> None:
    inner = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    tree = AsyncTreeWrapper(inner, **_LIMITS)
    await tree._mkdir("a/b")
    assert await inner._isdir("a/b")


async def test_a_sync_filesystem_makes_a_folder_and_dates_a_file(
    tmp_path: Path,
) -> None:
    local = LocalFileSystem()
    root = local._strip_protocol(str(tmp_path))
    tree = SyncTreeWrapper(local, asynchronous=True, **_LIMITS)
    await tree._mkdir(f"{root}/a/b")
    assert local.isdir(f"{root}/a/b")
    local.pipe_file(f"{root}/f.txt", b"f")
    assert tree.created(f"{root}/f.txt") == local.created(f"{root}/f.txt")
    assert tree.modified(f"{root}/f.txt") == local.modified(f"{root}/f.txt")
