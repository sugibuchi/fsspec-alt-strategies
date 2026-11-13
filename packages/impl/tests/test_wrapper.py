"""Tests of what the handlers of both families pass on to the filesystem wrapped."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.router import AsyncRoutingFileSystem
from fsspec_alt_strategies_impl.tree.filesystem import TreeFileSystem
from fsspec_alt_strategies_impl.wrapper.flat import FsspecFlatHandler
from fsspec_alt_strategies_impl.wrapper.tree import FsspecTreeHandler
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from pathlib import Path

_LIMITS: dict[str, Any] = {
    "operation_concurrency": 4,
    "io_concurrency": 2,
    "storage_profile": StorageProfile(),
}

# What a router passes on to its storages, each a filesystem on a handler maybe.
_ROUTED = {
    "_info", "_ls", "_cat_file", "_pipe_file", "_get_file", "_put_file",
    "_rm_file", "_rmdir", "_mkdir", "_makedirs", "_cp_file", "_mv_file",
    "open_async", "_open", "created", "modified", "sign", "invalidate_cache",
}  # fmt: skip
# What a handler wrapping a filesystem passes on beside the primitives.
_PASSED_ON = {
    "cat_file", "pipe_file", "get_file", "put_file", "open_async_file",
    "open_sync_file", "created", "modified", "sign", "invalidate_cache",
}  # fmt: skip


@pytest.mark.parametrize("handler", [FsspecTreeHandler, FsspecFlatHandler])
def test_every_primitive_is_passed_on(handler: type) -> None:
    assert not handler.__abstractmethods__
    assert set(vars(AsyncRoutingFileSystem)) >= _ROUTED
    assert {n for c in handler.__mro__ for n in vars(c)} >= _PASSED_ON


def _tree(fs: Any, **options: Any) -> TreeFileSystem:
    """A ``TreeFileSystem`` on a handler of ``fs``."""
    asynchronous = options.pop("asynchronous", getattr(fs, "asynchronous", False))
    return TreeFileSystem.of(
        FsspecTreeHandler(fs, **_LIMITS, **options), asynchronous=asynchronous
    )


async def test_an_async_filesystem_makes_a_dir() -> None:
    inner = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    tree = _tree(inner)
    await tree._mkdir("a/b")
    assert await inner._isdir("a/b")


async def test_a_sync_filesystem_makes_a_dir_and_dates_a_file(
    tmp_path: Path,
) -> None:
    local = LocalFileSystem()
    root = local._strip_protocol(str(tmp_path))
    tree = _tree(local, asynchronous=True)
    assert tree.handler is not None and not tree.handler.asynchronous  # type: ignore[attr-defined]
    await tree._mkdir(f"{root}/a/b")
    assert local.isdir(f"{root}/a/b")
    local.pipe_file(f"{root}/f.txt", b"f")
    assert tree.created(f"{root}/f.txt") == local.created(f"{root}/f.txt")
    assert tree.modified(f"{root}/f.txt") == local.modified(f"{root}/f.txt")


async def test_a_sync_filesystem_of_an_async_class_is_wrapped_as_sync(
    tmp_path: Path,
) -> None:
    """A ``DirFileSystem`` over a local one is sync: its ``async_impl`` says so."""
    local = LocalFileSystem()
    root = local._strip_protocol(str(tmp_path))
    local.makedirs(f"{root}/a")
    local.pipe_file(f"{root}/a/x.txt", b"x")
    dirfs = DirFileSystem(path=root, fs=local)
    tree = _tree(dirfs, asynchronous=True)
    assert not tree.handler.asynchronous  # type: ignore[union-attr]
    assert await tree._find("a") == ["a/x.txt"]
