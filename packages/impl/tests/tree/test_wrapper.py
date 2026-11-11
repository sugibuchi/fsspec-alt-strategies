"""Tests of the tree wrappers: existing filesystems, as a `TreeFileSystem`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.local import LocalFileSystem

from fsspec_alt_strategies.monitor import Call, Monitor, monitoring
from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.wrapper.tree import AsyncTreeWrapper, SyncTreeWrapper
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_LIMITS: dict[str, Any] = {
    "operation_concurrency": 4,
    "io_concurrency": 2,
    "storage_profile": StorageProfile(),
}
# Declared to delete a directory whole, in one request.
_DELETES: dict[str, Any] = {
    **_LIMITS,
    "storage_profile": StorageProfile(deletes_directories=True),
}
_FILES = ["data/2023/01/a.txt", "data/2023/02/b.txt", "data/2024/c.txt", "data/d.txt"]


class _Calls(Monitor):
    def __init__(self) -> None:
        self.methods: list[str] = []
        self.filesystems: list[Any] = []

    async def started(self, call: Call, state: Any, op: Any) -> None:
        self.methods.append(call.method)
        self.filesystems.append(call.fs)


def _simulated(**kwargs: Any) -> SimulatedFileSystem:
    fs = SimulatedFileSystem(hierarchical=True, **kwargs)
    for path in _FILES:
        fs.add_file(path, path.encode())
    return fs


async def test_an_async_filesystem_answers_as_fsspec() -> None:
    inner = _simulated(asynchronous=True)
    tree = AsyncTreeWrapper(inner, **_LIMITS)
    assert await tree._find("data") == await AsyncFileSystem._find(inner, "data")
    assert await tree._du("data") == await AsyncFileSystem._du(inner, "data")
    assert await tree._cat_file("data/d.txt") == b"data/d.txt"
    calls = _Calls()
    with monitoring(calls):
        await tree._mv("data/2023", "moved", recursive=True)
        await tree._rm("data", recursive=True)
    assert calls.methods == ["mv", "rm"]
    assert sorted(inner._file_data) == ["moved/01/a.txt", "moved/02/b.txt"]
    # The monitor is told of the filesystem wrapped, not of the wrapper.
    assert calls.filesystems == [inner, inner]


async def test_a_monitor_is_told_of_the_filesystem_under_all_wrappers() -> None:
    inner = _simulated(asynchronous=True)
    tree = AsyncTreeWrapper(AsyncTreeWrapper(inner, **_LIMITS), **_LIMITS)
    calls = _Calls()
    with monitoring(calls):
        await tree._find("data")
    assert calls.filesystems == [inner]


def test_an_async_filesystem_on_its_own_loop() -> None:
    inner = _simulated()  # Sync, its coroutines on fsspec's loop.
    tree = AsyncTreeWrapper(inner, **_LIMITS)
    assert tree.loop is inner.loop
    assert tree.find("data") == inner.find("data")
    tree.mv("data", "moved", recursive=True)
    assert sorted(inner._file_data) == sorted("moved" + p[4:] for p in _FILES)


class _RecursiveDelete(SimulatedFileSystem):
    """A storage deleting a directory whole, in one request."""

    async def _rm_tree(self, path: str) -> None:
        await self._rm(path, recursive=True)


async def test_an_async_filesystem_deletes_a_dir_by_its_rm_tree() -> None:
    inner = _RecursiveDelete(hierarchical=True, asynchronous=True)
    for path in _FILES:
        inner.add_file(path, path.encode())
    tree = AsyncTreeWrapper(inner, **_DELETES)
    inner.reset_stats()
    await tree._rm("data", recursive=True)
    # One deletion: no listing.
    assert inner.stats.calls == {"delete": 1}
    assert not inner._file_data


async def test_a_filesystem_without_rm_tree_deletes_by_its_recursive_rm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = _simulated()
    called: list[tuple[str, bool]] = []

    async def own_rm(path: str, recursive: bool = False, **kwargs: Any) -> None:
        called.append((path, recursive))

    monkeypatch.setattr(inner, "_rm", own_rm)
    await AsyncTreeWrapper(inner, **_DELETES)._rm_tree("data")
    assert called == [("data", True)]


async def test_a_sync_filesystem_deletes_by_its_recursive_rm(tmp_path: Path) -> None:
    local = LocalFileSystem()
    root = local._strip_protocol(str(tmp_path))
    local.makedirs(f"{root}/data/a")
    local.pipe_file(f"{root}/data/a/x.txt", b"x")
    tree = SyncTreeWrapper(local, asynchronous=True, **_DELETES)
    await tree._rm(f"{root}/data", recursive=True)
    assert not local.exists(f"{root}/data")


class _Renaming(SimulatedFileSystem):
    """A storage renaming a directory whole, in one request."""

    async def _mv_tree(self, path1: str, path2: str) -> None:
        await self._rename(path1, path2)


async def test_an_async_filesystem_moves_a_dir_by_its_mv_tree() -> None:
    inner = _Renaming(hierarchical=True, asynchronous=True)
    for path in _FILES:
        inner.add_file(path, path.encode())
    tree = AsyncTreeWrapper(
        inner,
        **{**_LIMITS, "storage_profile": StorageProfile(renames_directories=True)},
    )
    inner.reset_stats()
    await tree._mv("data", "moved", recursive=True)
    # The target found free, then one rename: no listing.
    assert inner.stats.calls == {"info": 1, "rename": 1}
    assert sorted(inner._file_data) == sorted("moved" + p[4:] for p in _FILES)


async def test_rm_tree_is_unused_unless_declared() -> None:
    inner = _RecursiveDelete(hierarchical=True, asynchronous=True)
    for path in _FILES:
        inner.add_file(path, path.encode())
    inner.reset_stats()
    await AsyncTreeWrapper(inner, **_LIMITS)._rm("data", recursive=True)
    assert inner.stats.calls["delete"] > 1  # Walked, file by file.
    assert not inner._file_data


def test_wrappers_are_made_anew() -> None:
    inner = _simulated()
    assert AsyncTreeWrapper(inner, **_LIMITS) is not AsyncTreeWrapper(inner, **_LIMITS)


def _local(tmp_path: Path) -> tuple[LocalFileSystem, str]:
    local = LocalFileSystem()
    root = local._strip_protocol(str(tmp_path))
    for path in _FILES:
        local.makedirs(f"{root}/{path}".rpartition("/")[0], exist_ok=True)
        local.pipe_file(f"{root}/{path}", path.encode())
    return local, root


def test_a_sync_filesystem_answers_as_fsspec(tmp_path: Path) -> None:
    local, root = _local(tmp_path)
    tree = SyncTreeWrapper(local, **_LIMITS)
    assert tree.find(f"{root}/data") == local.find(f"{root}/data")
    assert tree.du(f"{root}/data") == local.du(f"{root}/data")
    assert tree.cat_file(f"{root}/data/d.txt") == b"data/d.txt"
    with tree.open(f"{root}/data/d.txt", "rb") as f:
        assert f.read() == b"data/d.txt"
    tree.rm(f"{root}/data/2023", recursive=True)
    assert not local.exists(f"{root}/data/2023")
    assert local.exists(f"{root}/data/2024/c.txt")


async def test_a_sync_filesystem_moves_a_dir_whole(tmp_path: Path) -> None:
    local, root = _local(tmp_path)
    tree = SyncTreeWrapper(
        local,
        asynchronous=True,
        **{**_LIMITS, "storage_profile": StorageProfile(renames_directories=True)},
    )
    await tree._mv(f"{root}/data", f"{root}/moved", recursive=True)
    assert not local.exists(f"{root}/data")
    assert local.cat_file(f"{root}/moved/2023/01/a.txt") == b"data/2023/01/a.txt"
    async with await tree.open_async(f"{root}/moved/d.txt") as f:
        assert await f.read() == b"data/d.txt"
