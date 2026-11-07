"""Tests of the tree wrappers: existing filesystems, as a `TreeFileSystem`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.local import LocalFileSystem

from fsspec_alt_strategies.tree.wrapper import AsyncTreeWrapper, SyncTreeWrapper
from fsspec_alt_strategies_common.monitor import Call, Monitor, monitoring
from fsspec_alt_strategies_common.profile import StorageProfile
from fsspec_alt_strategies_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from pathlib import Path

_LIMITS = {"operation_concurrency": 4, "io_concurrency": 2}
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


async def test_a_sync_filesystem_moves_a_folder_whole(tmp_path: Path) -> None:
    local, root = _local(tmp_path)
    tree = SyncTreeWrapper(
        local,
        storage_profile=StorageProfile(renames_folders=True),
        asynchronous=True,
        **_LIMITS,
    )
    await tree._mv(f"{root}/data", f"{root}/moved", recursive=True)
    assert not local.exists(f"{root}/data")
    assert local.cat_file(f"{root}/moved/2023/01/a.txt") == b"data/2023/01/a.txt"
    async with await tree.open_async(f"{root}/moved/d.txt") as f:
        assert await f.read() == b"data/d.txt"
