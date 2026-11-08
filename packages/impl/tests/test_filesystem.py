"""Tests of `common.filesystem` and `common.wrapper`: what the families' classes share.

The families test their filesystems through fsspec's methods. These test the
data methods the wrappers pass on, and the helpers shaping fsspec's answers.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from fsspec_alt_strategies.common.filesystem import (
    _BaseFileSystem,
    _copy_errors,
    _failed_folder,
    _raise,
)
from fsspec_alt_strategies.files.fsspec_async import SyncFileAdapter
from fsspec_alt_strategies.flat.wrapper import AsyncFlatWrapper
from fsspec_alt_strategies.tree.wrapper import AsyncTreeWrapper, SyncTreeWrapper
from fsspec_alt_strategies_common.profile import StorageProfile
from fsspec_alt_strategies_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

_LIMITS: dict[str, Any] = {
    "operation_concurrency": 2,
    "io_concurrency": 1,
    "storage_profile": StorageProfile(),
}


class _Transfers(SimulatedFileSystem):
    """A simulated storage noting the transfers passed on to it, and the files opened."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.transfers: list[tuple[str, str, str]] = []

    async def _get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        self.transfers.append(("get", rpath, lpath))

    async def _put_file(self, lpath: str, rpath: str, **kwargs: Any) -> None:
        self.transfers.append(("put", lpath, rpath))

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return ("opened", path, mode)

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return ("open", path, mode)


async def test_an_async_wrapper_passes_data_methods_on() -> None:
    inner = _Transfers(asynchronous=True)
    inner.add_file("a", b"0123456789")
    tree = AsyncTreeWrapper(inner, **_LIMITS)
    assert await tree._cat_file("a", start=2, end=5) == b"234"
    await tree._pipe_file("b", b"new")
    assert await inner._cat_file("b") == b"new"
    await tree._get_file("a", "local")
    await tree._put_file("local", "c")
    assert inner.transfers == [("get", "a", "local"), ("put", "local", "c")]
    assert await tree.open_async("a") == ("opened", "a", "rb")
    assert tree._open("a", "wb") == ("open", "a", "wb")
    assert tree.io_limit._value == 1


async def test_a_sync_wrapper_passes_data_methods_on(tmp_path: Path) -> None:
    inner = MemoryFileSystem()
    inner.pipe_file("/w/a", b"0123456789")
    tree = SyncTreeWrapper(inner, asynchronous=True, **_LIMITS)
    assert await tree._cat_file("/w/a", start=2, end=5) == b"234"
    await tree._pipe_file("/w/b", b"new")
    assert inner.cat_file("/w/b") == b"new"
    local = tmp_path / "a"
    await tree._get_file("/w/a", str(local))
    assert local.read_bytes() == b"0123456789"
    await tree._put_file(str(local), "/w/c")
    assert inner.cat_file("/w/c") == b"0123456789"
    with tree._open("/w/c", "rb") as f:
        assert f.read() == b"0123456789"
    await tree._cp_file("/w/c", "/w/e")
    await tree._mv_file("/w/c", "/w/d")
    assert sorted(inner.ls("/w", detail=False)) == ["/w/a", "/w/b", "/w/d", "/w/e"]
    assert tree.unstrip_protocol("/w/d") == "memory:///w/d"
    assert tree.io_limit._value == 1


async def test_blocking_calls_run_on_bounded_threads() -> None:
    fs = SyncTreeWrapper(MemoryFileSystem(), asynchronous=True, **_LIMITS)
    lock = threading.Lock()
    running = peak = 0
    threads: set[str] = set()

    def call(value: int) -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
            threads.add(threading.current_thread().name)
        time.sleep(0.02)
        with lock:
            running -= 1
        if value < 0:
            raise ValueError(value)
        return value

    results = await asyncio.gather(*(fs._blocking(call, i) for i in range(10)))
    assert results == list(range(10))
    # Two threads: the two operations allowed at once.
    assert peak == 2
    assert len(threads) == 2
    assert all(name.startswith("SyncTreeWrapper") for name in threads)
    with pytest.raises(ValueError, match="-1"):
        await fs._blocking(call, -1)


async def test_close_shuts_the_threads_down() -> None:
    fs = SyncTreeWrapper(MemoryFileSystem(), asynchronous=True, **_LIMITS)
    fs.close()  # Nothing to shut down yet.
    assert await fs._blocking(threading.current_thread) is not threading.main_thread()
    executor = fs._executor
    assert executor is not None
    fs.close()
    assert fs._executor is None
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(int)
    # A call after closing gets new threads.
    assert await fs._blocking(int, "4") == 4
    fs.close()


async def test_a_cancelled_blocking_call_is_not_swallowed() -> None:
    fs = SyncTreeWrapper(MemoryFileSystem(), asynchronous=True, **_LIMITS)
    started, release = threading.Event(), threading.Event()

    def call() -> None:
        started.set()
        release.wait(5)

    task = asyncio.create_task(fs._blocking(call))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()  # The thread ran on: it ends only now.
    fs.close()


class _Unseekable(io.BytesIO):
    """An SDK's handle streaming its data: it cannot seek, and notes its reads."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.reads: list[int] = []

    def seekable(self) -> bool:
        return False

    def seek(self, *args: Any) -> int:
        raise io.UnsupportedOperation("seek")

    def read(self, size: int | None = -1) -> bytes:
        self.reads.append(-1 if size is None else size)
        return super().read(size)


class _Streaming(SyncTreeWrapper):
    """A storage whose files cannot seek."""

    def __init__(self, data: bytes) -> None:
        super().__init__(MemoryFileSystem(), asynchronous=True, **_LIMITS)
        self.data = data
        self.opened: list[_Unseekable] = []

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        self.opened.append(_Unseekable(self.data))
        return SyncFileAdapter(self.opened[-1], mode=mode, size=len(self.data))


async def test_a_range_of_an_unseekable_file_skips_its_start() -> None:
    fs = _Streaming(b"0123456789")
    fs.transfer_size = 3
    cat = _BaseFileSystem._cat_file
    assert await cat(fs, "f", start=7, end=9) == b"78"
    # Read and dropped in chunks no bigger than a transfer.
    assert fs.opened[-1].reads == [3, 3, 1, 2]
    assert await cat(fs, "f", start=-3) == b"789"
    assert await cat(fs, "f", start=4, end=-4) == b"45"
    assert await cat(fs, "f", start=20, end=30) == b""
    assert await cat(fs, "f", start=6, end=2) == b""


async def test_a_local_folder_is_no_transfer(tmp_path: Path) -> None:
    # Both return before opening a file: fsspec makes local folders itself.
    tree = SyncTreeWrapper(MemoryFileSystem(), asynchronous=True, **_LIMITS)
    assert await _BaseFileSystem._get_file(tree, "/w/a", str(tmp_path)) is None
    assert await _BaseFileSystem._put_file(tree, str(tmp_path), "/w/a") is None


def test_plain_paths_keep_a_trailing_slash() -> None:
    tree = SyncTreeWrapper(MemoryFileSystem(), **_LIMITS)
    assert tree._plain(["memory://w/a/", "w/b"]) == ["/w/a/", "/w/b"]


def test_a_copy_drops_sources_gone_unless_errors_raise() -> None:
    gone, other = FileNotFoundError("a"), OSError("b")
    assert _copy_errors([gone, other], None) == [other]
    assert _copy_errors([gone, other], "raise") == [gone, other]


def test_errors_are_raised_alone_or_in_a_group() -> None:
    _raise([], "Deleting")
    with pytest.raises(OSError, match=r"^a$"):
        _raise([OSError("a")], "Deleting")
    with pytest.raises(ExceptionGroup, match="Deleting failed for 2 paths"):
        _raise([OSError("a"), OSError("b")], "Deleting")


def test_a_folder_failing_to_list_is_raised_reported_or_empty() -> None:
    error = OSError("a")
    with pytest.raises(OSError, match=r"^a$"):
        _failed_folder("a", error, "raise", detail=False)
    reported: list[Exception] = []
    assert _failed_folder("a", error, reported.append, detail=True) == ("a", {}, {})
    assert reported == [error]
    assert _failed_folder("a", error, "omit", detail=False) == ("a", [], [])


class _Flat(AsyncFlatWrapper):
    """A simulated storage as a `FlatFileSystem`, listed by its own ``_ils``."""

    def _ils(self, prefix: str) -> AsyncIterator[dict[str, Any]]:
        return self.fs._ils(prefix)


@pytest.mark.parametrize("family", ["tree", "flat"])
@pytest.mark.parametrize("method", ["_copy", "_mv"])
@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("path", ["data/a/1.txt", "data/a/1.*", ["data/a/1.txt"]])
async def test_a_file_copied_or_moved_into_a_missing_folder(
    family: str, method: str, recursive: bool, path: str | list[str]
) -> None:
    sim = SimulatedFileSystem(hierarchical=family == "tree", asynchronous=True)
    sim.add_file("data/a/1.txt", b"1")
    if family == "tree":
        fs: Any = AsyncTreeWrapper(sim, asynchronous=True, **_LIMITS)
    else:
        fs = _Flat(sim, rm_batch_size=10, asynchronous=True, **_LIMITS)

    await getattr(fs, method)(path, "new/", recursive=recursive)
    assert sim._file_data["new/1.txt"].data == b"1"
    assert "new" not in sim._file_data
