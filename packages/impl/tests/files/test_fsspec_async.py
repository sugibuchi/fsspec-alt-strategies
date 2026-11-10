"""Tests of `files.fsspec_async`: the file objects as `open_async` returns them."""

from __future__ import annotations

import asyncio
import io
import os
import threading
from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies_impl.files.adaptive import (
    AdaptiveReadableFile,
    AdaptiveWritableFile,
)
from fsspec_alt_strategies_impl.files.fsspec_async import (
    AsyncFileAdapter,
    SyncFileAdapter,
)
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from pathlib import Path

_KiB = 1024


class _Reader(AdaptiveReadableFile):
    def __init__(self, fs: SimulatedFileSystem, path: str, size: int) -> None:
        super().__init__(
            path, asyncio.get_running_loop(), size, min_block_size=4 * _KiB
        )
        self.fs = fs

    async def _fetch_range(self, start: int, end: int) -> bytes:
        return await self.fs._cat_file(self.path, start=start, end=end)


class _Writer(AdaptiveWritableFile):
    def __init__(self, fs: SimulatedFileSystem, path: str, loop: Any = None) -> None:
        super().__init__(
            path,
            loop or asyncio.get_running_loop(),
            min_block_size=4 * _KiB,
            max_block_size=8 * _KiB,
            min_multipart_size=4 * _KiB,
        )
        self.fs = fs
        self.parts: dict[int, bytes] = {}

    async def _upload_all(self, data: bytes) -> None:
        await self.fs._pipe_file(self.path, data)

    async def _start_multipart_upload(self) -> None:
        pass

    async def _upload_part(
        self, data: bytes, start: int, end: int, index: int, last: bool
    ) -> Any:
        self.parts[index] = data
        return index

    async def _complete_multipart_upload(self, parts: list[Any]) -> None:
        await self.fs._pipe_file(self.path, b"".join(self.parts[i] for i in parts))

    async def _abort_multipart_upload(self) -> None:
        pass


@pytest.fixture
def fs() -> SimulatedFileSystem:
    return SimulatedFileSystem(hierarchical=True)


async def test_reads_and_seeks_as_bytesio(fs: SimulatedFileSystem) -> None:
    data = os.urandom(100 * _KiB)
    fs.pipe_file("f.bin", data)
    reference = io.BytesIO(data)
    async with AsyncFileAdapter(_Reader(fs, "f.bin", len(data))) as f:
        assert (f.mode, f.size, f.readable(), f.writable()) == ("rb", len(data), 1, 0)
        assert await f.read(10) == reference.read(10)
        # Not awaited, as in fsspec's async files: seeking sends nothing.
        assert f.seek(-8, io.SEEK_END) == reference.seek(-8, io.SEEK_END)
        assert f.tell() == f.loc == reference.tell()
        assert await f.read() == reference.read()
        with pytest.raises(io.UnsupportedOperation):
            await f.write(b"x")
    assert f.closed


async def test_writes_in_parts(fs: SimulatedFileSystem) -> None:
    data = os.urandom(40 * _KiB)
    file = _Writer(fs, "f.bin")
    async with AsyncFileAdapter(file) as f:
        assert (f.mode, f.size) == ("wb", None)
        for i in range(0, len(data), 3 * _KiB):
            await f.write(data[i : i + 3 * _KiB])
        await f.flush()
        assert not fs.exists("f.bin")  # Unforced, nothing is shown.
        await f.flush(force=True)
        # Forced, the file is finished, as fsspec's files are.
        assert fs.cat_file("f.bin") == data
        with pytest.raises(ValueError, match="only close"):
            await f.write(b"more")
        assert f.tell() == len(data)
        with pytest.raises(io.UnsupportedOperation):
            f.seek(0)
    assert len(file.parts) > 1
    assert fs.cat_file("f.bin") == data


async def test_a_sync_file_reads_and_writes_in_threads(tmp_path: Path) -> None:
    data = os.urandom(10 * _KiB)
    path = tmp_path / "f.bin"
    async with SyncFileAdapter(open(path, "wb")) as f:  # noqa: ASYNC230
        assert (f.path, f.mode, f.writable(), f.readable()) == (str(path), "wb", 1, 0)
        assert await f.write(data) == len(data)
        await f.flush()
    assert f.closed
    assert path.read_bytes() == data

    reference = io.BytesIO(data)
    f = SyncFileAdapter(open(path, "rb"))  # noqa: ASYNC230, SIM115
    assert await f.read(10) == reference.read(10)
    # Not awaited, as in fsspec's async files.
    assert f.seek(-8, io.SEEK_END) == reference.seek(-8, io.SEEK_END)
    assert f.tell() == f.loc == reference.tell()
    assert await f.read(None) == reference.read()
    await f.close()
    assert f.closed


class _Blocking(io.BytesIO):
    """A file whose reads block until let go, recording the threads they run in."""

    mode = "rb"

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.go = threading.Event()
        self.started = threading.Event()
        self.threads: list[int] = []
        self.flushes: list[bool] = []

    def read(self, size: int | None = -1) -> bytes:
        self.threads.append(threading.get_ident())
        self.started.set()
        self.go.wait(5)
        return super().read(size)

    def flush(self, force: bool = False) -> None:  # type: ignore[override]
        self.flushes.append(force)


async def test_a_sync_file_leaves_the_loop_free() -> None:
    file = _Blocking(b"0123456789")
    f = SyncFileAdapter(file)
    reading = asyncio.create_task(f.read(4))
    await asyncio.to_thread(file.started.wait, 5)
    assert not reading.done()  # The loop runs on while the read blocks.
    file.go.set()
    assert await reading == b"0123"
    assert file.threads != [threading.get_ident()]
    # force reaches a file taking it, as fsspec's.
    await f.flush()
    await f.flush(force=True)
    assert file.flushes == [False, True]


class _Handle:
    """An SDK's handle: a sync file telling neither its mode nor its size."""

    def __init__(self, data: bytes) -> None:
        self._data = io.BytesIO(data)
        self.read = self._data.read
        self.close = self._data.close
        self.opened_in = threading.get_ident()

    @property
    def closed(self) -> bool:
        return self._data.closed


async def test_a_handle_is_given_its_mode_and_size() -> None:
    def opener() -> Any:
        return _Handle(b"0123456789")

    f = await SyncFileAdapter.open(opener, mode="rb", size=10)
    assert f.file.opened_in != threading.get_ident()  # Opened in a thread.
    assert (f.mode, f.size) == ("rb", 10)
    assert await f.read() == b"0123456789"
    await f.close()
    assert f.closed


async def test_a_file_tells_its_own_mode_and_size() -> None:
    f = SyncFileAdapter(_Blocking(b"0123"))
    assert (f.mode, f.size) == ("rb", None)


async def test_seeking_while_a_thread_runs_is_refused() -> None:
    file = _Blocking(b"0123456789")
    f = SyncFileAdapter(file)
    reading = asyncio.create_task(f.read(4))
    await asyncio.to_thread(file.started.wait, 5)
    # The file's own would wait for the read, blocking the loop.
    with pytest.raises(RuntimeError, match="runs in a thread"):
        f.seek(0)
    with pytest.raises(RuntimeError, match="runs in a thread"):
        f.tell()
    file.go.set()
    assert await reading == b"0123"
    assert f.tell() == 4


async def test_a_cancelled_call_frees_the_file_only_when_its_thread_ends() -> None:
    file = _Blocking(b"0123456789")
    f = SyncFileAdapter(file)
    first = asyncio.create_task(f.read(4))
    await asyncio.to_thread(file.started.wait, 5)
    first.cancel()
    second = asyncio.create_task(f.read(2))
    await asyncio.sleep(0.05)
    # The first's thread holds the file still: neither has ended.
    assert not first.done()
    assert not second.done()
    assert len(file.threads) == 1
    file.go.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    # The first's read happened, in its thread: the second reads on from there.
    assert await second == b"45"


async def test_a_call_cancelled_twice_still_waits_for_its_thread() -> None:
    file = _Blocking(b"0123456789")
    f = SyncFileAdapter(file)
    first = asyncio.create_task(f.read(4))
    await asyncio.to_thread(file.started.wait, 5)
    first.cancel()
    await asyncio.sleep(0.01)
    first.cancel()  # Again, while it waits for its thread.
    second = asyncio.create_task(f.read(2))
    await asyncio.sleep(0.05)
    assert not first.done()
    assert not second.done()
    assert len(file.threads) == 1  # Never two threads in the file at once.
    file.go.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second == b"45"
