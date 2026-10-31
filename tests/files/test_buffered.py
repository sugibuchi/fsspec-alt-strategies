"""Tests of `LocallyBufferedFile`, for storages transferring only whole files."""

from __future__ import annotations

import asyncio
import errno
import io
from typing import IO, Any

import pytest

from fsspec_alt_strategies.files.buffered import LocallyBufferedFile
from fsspec_alt_strategies.testing import SimulatedFileSystem


class _Whole(LocallyBufferedFile):
    """A file of a `SimulatedFileSystem` read and written whole, as some storages do."""

    def __init__(self, fs: SimulatedFileSystem, path: str, mode: str, **kwargs: Any):
        super().__init__(path, fs.loop, mode, **kwargs)
        self.fs = fs

    async def _download(self, target: IO[bytes]) -> None:
        target.write(await self.fs._cat_file(self.path))

    async def _upload(self, source: IO[bytes]) -> None:
        await self.fs._pipe_file(self.path, source.read())

    async def _exists(self) -> bool:
        return await self.fs._exists(self.path)


@pytest.fixture
def fs() -> SimulatedFileSystem:
    # A sync filesystem: the file's sync methods wait on its loop.
    fs = SimulatedFileSystem(hierarchical=True)
    fs.add_file("data/f.bin", b"0123456789")
    return fs


def test_reads_with_one_request(fs: SimulatedFileSystem) -> None:
    with _Whole(fs, "data/f.bin", "rb") as f:
        assert f.read(3) == b"012"
        f.seek(-2, io.SEEK_END)
        assert f.read() == b"89"
        f.seek(0)
        assert f.read() == b"0123456789"
    # One download, then all is local. Nothing changed: no upload.
    assert fs.stats.calls == {"read": 1}


def test_opening_downloads_nothing_until_used(fs: SimulatedFileSystem) -> None:
    f = _Whole(fs, "missing.bin", "rb")
    assert fs.stats.calls == {}
    with pytest.raises(FileNotFoundError):
        f.load()
    f.close()
    assert fs.stats.calls == {"read": 1}


def test_updates_in_place(fs: SimulatedFileSystem) -> None:
    with _Whole(fs, "data/f.bin", "r+b") as f:
        f.seek(2)
        f.write(b"ab")
        f.truncate(6)
    assert fs.cat_file("data/f.bin") == b"01ab45"
    # The download, one upload, and this test's own read.
    assert fs.stats.calls == {"read": 2, "write": 1}

    fs.reset_stats()
    with _Whole(fs, "data/f.bin", "r+b") as f:
        f.read()
    assert fs.stats.calls == {"read": 1}  # Read only: no upload.


def test_writes_a_new_file_even_empty(fs: SimulatedFileSystem) -> None:
    with _Whole(fs, "new.bin", "wb") as f:
        f.write(b"hello")
    assert fs.cat_file("new.bin") == b"hello"
    with _Whole(fs, "empty.bin", "wb"):
        pass
    assert fs.cat_file("empty.bin") == b""


def test_appends_at_the_end(fs: SimulatedFileSystem) -> None:
    with _Whole(fs, "data/f.bin", "a+b") as f:
        f.seek(0)
        assert f.read(2) == b"01"
        f.write(b"!")  # At the end, wherever the position was.
    assert fs.cat_file("data/f.bin") == b"0123456789!"
    with _Whole(fs, "created.bin", "ab") as f:
        f.write(b"x")
    assert fs.cat_file("created.bin") == b"x"


def test_exclusive_creation(fs: SimulatedFileSystem) -> None:
    f = _Whole(fs, "data/f.bin", "xb")
    with pytest.raises(FileExistsError):
        f.write(b"x")
    f.discard()
    f.close()
    assert fs.cat_file("data/f.bin") == b"0123456789"
    with _Whole(fs, "fresh.bin", "xb") as f:
        f.write(b"x")
    assert fs.cat_file("fresh.bin") == b"x"


def test_an_exception_in_the_block_uploads_nothing(fs: SimulatedFileSystem) -> None:
    with pytest.raises(RuntimeError), _Whole(fs, "data/f.bin", "r+b") as f:
        f.write(b"half")
        raise RuntimeError("failed midway")
    assert fs.cat_file("data/f.bin") == b"0123456789"
    with _Whole(fs, "data/f.bin", "wb") as f:
        f.write(b"never")
        f.discard()
    assert fs.cat_file("data/f.bin") == b"0123456789"


def test_max_size(fs: SimulatedFileSystem) -> None:
    with _Whole(fs, "new.bin", "wb", max_size=4) as f:
        f.write(b"1234")
        with pytest.raises(OSError, match="too large") as info:
            f.write(b"5")
        assert info.value.errno == errno.EFBIG
    f = _Whole(fs, "data/f.bin", "rb", max_size=4)
    with pytest.raises(OSError, match="too large"):
        f.load()
    f.close()


def test_spills_to_disk_beyond_spool_size(fs: SimulatedFileSystem) -> None:
    data = b"x" * 10_000
    with _Whole(fs, "big.bin", "w+b", spool_size=1024) as f:
        f.write(data)
        assert f._buffer is not None
        assert f._buffer._rolled  # type: ignore[attr-defined]
        f.seek(0)
        assert f.read() == data
    assert fs.cat_file("big.bin") == data


def test_modes(fs: SimulatedFileSystem) -> None:
    for mode in ("r", "rt", "rbb", "rwb", "b", "r+"):
        with pytest.raises(ValueError, match="mode"):
            _Whole(fs, "data/f.bin", mode)
    with _Whole(fs, "data/f.bin", "rb") as f:
        assert f.readable() and not f.writable()
        with pytest.raises(io.UnsupportedOperation):
            f.write(b"x")
    with _Whole(fs, "new.bin", "wb") as f:
        assert f.writable() and not f.readable()
        with pytest.raises(io.UnsupportedOperation):
            f.read()


async def test_async_methods_and_the_concurrency_limit() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    fs.add_file("f.bin", b"abc")
    limit = asyncio.Semaphore(1)
    await limit.acquire()  # Held by another file.
    f = _Whole(fs, "f.bin", "r+b", concurrency_limit=limit)
    f.loop = asyncio.get_running_loop()
    reading = asyncio.ensure_future(f._read())
    await asyncio.sleep(0.01)
    assert fs.stats.calls == {}  # The download waits for the limit.
    limit.release()
    assert await reading == b"abc"
    await f._write(b"d")
    await f.aclose()
    assert await fs._cat_file("f.bin") == b"abcd"
