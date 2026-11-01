"""Tests of `files.fsspec_async`: the file objects as `open_async` returns them."""

from __future__ import annotations

import asyncio
import io
import os
from typing import IO, Any

import pytest

from fsspec_alt_strategies.files.adaptive import (
    AdaptiveReadableFile,
    AdaptiveWritableFile,
)
from fsspec_alt_strategies.files.buffered import LocallyBufferedFile
from fsspec_alt_strategies.files.fsspec_async import AsyncFileAdapter
from fsspec_alt_strategies.testing import SimulatedFileSystem

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
    def __init__(self, fs: SimulatedFileSystem, path: str) -> None:
        super().__init__(
            path,
            asyncio.get_running_loop(),
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


class _Whole(LocallyBufferedFile):
    def __init__(
        self, fs: SimulatedFileSystem, path: str, mode: str, loop: Any = None
    ) -> None:
        super().__init__(path, loop or asyncio.get_running_loop(), mode)
        self.fs = fs

    async def _download(self, target: IO[bytes]) -> None:
        target.write(await self.fs._cat_file(self.path))

    async def _upload(self, source: IO[bytes]) -> None:
        await self.fs._pipe_file(self.path, source.read())

    async def _exists(self) -> bool:
        return await self.fs._exists(self.path)


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
        await f.flush(force=True)
        # The class chose: nothing is shown before closing.
        assert not fs.exists("f.bin")
        assert f.tell() == len(data)
        with pytest.raises(io.UnsupportedOperation):
            f.seek(0)
    assert len(file.parts) > 1
    assert fs.cat_file("f.bin") == data


async def test_a_locally_buffered_file(fs: SimulatedFileSystem) -> None:
    fs.pipe_file("f.bin", b"0123456789")
    fs.reset_stats()
    file = _Whole(fs, "f.bin", "r+b")
    await file.aload()
    async with AsyncFileAdapter(file) as f:
        # Loaded first: seek and tell are local.
        assert fs.stats.calls == {"read": 1}
        assert (f.size, f.tell()) == (10, 0)
        f.seek(4)
        assert await f.write(b"ab") == 2
        assert await f.read() == b"6789"
    assert fs.cat_file("f.bin") == b"0123ab6789"

    # An exception in the block discards the changes, as in a with block.
    with pytest.raises(RuntimeError):
        async with AsyncFileAdapter(_Whole(fs, "f.bin", "wb")) as f:
            await f.write(b"lost")
            raise RuntimeError("failed midway")
    assert fs.cat_file("f.bin") == b"0123ab6789"


async def test_an_unloaded_buffered_file_cannot_seek_on_its_loop(
    fs: SimulatedFileSystem,
) -> None:
    fs.pipe_file("f.bin", b"0123456789")
    file = _Whole(fs, "f.bin", "rb")
    with pytest.raises(RuntimeError, match="own loop"):
        file.tell()
    # An awaited read loads it: then seek and tell are local.
    f = AsyncFileAdapter(file)
    assert await f.read(2) == b"01"
    assert f.seek(-1, io.SEEK_END) == 9
    await f.close()


async def test_refuses_another_loop(fs: SimulatedFileSystem) -> None:
    # The file's loop is the filesystem's, not the running one.
    file = _Whole(fs, "f.bin", "wb", loop=fs.loop)
    with pytest.raises(RuntimeError, match="outside the loop"):
        await AsyncFileAdapter(file).write(b"x")
    file.close()
