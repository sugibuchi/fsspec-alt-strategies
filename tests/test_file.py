"""Tests of the adaptive file objects, on `SimulatedFileSystem`."""

from __future__ import annotations

import asyncio
import io
import os
import random
import struct
from typing import Any

import pytest

from fsspec_alt_strategies import file
from fsspec_alt_strategies.file import AdaptiveReadableFile, AdaptiveWritableFile
from fsspec_alt_strategies.testing import SimulatedFileSystem

_KiB = 1024

# Small blocks, so that small files show the same behaviour as large ones.
_BLOCKS = {"min_block_size": 4 * _KiB, "max_block_size": 32 * _KiB}


class _Reader(AdaptiveReadableFile):
    """Fetches from a `SimulatedFileSystem`, recording each range."""

    def __init__(
        self, fs: SimulatedFileSystem, path: str, size: int, loop: Any, **kwargs: Any
    ) -> None:
        super().__init__(path, loop, size, **{**_BLOCKS, **kwargs})
        self.fs = fs
        self.fetched: list[tuple[int, int]] = []
        self.fail = False

    async def _fetch_range(self, start: int, end: int) -> bytes:
        self.fetched.append((start, end))
        if self.fail:
            raise OSError("fetch failed")
        return await self.fs._cat_file(self.path, start=start, end=end)


async def _reader(fs: SimulatedFileSystem, data: bytes, **kwargs: Any) -> _Reader:
    fs.add_file("f.bin", data)
    return _Reader(fs, "f.bin", len(data), asyncio.get_running_loop(), **kwargs)


def _fetched_bytes(reader: _Reader) -> int:
    return sum(end - start for start, end in reader.fetched)


async def test_reads_as_bytesio(flat_fs: SimulatedFileSystem) -> None:
    data = os.urandom(300 * _KiB)
    reference = io.BytesIO(data)
    rng = random.Random(0)
    async with await _reader(flat_fs, data, kept_size=64 * _KiB) as f:
        for _ in range(400):
            op = rng.random()
            if op < 0.3:
                pos = rng.randrange(len(data) + 10)
                assert await f._seek(pos) == reference.seek(pos)
            elif op < 0.35:
                offset = -rng.randrange(1, 100)
                assert await f._seek(offset, io.SEEK_END) == reference.seek(
                    offset, io.SEEK_END
                )
            elif op < 0.38:
                assert await f._read() == reference.read()
            else:
                size = rng.choice([1, 100, 5 * _KiB, 40 * _KiB])
                assert await f._read(size) == reference.read(size)
            assert f._pos == reference.tell()


async def test_sequential_reads_grow_their_blocks(flat_fs: SimulatedFileSystem) -> None:
    data = os.urandom(1024 * _KiB)
    async with await _reader(flat_fs, data) as f:
        chunks = [await f._read(4 * _KiB) for _ in range(256)]
    assert b"".join(chunks) == data
    sizes = [end - start for start, end in f.fetched]
    assert sizes[0] == 4 * _KiB
    assert max(sizes) == 32 * _KiB
    # Far fewer requests than reads, and no byte fetched twice.
    assert len(f.fetched) < len(chunks) / 3
    assert _fetched_bytes(f) == len(data)


async def test_read_to_the_end_splits_under_the_concurrency_limit(
    flat_fs: SimulatedFileSystem,
) -> None:
    flat_fs.latency = 0.001
    data = os.urandom(200 * _KiB)
    async with await _reader(
        flat_fs, data, concurrency_limit=asyncio.Semaphore(4)
    ) as f:
        assert await f._read() == data
    assert len(f.fetched) == 7  # 200 KiB in even parts of at most 32 KiB.
    assert flat_fs.stats.max_concurrency == 4


async def test_scattered_reads_fetch_one_small_block_each(
    flat_fs: SimulatedFileSystem,
) -> None:
    data = os.urandom(1024 * _KiB)
    rng = random.Random(1)
    async with await _reader(flat_fs, data, kept_size=0) as f:
        for _ in range(50):
            pos = rng.randrange(len(data) - 100)
            await f._seek(pos)
            assert await f._read(100) == data[pos : pos + 100]
    assert all(end - start <= 2 * 4 * _KiB for start, end in f.fetched)
    assert _fetched_bytes(f) <= 50 * 2 * 4 * _KiB


async def test_seeking_to_where_it_is_keeps_the_run(
    flat_fs: SimulatedFileSystem,
) -> None:
    # Many readers seek before each read, even when reading in sequence.
    data = os.urandom(512 * _KiB)
    async with await _reader(flat_fs, data) as plain:
        while await plain._read(4 * _KiB):
            pass
    async with await _reader(flat_fs, data) as seeking:
        while True:
            await seeking._seek(seeking.tell())
            if not await seeking._read(4 * _KiB):
                break
    assert seeking.fetched == plain.fetched


def _parquet_like(
    row_groups: int, columns: int, chunk: int
) -> tuple[bytes, list[list[tuple[int, int]]]]:
    """A file laid out as Parquet is: column chunks, then metadata and footer.

    Returns the file, and each row group's column chunks as (offset, length).
    """
    body = io.BytesIO()
    body.write(b"PAR1")
    layout = []
    for _ in range(row_groups):
        group = []
        for _ in range(columns):
            group.append((body.tell(), chunk))
            body.write(os.urandom(chunk))
        layout.append(group)
    metadata = os.urandom(2 * _KiB)
    body.write(metadata)
    body.write(struct.pack("<I", len(metadata)) + b"PAR1")
    return body.getvalue(), layout


async def _read_parquet_like(
    f: _Reader, data: bytes, layout: list[list[tuple[int, int]]], selected: list[int]
) -> None:
    """Read as a Parquet reader does: footer, metadata, then the selected columns."""
    await f._seek(-8, io.SEEK_END)
    footer = await f._read(8)
    (metadata_length,) = struct.unpack("<I", footer[:4])
    await f._seek(-8 - metadata_length, io.SEEK_END)
    assert len(await f._read(metadata_length)) == metadata_length
    for group in layout:
        for column in selected:
            offset, length = group[column]
            await f._seek(offset)
            assert await f._read(length) == data[offset : offset + length]


@pytest.mark.parametrize("chunk", [1 * _KiB, 24 * _KiB, 200 * _KiB])
async def test_parquet_like_reads(flat_fs: SimulatedFileSystem, chunk: int) -> None:
    data, layout = _parquet_like(row_groups=4, columns=8, chunk=chunk)
    selected = [1, 5]
    async with await _reader(flat_fs, data) as f:
        await _read_parquet_like(f, data, layout, selected)
    reads = len(layout) * len(selected)
    # The footer's block holds the metadata too: one request for both.
    footer_start = len(data) - 8 - (len(data) - 8) % (4 * _KiB)
    assert f.fetched[0] == (footer_start, len(data))
    if chunk < 4 * _KiB:
        # Small chunks fetch their aligned blocks, which neighbours share.
        assert len(f.fetched) <= 1 + reads
        return
    # A chunk fetches itself and nothing more, in as few requests as allowed,
    # and nothing ahead: each is a seek away from the last.
    per_chunk = -(-chunk // (32 * _KiB))
    assert len(f.fetched) == 1 + reads * per_chunk
    assert _fetched_bytes(f) == len(data) - footer_start + reads * chunk


async def test_kept_blocks_serve_reads_that_come_back(
    flat_fs: SimulatedFileSystem,
) -> None:
    data = os.urandom(256 * _KiB)
    async with await _reader(flat_fs, data) as f:
        await f._seek(100 * _KiB)
        await f._read(100)
        await f._seek(0)
        await f._read(100)
        requests = len(f.fetched)
        await f._seek(100 * _KiB + 50)
        assert await f._read(100) == data[100 * _KiB + 50 : 100 * _KiB + 150]
        assert len(f.fetched) == requests


async def test_failed_fetch_raises_and_the_next_read_fetches_again(
    flat_fs: SimulatedFileSystem,
) -> None:
    data = os.urandom(64 * _KiB)
    async with await _reader(flat_fs, data) as f:
        f.fail = True
        with pytest.raises(OSError, match="fetch failed"):
            await f._read(100)
        f.fail = False
        assert await f._read(100) == data[:100]


async def test_closing_cancels_fetches_in_flight(flat_fs: SimulatedFileSystem) -> None:
    flat_fs.latency = 0.05
    data = os.urandom(256 * _KiB)
    f = await _reader(flat_fs, data, concurrency_limit=asyncio.Semaphore(8))
    await f._read(4 * _KiB)
    await f._read(64 * _KiB)  # Runs ahead: more blocks are in flight.
    await f.aclose()
    assert flat_fs._in_flight == 0
    with pytest.raises(ValueError, match="closed file"):
        await f._read(1)


def test_sync_methods_run_on_the_loop() -> None:
    fs = SimulatedFileSystem()
    data = os.urandom(64 * _KiB)
    fs.add_file("f.bin", data)
    with _Reader(fs, "f.bin", len(data), fs.loop) as f:
        assert f.read(10) == data[:10]
        buffer = bytearray(20)
        assert f.readinto(buffer) == 20
        assert bytes(buffer) == data[10:30]
        assert f.seek(-5, io.SEEK_END) == len(data) - 5
        assert f.read() == data[-5:]
        assert io.BufferedReader(f).read(1) == b""
    assert f.closed


async def test_sync_methods_refuse_to_run_on_the_loop(
    flat_fs: SimulatedFileSystem,
) -> None:
    async with await _reader(flat_fs, b"abc") as f:
        with pytest.raises(NotImplementedError):
            f.read(1)


class _Writer(AdaptiveWritableFile):
    """Uploads to a `SimulatedFileSystem`, recording each request."""

    def __init__(self, fs: SimulatedFileSystem, path: str, **kwargs: Any) -> None:
        options = {
            "min_block_size": 4 * _KiB,
            "max_block_size": 16 * _KiB,
            "min_multipart_size": 8 * _KiB,
            "concurrency_limit": asyncio.Semaphore(4),
            "spread": 4,
        }
        super().__init__(path, asyncio.get_running_loop(), **{**options, **kwargs})
        self.fs = fs
        self.requests: list[str] = []
        self.parts: dict[int, tuple[int, int, bool, bytes]] = {}
        self.completed: list[Any] | None = None
        self.fail_part: int | None = None
        self.in_flight = 0
        self.max_in_flight = 0

    async def _upload_all(self, data: bytes) -> None:
        self.requests.append("all")
        await self.fs._pipe_file(self.path, data)

    async def _start_multipart_upload(self) -> None:
        self.requests.append("start")

    async def _upload_part(
        self, data: bytes, start: int, end: int, index: int, last: bool
    ) -> Any:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.001)
        finally:
            self.in_flight -= 1
        if index == self.fail_part:
            raise OSError("part failed")
        self.parts[index] = start, end, last, data
        return f"etag-{index}"

    async def _complete_multipart_upload(self, parts: list[Any]) -> None:
        self.requests.append("complete")
        self.completed = parts
        data = b"".join(self.parts[i][3] for i in sorted(self.parts))
        await self.fs._pipe_file(self.path, data)

    async def _abort_multipart_upload(self) -> None:
        self.requests.append("abort")


async def test_small_file_is_uploaded_in_one_request(
    flat_fs: SimulatedFileSystem,
) -> None:
    async with _Writer(flat_fs, "f.bin") as f:
        for _ in range(3):
            await f._write(b"x" * 100)
        assert f.requests == []
    assert f.requests == ["all"]
    assert await flat_fs._cat_file("f.bin") == b"x" * 300


async def test_empty_file_is_created(flat_fs: SimulatedFileSystem) -> None:
    async with _Writer(flat_fs, "f.bin"):
        pass
    assert await flat_fs._cat_file("f.bin") == b""


async def test_large_file_is_uploaded_in_growing_parts(
    flat_fs: SimulatedFileSystem,
) -> None:
    data = os.urandom(300 * _KiB)
    async with _Writer(flat_fs, "f.bin") as f:
        for i in range(0, len(data), 1000):
            await f._write(memoryview(data)[i : i + 1000])
        assert f.tell() == len(data)
    assert f.requests == ["start", "complete"]
    assert await flat_fs._cat_file("f.bin") == data
    parts = [f.parts[i] for i in sorted(f.parts)]
    assert f.completed == [f"etag-{i}" for i in range(len(parts))]
    assert [last for _, _, last, _ in parts] == [False] * (len(parts) - 1) + [True]
    sizes = [end - start for start, end, _, _ in parts]
    assert all(size >= 4 * _KiB for size in sizes[:-1])
    assert sizes[:-1] == sorted(sizes[:-1])
    assert max(sizes) >= 16 * _KiB
    assert f.max_in_flight <= 4


async def test_failed_part_aborts_the_upload(flat_fs: SimulatedFileSystem) -> None:
    f = _Writer(flat_fs, "f.bin")
    f.fail_part = 1
    with pytest.raises(OSError, match="part"):
        for _ in range(100):
            await f._write(b"x" * 1000)
        await f.aclose()
    if not f.closed:
        with pytest.raises(OSError, match="part"):
            await f.aclose()
    assert f.requests[-1] == "abort"
    assert not await flat_fs._exists("f.bin")


def test_the_base_class_is_not_public() -> None:
    # The base class is machinery, and its name would suggest fsspec's own.
    assert file.__all__ == [
        "AdaptiveReadableFile",
        "AdaptiveWritableFile",
        "LocallyBufferedFile",
    ]


async def test_a_shared_concurrency_limit_limits_files_together(
    flat_fs: SimulatedFileSystem,
) -> None:
    flat_fs.latency = 0.002
    data = os.urandom(200 * _KiB)
    flat_fs.add_file("a.bin", data)
    flat_fs.add_file("b.bin", data)
    limit = asyncio.Semaphore(3)
    loop = asyncio.get_running_loop()
    a = _Reader(flat_fs, "a.bin", len(data), loop, concurrency_limit=limit)
    b = _Reader(flat_fs, "b.bin", len(data), loop, concurrency_limit=limit)
    async with a, b:
        assert await asyncio.gather(a._read(), b._read()) == [data, data]
    # Each read alone is 7 parts: together, they still run 3 at a time.
    assert flat_fs.stats.max_concurrency == 3


async def test_a_shared_concurrency_limit_bounds_the_parts_in_flight(
    flat_fs: SimulatedFileSystem,
) -> None:
    limit = asyncio.Semaphore(2)
    writers = [_Writer(flat_fs, f"{i}.bin", concurrency_limit=limit) for i in range(3)]
    in_flight = 0
    most = 0

    def counting(writer: _Writer) -> None:
        upload = writer._upload_part

        async def counted(*args: Any) -> Any:
            nonlocal in_flight, most
            in_flight += 1
            most = max(most, in_flight)
            try:
                return await upload(*args)
            finally:
                in_flight -= 1

        writer._upload_part = counted  # type: ignore[method-assign]

    async def write(writer: _Writer) -> None:
        counting(writer)
        async with writer:
            for _ in range(100):
                await writer._write(b"x" * 1000)

    await asyncio.gather(*(write(w) for w in writers))
    # Writes wait for the concurrency limit: the three files never had more than two
    # parts in flight, so never held more than those in memory.
    assert most == 2
    for i in range(3):
        assert await flat_fs._cat_file(f"{i}.bin") == b"x" * 100_000


async def test_readahead_is_capped(flat_fs: SimulatedFileSystem) -> None:
    data = os.urandom(512 * _KiB)
    ahead = []
    async with await _reader(flat_fs, data, max_readahead=16 * _KiB) as f:
        while await f._read(4 * _KiB):
            if f._requests.queue:
                ahead.append(f._requests.queue[-1].end - f._pos)
    assert max(ahead) <= 16 * _KiB


async def test_spread_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="spread"):
        await _reader(flat_fs, b"abc", spread=0)
