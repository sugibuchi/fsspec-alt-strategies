"""Tests of `files.adaptive`, on `SimulatedFileSystem`."""

from __future__ import annotations

import asyncio
import errno
import gc
import io
import os
import random
import struct
import subprocess
import sys
import time
from typing import Any

import pytest

from fsspec_alt_strategies_impl.files import adaptive
from fsspec_alt_strategies_impl.files.adaptive import (
    AdaptiveReadableFile,
    AdaptiveWritableFile,
)
from fsspec_alt_strategies_impl.files.fsspec_async import AsyncFileAdapter
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

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
                assert f.seek(pos) == reference.seek(pos)
            elif op < 0.35:
                offset = -rng.randrange(1, 100)
                assert f.seek(offset, io.SEEK_END) == reference.seek(
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
            f.seek(pos)
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
            seeking.seek(seeking.tell())
            if not await seeking._read(4 * _KiB):
                break
    assert seeking.fetched == plain.fetched


async def test_seeking_away_and_back_before_reading_keeps_the_run(
    flat_fs: SimulatedFileSystem,
) -> None:
    # A seek only moves: the next read settles the blocks, where it reads.
    data = os.urandom(512 * _KiB)
    async with await _reader(flat_fs, data) as plain:
        while await plain._read(4 * _KiB):
            pass
    async with await _reader(flat_fs, data) as seeking:
        while True:
            here = seeking.tell()
            seeking.seek(0, io.SEEK_END)
            seeking.seek(here)
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
    f.seek(-8, io.SEEK_END)
    footer = await f._read(8)
    (metadata_length,) = struct.unpack("<I", footer[:4])
    f.seek(-8 - metadata_length, io.SEEK_END)
    assert len(await f._read(metadata_length)) == metadata_length
    for group in layout:
        for column in selected:
            offset, length = group[column]
            f.seek(offset)
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
        f.seek(100 * _KiB)
        await f._read(100)
        f.seek(0)
        await f._read(100)
        requests = len(f.fetched)
        f.seek(100 * _KiB + 50)
        assert await f._read(100) == data[100 * _KiB + 50 : 100 * _KiB + 150]
        assert len(f.fetched) == requests


async def test_closing_releases_the_kept_blocks(flat_fs: SimulatedFileSystem) -> None:
    data = os.urandom(256 * _KiB)
    f = await _reader(flat_fs, data)
    await f._read(4 * _KiB)
    await f._read(64 * _KiB)  # A run: blocks are kept, and fetched ahead.
    assert f._kept
    await f.aclose()
    # A closed file holds no data, even before it is collected.
    assert not f._kept
    assert f._kept_bytes == 0
    assert not f._requests.queue
    assert f._run == 0


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
        # fsspec's sync raises NotImplementedError, a RuntimeError.
        with pytest.raises(RuntimeError, match="within a running loop"):
            f.read(1)


class _Store:
    """One object of a versioned storage, which another client may replace."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.version = "v1"

    def replace(self, data: bytes) -> None:
        self.data = data
        self.version = f"v{int(self.version[1:]) + 1}"


class _Versioned(AdaptiveReadableFile):
    """Reads a `_Store` by range, asking for the version if ``conditional``."""

    def __init__(self, store: _Store, conditional: bool, **kwargs: Any) -> None:
        super().__init__(
            "f.bin", asyncio.get_running_loop(), len(store.data), **_BLOCKS, **kwargs
        )
        self.store = store
        self.conditional = conditional

    async def _fetch_range(self, start: int, end: int) -> bytes:
        await asyncio.sleep(0)
        store = self.store
        if self.conditional and self.version not in (None, store.version):
            # The storage refuses, e.g. with 412 Precondition Failed.
            raise OSError(errno.ESTALE, "Precondition failed", self.path)
        self._check_version(store.version)
        return store.data[start:end]


@pytest.mark.parametrize("conditional", [False, True])
async def test_reads_one_version(conditional: bool) -> None:
    old, new = os.urandom(256 * _KiB), os.urandom(256 * _KiB)
    store = _Store(old)
    async with _Versioned(store, conditional) as f:
        assert await f._read(8 * _KiB) == old[: 8 * _KiB]
        assert f.version == "v1"  # Set by the first response.
        store.replace(new)
        # Elsewhere, so that it fetches from the new version: never mixed.
        f.seek(200 * _KiB)
        with pytest.raises(OSError) as raised:
            await f._read(4 * _KiB)
        assert raised.value.errno == errno.ESTALE
        # And again: the file keeps to the version it began reading.
        with pytest.raises(OSError) as raised:
            await f._read(4 * _KiB)
        assert raised.value.errno == errno.ESTALE


async def test_reads_the_version_given() -> None:
    store = _Store(b"x" * 10)
    async with _Versioned(store, conditional=False, version="v1") as f:
        assert await f._read() == b"x" * 10
    store.replace(b"y" * 10)
    async with _Versioned(store, conditional=False, version="v1") as f:
        with pytest.raises(OSError, match="from version v1 to v2"):
            await f._read()


class _Unversioned(AdaptiveReadableFile):
    """Reads a `_Store` whose responses name no version, checking its status."""

    def __init__(self, store: _Store, **kwargs: Any) -> None:
        super().__init__(
            "f.bin", asyncio.get_running_loop(), len(store.data), **_BLOCKS, **kwargs
        )
        self.store = store
        self.status_requests = 0
        self.fetches = 0

    async def _fetch_range(self, start: int, end: int) -> bytes:
        self.fetches += 1
        await asyncio.sleep(0)
        return self.store.data[start:end]

    async def _current_version(self) -> str | None:
        self.status_requests += 1
        return f"{self.store.version}:{len(self.store.data)}"


async def test_a_storage_naming_no_version_detects_a_file_replaced() -> None:
    old, new = os.urandom(256 * _KiB), os.urandom(256 * _KiB)
    store = _Store(old)
    async with _Unversioned(store, version="v1:262144") as f:
        assert await f._read(8 * _KiB) == old[: 8 * _KiB]
        assert f.status_requests == f.fetches  # One check per fetch.
        store.replace(new)
        f.seek(200 * _KiB)
        with pytest.raises(OSError, match="from version v1:262144 to v2") as raised:
            await f._read(4 * _KiB)
        assert raised.value.errno == errno.ESTALE


async def test_no_current_version_sends_no_request(
    flat_fs: SimulatedFileSystem,
) -> None:
    f = await _reader(flat_fs, os.urandom(64 * _KiB))
    async with f:
        await f._read()
    assert f.version is None


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


class _Exclusive(_Writer):
    """A storage's conditional write: with exclusive, only where no file is."""

    async def _refuse_if_there(self) -> None:
        if self.exclusive and await self.fs._exists(self.path):
            raise FileExistsError(self.path)

    async def _upload_all(self, data: bytes) -> None:
        await self._refuse_if_there()
        await super()._upload_all(data)

    async def _complete_multipart_upload(self, parts: list[Any]) -> None:
        await self._refuse_if_there()
        await super()._complete_multipart_upload(parts)


@pytest.mark.parametrize("size", [1 * _KiB, 40 * _KiB])
async def test_xb_creates_a_file_only_where_none_is(
    flat_fs: SimulatedFileSystem, size: int
) -> None:
    data = os.urandom(size)
    async with _Exclusive(flat_fs, "new.bin", mode="xb") as f:
        assert (f.mode, f.exclusive) == ("xb", True)
        await f._write(data)
    assert await flat_fs._cat_file("new.bin") == data

    flat_fs.add_file("old.bin", b"old")
    f = _Exclusive(flat_fs, "old.bin", mode="xb")
    await f._write(data)
    with pytest.raises(FileExistsError):
        await f.aclose()
    # The file there is kept, and a multipart upload aborted.
    assert await flat_fs._cat_file("old.bin") == b"old"
    assert ("abort" in f.requests) == (size > 8 * _KiB)


async def test_a_writable_file_takes_only_wb_or_xb(
    flat_fs: SimulatedFileSystem,
) -> None:
    f = _Writer(flat_fs, "f.bin")
    assert (f.mode, f.exclusive) == ("wb", False)
    await f.aclose()
    with pytest.raises(ValueError, match="mode"):
        _Writer(flat_fs, "f.bin", mode="ab")


async def test_sequential_may_be_chosen_when_the_upload_starts(
    flat_fs: SimulatedFileSystem,
) -> None:
    # As a storage choosing its kind of upload in its answer to the start.
    class Chosen(_Writer):
        async def _start_multipart_upload(self) -> None:
            await super()._start_multipart_upload()
            self.sequential = True

    data = os.urandom(100 * _KiB)
    async with Chosen(flat_fs, "f.bin") as f:
        for i in range(0, len(data), 3 * _KiB):
            await f._write(data[i : i + 3 * _KiB])
    assert len(f.parts) > 2
    assert f.max_in_flight == 1
    assert await flat_fs._cat_file("f.bin") == data


@pytest.mark.parametrize(
    ("size", "requests"),
    [
        # Past min_multipart_size, but no bigger than the smallest part: a multipart
        # upload's start and completion would be two requests more, for nothing.
        (4 * _KiB, ["all"]),
        # Bigger, it goes in parts, sent at once.
        (6 * _KiB, ["start", "complete"]),
    ],
)
async def test_a_file_no_bigger_than_a_part_is_uploaded_in_one_request(
    flat_fs: SimulatedFileSystem, size: int, requests: list[str]
) -> None:
    data = b"x" * size
    async with _Writer(flat_fs, "f.bin", min_multipart_size=3 * _KiB) as f:
        await f._write(data)
    assert f.requests == requests
    assert await flat_fs._cat_file("f.bin") == data


async def test_small_file_is_uploaded_in_one_request(
    flat_fs: SimulatedFileSystem,
) -> None:
    async with _Writer(flat_fs, "f.bin") as f:
        for _ in range(3):
            await f._write(b"x" * 100)
        assert f.requests == []
    assert f.requests == ["all"]
    assert await flat_fs._cat_file("f.bin") == b"x" * 300


@pytest.mark.parametrize("size", [300, 40 * _KiB])  # One upload, or parts.
async def test_a_forced_flush_finishes_the_upload(
    flat_fs: SimulatedFileSystem, size: int
) -> None:
    data = os.urandom(size)
    f = _Writer(flat_fs, "f.bin")
    await f._write(data)
    await f._flush()
    assert not await flat_fs._exists("f.bin")  # Unforced: nothing shown.
    await f._flush(force=True)
    # Shown before closing, as fsspec's files do.
    assert await flat_fs._cat_file("f.bin") == data
    with pytest.raises(ValueError, match="only close"):
        await f._write(b"more")
    with pytest.raises(ValueError, match="more than once"):
        await f._flush(force=True)
    requests = list(f.requests)
    await f.aclose()
    assert f.requests == requests  # Nothing more on closing.


async def test_a_failed_forced_flush_aborts_and_closing_sends_nothing(
    flat_fs: SimulatedFileSystem,
) -> None:
    f = _Writer(flat_fs, "f.bin")
    f.fail_part = 1
    await f._write(os.urandom(40 * _KiB))
    with pytest.raises(OSError, match="Failed to upload a part") as raised:
        await f._flush(force=True)
    assert str(raised.value.__cause__) == "part failed"
    await f.aclose()  # Not tried again.
    assert f.requests == ["start", "abort"]
    assert not await flat_fs._exists("f.bin")


def test_a_forced_flush_waits_on_the_loop() -> None:
    fs = SimulatedFileSystem()  # Sync: its loop runs in a thread of its own.

    async def open_file() -> _Writer:
        return _Writer(fs, "f.bin")

    f = asyncio.run_coroutine_threadsafe(open_file(), fs.loop).result()
    f.write(b"x" * 20 * _KiB)
    f.flush(force=True)
    assert fs.cat_file("f.bin") == b"x" * 20 * _KiB
    del f  # Finished: nothing to abort, nor to warn of.
    gc.collect()
    assert fs.cat_file("f.bin") == b"x" * 20 * _KiB


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


class _CappedWriter(_Writer):
    """Records the size of every request uploading bytes."""

    def __init__(self, fs: SimulatedFileSystem, path: str, **kwargs: Any) -> None:
        super().__init__(fs, path, **kwargs)
        self.sizes: list[int] = []

    async def _upload_all(self, data: bytes) -> None:
        self.sizes.append(len(data))
        await super()._upload_all(data)

    async def _upload_part(
        self, data: bytes, start: int, end: int, index: int, last: bool
    ) -> Any:
        self.sizes.append(len(data))
        return await super()._upload_part(data, start, end, index, last)


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize(
    "size", [0, 5 * _KiB, 6 * _KiB, 7 * _KiB, 11 * _KiB, 13 * _KiB, 300 * _KiB]
)
async def test_no_request_exceeds_max_request_size(
    flat_fs: SimulatedFileSystem, size: int, sequential: bool
) -> None:
    data = os.urandom(size)
    async with _CappedWriter(
        flat_fs, "f.bin", max_request_size=6 * _KiB, sequential=sequential
    ) as f:
        for i in range(0, len(data), 1000):
            await f._write(data[i : i + 1000])
    assert await flat_fs._cat_file("f.bin") == data
    assert all(s <= 6 * _KiB for s in f.sizes)
    if size <= 6 * _KiB:
        assert f.requests == ["all"]  # Within the cap: one request.
    else:
        assert f.requests == ["start", "complete"]
        parts = [f.parts[i] for i in sorted(f.parts)]
        assert all(end - start >= 4 * _KiB for start, end, _, _ in parts[:-1])
        assert [last for _, _, last, _ in parts] == [False] * (len(parts) - 1) + [True]


async def test_a_small_file_beyond_max_request_size_goes_in_parts(
    flat_fs: SimulatedFileSystem,
) -> None:
    # Below min_multipart_size, but too big for one request.
    data = os.urandom(7 * _KiB)
    async with _CappedWriter(
        flat_fs, "f.bin", min_multipart_size=64 * _KiB, max_request_size=4 * _KiB
    ) as f:
        await f._write(data)
    assert f.requests == ["start", "complete"]
    assert f.sizes == [4 * _KiB, 3 * _KiB]
    assert await flat_fs._cat_file("f.bin") == data


async def test_max_request_size_must_fit_a_part(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="max_request_size"):
        _Writer(flat_fs, "f.bin", max_request_size=1 * _KiB)


async def test_sequential_parts_go_one_at_a_time_in_order(
    flat_fs: SimulatedFileSystem,
) -> None:
    data = os.urandom(300 * _KiB)
    limit = asyncio.Semaphore(4)  # Shared: the other file goes on meanwhile.
    f = _Writer(flat_fs, "f.bin", concurrency_limit=limit, sequential=True)
    other = _Writer(flat_fs, "g.bin", concurrency_limit=limit)
    async with f, other:
        for i in range(0, len(data), 1000):
            await f._write(data[i : i + 1000])
            await other._write(data[i : i + 1000])
    assert f.max_in_flight == 1
    assert other.max_in_flight > 1
    parts = [f.parts[i] for i in sorted(f.parts)]
    # Each part starts where the one before ended.
    assert [start for start, _, _, _ in parts[1:]] == [
        end for _, end, _, _ in parts[:-1]
    ]
    assert await flat_fs._cat_file("f.bin") == data


class _AlignedWriter(_Writer):
    """Keeps every part but the last a multiple of 4 KiB, as GCS does of 256 KiB."""

    def _block_size(self, target: int) -> int:
        return -(-super()._block_size(target) // (4 * _KiB)) * 4 * _KiB


async def test_subclass_sizes_the_parts(flat_fs: SimulatedFileSystem) -> None:
    f = _AlignedWriter(
        flat_fs, "f.bin", min_block_size=5 * _KiB, max_block_size=27 * _KiB
    )
    data = os.urandom(301 * _KiB)
    async with f:
        for i in range(0, len(data), 1000):
            await f._write(data[i : i + 1000])
    sizes = [end - start for start, end, _, _ in (f.parts[i] for i in sorted(f.parts))]
    assert len(sizes) > 2
    assert all(size % (4 * _KiB) == 0 for size in sizes[:-1])
    assert await flat_fs._cat_file("f.bin") == data


async def test_file_collected_unclosed_aborts_its_upload(
    flat_fs: SimulatedFileSystem,
) -> None:
    f = _Writer(flat_fs, "f.bin")
    requests = f.requests
    for _ in range(20):
        await f._write(b"x" * 1000)
    await asyncio.sleep(0.01)  # The parts end, and with them their hold on f.
    # Collected on its own loop, where closing could not wait for it.
    with pytest.warns(ResourceWarning, match="Unclosed"):
        del f
        gc.collect()
    for _ in range(10):
        await asyncio.sleep(0.001)
    assert requests == ["start", "abort"]
    assert not await flat_fs._exists("f.bin")


def test_file_collected_unclosed_off_its_loop_waits_for_the_abort() -> None:
    fs = SimulatedFileSystem()  # Sync: its loop runs in a thread of its own.

    async def open_file() -> _Writer:
        return _Writer(fs, "f.bin")

    f = asyncio.run_coroutine_threadsafe(open_file(), fs.loop).result()
    requests = f.requests
    f.write(b"x" * 20 * _KiB)
    time.sleep(0.05)  # The parts end, and with them their hold on f.
    with pytest.warns(ResourceWarning, match="Unclosed"):
        del f
        gc.collect()
    assert requests == ["start", "abort"]  # Already: waited for, not scheduled.
    assert not fs.exists("f.bin")


class _FailingAbort(_Writer):
    async def _abort_multipart_upload(self) -> None:
        await super()._abort_multipart_upload()
        raise OSError("abort failed")


async def test_a_failed_abort_of_a_collected_file_is_logged(
    flat_fs: SimulatedFileSystem, caplog: pytest.LogCaptureFixture
) -> None:
    f = _FailingAbort(flat_fs, "f.bin")
    for _ in range(20):
        await f._write(b"x" * 1000)
    await asyncio.sleep(0.01)
    with pytest.warns(ResourceWarning, match="Unclosed"):
        del f
        gc.collect()
    for _ in range(10):
        await asyncio.sleep(0.001)
    # Nobody awaits the abort scheduled: only the log tells of its failure.
    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert "f.bin" in record.getMessage()
    assert isinstance(record.exc_info[1], OSError)  # type: ignore[index]


def test_a_failed_abort_off_the_loop_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fs = SimulatedFileSystem()

    async def open_file() -> _Writer:
        return _FailingAbort(fs, "f.bin")

    f = asyncio.run_coroutine_threadsafe(open_file(), fs.loop).result()
    f.write(b"x" * 20 * _KiB)
    time.sleep(0.05)
    with pytest.warns(ResourceWarning, match="Unclosed"):
        del f
        gc.collect()
    [record] = caplog.records
    assert record.levelname == "WARNING"
    assert "f.bin" in record.getMessage()


_LEFT_OPEN_AT_EXIT = """
from fsspec.asyn import get_loop

from fsspec_alt_strategies_impl.files.adaptive import AdaptiveWritableFile
from fsspec_alt_strategies_impl.files.fsspec_transaction import (
    AdaptiveWritableTransaction,
    TransactionFile,
)


class Writer(AdaptiveWritableFile):
    async def _upload_all(self, data): pass
    async def _start_multipart_upload(self): pass
    async def _upload_part(self, data, start, end, index, last): return index
    async def _complete_multipart_upload(self, parts): pass
    async def _abort_multipart_upload(self): pass


def open_file():
    sizes = {"min_block_size": 1024, "max_block_size": 1024}
    return Writer("f.bin", get_loop(), min_multipart_size=1024, **sizes)


# Module globals: finalized during shutdown, once the loop's thread is frozen.
f = open_file()
f.write(b"x" * 4096)
t = TransactionFile(AdaptiveWritableTransaction(open_file()))
t.write(b"x" * 4096)
"""


def test_files_left_open_at_exit_do_not_hang_it() -> None:
    # The loop's daemon thread is frozen during shutdown, but its loop still
    # looks running: waiting for it would never end.
    done = subprocess.run(
        [sys.executable, "-c", _LEFT_OPEN_AT_EXIT],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert done.returncode == 0, done.stderr.decode()


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


async def test_a_failed_abort_keeps_the_original_error(
    flat_fs: SimulatedFileSystem,
) -> None:
    f = _Writer(flat_fs, "f.bin")
    f.fail_part = 1

    async def fail_abort() -> None:
        raise OSError("abort failed")

    f._abort_multipart_upload = fail_abort  # type: ignore[method-assign]
    await f._write(os.urandom(40 * _KiB))
    with pytest.raises(OSError, match="part") as raised:
        await f.aclose()
    # The error the caller acts on is the first one; the abort's is noted.
    assert any("abort failed" in note for note in raised.value.__notes__)


async def test_a_failed_start_aborts_and_keeps_its_error(
    flat_fs: SimulatedFileSystem,
) -> None:
    class FailedStart(_Writer):
        async def _start_multipart_upload(self) -> None:
            self.requests.append("start")  # A session the storage may keep.
            raise OSError("start failed")

        async def _abort_multipart_upload(self) -> None:
            await super()._abort_multipart_upload()
            raise OSError("abort failed")

    f = FailedStart(flat_fs, "f.bin")
    with pytest.raises(OSError, match="start failed") as raised:
        async with f:
            await f._write(os.urandom(40 * _KiB))
    assert f.requests == ["start", "abort"]
    assert any("abort failed" in note for note in raised.value.__notes__)
    assert not await flat_fs._exists("f.bin")


async def test_a_part_failing_while_closing_raises_as_earlier(
    flat_fs: SimulatedFileSystem,
) -> None:
    f = _Writer(flat_fs, "f.bin")
    f.fail_part = 3  # The last, sent by closing.
    await f._write(os.urandom(40 * _KiB))
    with pytest.raises(OSError, match="Failed to upload a part") as raised:
        await f.aclose()
    assert str(raised.value.__cause__) == "part failed"
    assert f.requests == ["start", "abort"]


@pytest.mark.parametrize("size", [300, 40 * _KiB])  # One upload, or parts.
async def test_leaving_async_with_by_an_exception_uploads_nothing(
    flat_fs: SimulatedFileSystem, size: int
) -> None:
    f = _Writer(flat_fs, "f.bin")
    with pytest.raises(KeyError):
        async with AsyncFileAdapter(f) as a:
            await a.write(os.urandom(size))
            raise KeyError("stop")
    assert f.closed
    assert f.requests == (["start", "abort"] if size > 300 else [])
    assert not await flat_fs._exists("f.bin")


@pytest.mark.parametrize("size", [300, 40 * _KiB])  # One upload, or parts.
def test_leaving_with_by_an_exception_uploads_nothing(size: int) -> None:
    fs = SimulatedFileSystem()  # Sync: its loop runs in a thread of its own.

    async def open_file() -> _Writer:
        return _Writer(fs, "f.bin")

    f = asyncio.run_coroutine_threadsafe(open_file(), fs.loop).result()
    with pytest.raises(KeyError), f:
        f.write(os.urandom(size))
        raise KeyError("stop")
    assert f.closed
    assert f.requests == (["start", "abort"] if size > 300 else [])
    assert not fs.exists("f.bin")
    data = os.urandom(size)
    g = asyncio.run_coroutine_threadsafe(open_file(), fs.loop).result()
    with g:
        g.write(data)
    assert fs.cat_file("f.bin") == data


async def test_a_write_cancelled_between_parts_sends_none_twice(
    flat_fs: SimulatedFileSystem,
) -> None:
    go = asyncio.Event()

    class Held(_Writer):
        async def _upload_part(self, *args: Any) -> Any:
            await go.wait()
            return await super()._upload_part(*args)

    data = os.urandom(40 * _KiB)
    f = Held(flat_fs, "f.bin", concurrency_limit=asyncio.Semaphore(1))
    writing = asyncio.create_task(f._write(data))
    await asyncio.sleep(0.01)  # The first part is sent, the second waits.
    assert len(f._requests.queue) == 1
    writing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writing
    go.set()
    await f.aclose()
    # The part sent stays out of the buffer: the file is uploaded as written.
    assert await flat_fs._cat_file("f.bin") == data


def test_the_base_class_is_not_public() -> None:
    # The base class is machinery, and its name would suggest fsspec's own.
    assert adaptive.__all__ == ["AdaptiveReadableFile", "AdaptiveWritableFile"]


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


async def test_readahead_is_capped_at_spread_largest_blocks(
    flat_fs: SimulatedFileSystem,
) -> None:
    data = os.urandom(512 * _KiB)
    ahead = []
    async with await _reader(flat_fs, data, spread=2, max_block_size=8 * _KiB) as f:
        while await f._read(4 * _KiB):
            if f._requests.queue:
                ahead.append(f._requests.queue[-1].end - f._pos)
    # 2 blocks of 8 KiB: reached, never passed, however long the run.
    assert max(ahead) == 2 * 8 * _KiB


@pytest.mark.parametrize(("kwargs", "limit"), [({}, 8), ({"spread": 3}, 3)])
async def test_a_file_given_no_limit_runs_spread_requests_at_a_time(
    flat_fs: SimulatedFileSystem, kwargs: dict[str, Any], limit: int
) -> None:
    flat_fs.latency = 0.001
    data = os.urandom(512 * _KiB)
    async with await _reader(flat_fs, data, **kwargs) as f:
        assert await f._read() == data
    assert len(f.fetched) == 16  # 512 KiB in parts of 32 KiB.
    assert flat_fs.stats.max_concurrency == limit


async def test_spread_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="spread"):
        await _reader(flat_fs, b"abc", spread=0)


async def test_seek_refuses_what_io_refuses(flat_fs: SimulatedFileSystem) -> None:
    async with await _reader(flat_fs, b"abcdef") as f:
        assert f.seekable()
        assert f.seek(2) == 2
        assert f.seek(1, io.SEEK_CUR) == 3
        with pytest.raises(ValueError, match="whence"):
            f.seek(0, 3)
        with pytest.raises(ValueError, match="Negative"):
            f.seek(-1)
        assert await f._read() == b"def"
