"""Tests of `files.fsspec_transaction`, alone and in fsspec's own transactions."""

from __future__ import annotations

import asyncio
import gc
import io
import time
from typing import IO, Any

import pytest

from fsspec_alt_strategies.files.adaptive import AdaptiveWritableFile
from fsspec_alt_strategies.files.buffered import LocallyBufferedFile
from fsspec_alt_strategies.files.fsspec_transaction import (
    AdaptiveWritableTransaction,
    FileTransaction,
    LocalBufferTransaction,
    LocallyBufferedTransaction,
    TempPathTransaction,
    TransactionFile,
)
from fsspec_alt_strategies_testing import SimulatedFileSystem

_KiB = 1024


class _Parts(AdaptiveWritableFile):
    """Writes to a `SimulatedFileSystem` in parts, visible only once completed."""

    def __init__(self, fs: SimulatedFileSystem, path: str) -> None:
        super().__init__(
            path,
            fs.loop,
            min_block_size=4 * _KiB,
            max_block_size=8 * _KiB,
            min_multipart_size=4 * _KiB,
        )
        self.fs = fs
        self.parts: dict[int, bytes] = {}
        self.aborted = False

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
        self.aborted = True


class _Whole(LocallyBufferedFile):
    """A file of a `SimulatedFileSystem` read and written whole."""

    def __init__(self, fs: SimulatedFileSystem, path: str, mode: str) -> None:
        super().__init__(path, fs.loop, mode)
        self.fs = fs

    async def _download(self, target: IO[bytes]) -> None:
        target.write(await self.fs._cat_file(self.path))

    async def _upload(self, source: IO[bytes]) -> None:
        await self.fs._pipe_file(self.path, source.read())

    async def _exists(self) -> bool:
        return await self.fs._exists(self.path)


@pytest.fixture
def fs() -> SimulatedFileSystem:
    # A sync filesystem: the wrappers' methods wait on its loop.
    return SimulatedFileSystem(hierarchical=True)


_BIG = b"x" * (20 * _KiB)  # Uploaded in parts.


def _temp(fs: SimulatedFileSystem, path: str) -> TransactionFile:
    t = TempPathTransaction(path, lambda p: _Parts(fs, p), fs.mv, fs.rm_file)
    return TransactionFile(t)


def test_temp_path_is_renamed_on_commit(fs: SimulatedFileSystem) -> None:
    f = _temp(fs, "data/f.bin")
    temp_path = f.transaction.temp_path
    f.write(_BIG)
    f.close()
    # Written in full, but at a hidden path beside it.
    assert not fs.exists("data/f.bin")
    assert fs.cat_file(temp_path) == _BIG
    assert temp_path.startswith("data/.f.bin.")
    f.commit()
    assert fs.cat_file("data/f.bin") == _BIG
    assert not fs.exists(temp_path)


def test_temp_path_is_deleted_on_discard(fs: SimulatedFileSystem) -> None:
    f = _temp(fs, "f.bin")
    f.write(b"abc")
    f.discard()
    assert fs.ls("", detail=False) == []
    # Discarded before a write, the temporary file never existed.
    _temp(fs, "g.bin").discard()


def test_temp_path_uses_the_rename_given(fs: SimulatedFileSystem) -> None:
    renamed = []

    def rename(source: str, destination: str) -> None:
        renamed.append((source, destination))
        fs.mv(source, destination)

    t = TempPathTransaction("f.bin", lambda p: _Parts(fs, p), rename, fs.rm_file)
    f = TransactionFile(t)
    f.write(b"abc")
    f.commit()  # Closes it first.
    assert renamed == [(f.transaction.temp_path, "f.bin")]
    assert fs.cat_file("f.bin") == b"abc"


def test_local_buffer_sends_nothing_before_commit(fs: SimulatedFileSystem) -> None:
    f = TransactionFile(LocalBufferTransaction("f.bin", lambda p: _Parts(fs, p)))
    f.write(_BIG)
    f.seek(0)
    assert f.read(3) == b"xxx"  # The buffer's, not the file's.
    f.close()
    assert fs.stats.calls == {}
    f.commit()
    assert fs.cat_file("f.bin") == _BIG

    fs.reset_stats()
    g = TransactionFile(LocalBufferTransaction("g.bin", lambda p: _Parts(fs, p)))
    g.write(b"abc")
    g.discard()
    assert fs.stats.calls == {}


def test_local_buffer_appends_in_the_mode_given(fs: SimulatedFileSystem) -> None:
    fs.pipe_file("f.bin", b"old,")
    t = LocalBufferTransaction("f.bin", lambda p: _Whole(fs, p, "ab"))
    f = TransactionFile(t)
    f.write(b"new")
    f.commit()
    assert fs.cat_file("f.bin") == b"old,new"


class _Resumable(_Parts):
    """Uploads parts in order, the last one making the file visible, as GCS does."""

    sequential = True

    async def _upload_part(
        self, data: bytes, start: int, end: int, index: int, last: bool
    ) -> Any:
        self.parts[index] = data
        if last:
            await self.fs._pipe_file(
                self.path, b"".join(self.parts[i] for i in sorted(self.parts))
            )
        return index

    async def _complete_multipart_upload(self, parts: list[Any]) -> None:
        self.completed = parts  # The last part completed the upload.


@pytest.mark.parametrize("commit", [True, False])
def test_adaptive_holds_back_a_last_part_that_publishes(
    fs: SimulatedFileSystem, commit: bool
) -> None:
    file = _Resumable(fs, "f.bin")
    f = TransactionFile(AdaptiveWritableTransaction(file, hold_last_part=True))
    f.write(_BIG)
    f.close()
    # All parts but the last are uploaded, and nothing is visible.
    assert 0 < len(file.parts) < 5
    assert not fs.exists("f.bin")
    if commit:
        f.commit()
        assert fs.cat_file("f.bin") == _BIG
        assert file.completed == list(range(len(file.completed)))
    else:
        f.discard()
        assert file.aborted
        assert not fs.exists("f.bin")


def test_adaptive_completes_the_upload_on_commit(fs: SimulatedFileSystem) -> None:
    file = _Parts(fs, "f.bin")
    f = TransactionFile(AdaptiveWritableTransaction(file))
    f.write(_BIG)
    f.close()
    # The parts are uploaded, but the upload is not completed.
    assert file.parts
    assert not fs.exists("f.bin")
    f.commit()
    assert fs.cat_file("f.bin") == _BIG


def test_adaptive_uploads_a_small_file_on_commit(fs: SimulatedFileSystem) -> None:
    f = TransactionFile(AdaptiveWritableTransaction(_Parts(fs, "f.bin")))
    f.write(b"abc")
    f.close()
    assert fs.stats.calls == {}
    f.commit()
    assert fs.cat_file("f.bin") == b"abc"


@pytest.mark.parametrize("closed", [True, False])
@pytest.mark.parametrize("data", [_BIG, b"abc"])
def test_adaptive_discard_leaves_no_file(
    fs: SimulatedFileSystem, data: bytes, closed: bool
) -> None:
    file = _Parts(fs, "f.bin")
    f = TransactionFile(AdaptiveWritableTransaction(file))
    f.write(data)
    if closed:
        f.close()
    f.discard()
    assert file.closed
    assert file.aborted == (len(data) > 4 * _KiB)
    assert not fs.exists("f.bin")


def test_locally_buffered_uploads_on_commit(fs: SimulatedFileSystem) -> None:
    f = TransactionFile(LocallyBufferedTransaction(_Whole(fs, "f.bin", "wb")))
    f.write(b"abc")
    f.close()
    assert fs.stats.calls == {}
    f.commit()
    assert fs.cat_file("f.bin") == b"abc"

    fs.reset_stats()
    file = _Whole(fs, "g.bin", "wb")
    g = TransactionFile(LocallyBufferedTransaction(file))
    g.write(b"abc")
    g.close()
    g.discard()
    assert file.closed
    assert fs.stats.calls == {}


def test_locally_buffered_left_undecided_uploads_nothing(
    fs: SimulatedFileSystem,
) -> None:
    file = _Whole(fs, "f.bin", "wb")
    f = TransactionFile(LocallyBufferedTransaction(file))
    f.write(b"abc")
    f.close()
    with pytest.warns(ResourceWarning, match="Undecided"):
        del f
        gc.collect()
    assert file.closed
    assert fs.stats.calls == {}


@pytest.mark.parametrize("closed", [True, False])
def test_undecided_files_are_discarded_when_collected(
    fs: SimulatedFileSystem, closed: bool
) -> None:
    file = _Parts(fs, "f.bin")
    f = TransactionFile(AdaptiveWritableTransaction(file))
    f.write(_BIG)
    if closed:
        f.close()
    with pytest.warns(ResourceWarning, match="Undecided"):
        del f
        gc.collect()
    assert file.closed
    assert file.aborted
    assert not fs.exists("f.bin")

    g = _temp(fs, "g.bin")
    g.write(b"abc")
    if closed:
        g.close()
    with pytest.warns(ResourceWarning, match="Undecided"):
        del g
        gc.collect()
    assert fs.ls("", detail=False) == []  # Not even the temporary file.


def test_undecided_file_collected_on_its_loop_aborts_later(
    fs: SimulatedFileSystem,
) -> None:
    file = _Parts(fs, "f.bin")
    files = [TransactionFile(AdaptiveWritableTransaction(file))]
    files[0].write(_BIG)
    files[0].close()

    async def collect() -> None:
        # Where discard cannot wait: the file aborts its upload later.
        with pytest.warns(ResourceWarning, match="Undecided"):
            files.clear()
            gc.collect()

    asyncio.run_coroutine_threadsafe(collect(), fs.loop).result()
    for _ in range(100):
        if file.aborted:
            break
        time.sleep(0.01)
    assert file.aborted
    assert not fs.exists("f.bin")


def test_temp_path_collected_on_the_loop_is_deleted_later(
    fs: SimulatedFileSystem,
) -> None:
    files = [_temp(fs, "f.bin")]
    files[0].write(b"abc")
    files[0].close()
    assert fs.ls("", detail=False) != []

    async def collect() -> None:
        # Where the sync delete cannot wait: it runs on a thread of its own.
        with pytest.warns(ResourceWarning, match="Undecided"):
            files.clear()
            gc.collect()

    asyncio.run_coroutine_threadsafe(collect(), fs.loop).result()
    for _ in range(100):
        if not fs.ls("", detail=False):
            break
        time.sleep(0.01)
    assert fs.ls("", detail=False) == []


class _SDKWriter:
    """An SDK's file-like object: only write and close, write returning nothing."""

    def __init__(self, fs: SimulatedFileSystem, path: str) -> None:
        self.fs = fs
        self.path = path
        self.data = bytearray()

    def write(self, data: bytes) -> None:
        self.data += data

    def close(self) -> None:
        self.fs.pipe_file(self.path, bytes(self.data))


@pytest.mark.parametrize("strategy", ["temp", "buffer"])
def test_an_sdk_file_like_object(fs: SimulatedFileSystem, strategy: str) -> None:
    def open_file(p: str) -> Any:
        return _SDKWriter(fs, p)

    if strategy == "temp":
        t: FileTransaction = TempPathTransaction("f.bin", open_file, fs.mv, fs.rm_file)
    else:
        t = LocalBufferTransaction("f.bin", open_file)
    f = TransactionFile(t)
    assert f.writable()
    assert f.write(b"abc") == 3
    if strategy == "temp":  # Written to the object itself, which cannot.
        with pytest.raises(io.UnsupportedOperation):
            f.truncate()
    f.close()
    assert not fs.exists("f.bin")
    f.commit()
    assert fs.cat_file("f.bin") == b"abc"


def test_an_exception_in_the_block_discards() -> None:
    t = _Recorded("f.bin", io.BytesIO())
    with pytest.raises(RuntimeError), TransactionFile(t) as f:
        f.write(b"abc")
        raise RuntimeError("failed midway")
    assert t.steps == ["start_staging", "discard"]
    with pytest.raises(ValueError, match="discarded"):
        f.commit()


def test_commit_and_discard_settle_once() -> None:
    t = _Recorded("f.bin", io.BytesIO())
    f = TransactionFile(t)
    f.write(b"abc")
    f.commit()
    f.commit()  # Already committed: nothing more.
    assert t.steps == ["start_staging", "end_staging", "commit"]
    with pytest.raises(ValueError, match="committed"):
        f.discard()
    with pytest.raises(ValueError, match="closed"):
        f.write(b"more")


class _Failing(io.BytesIO):
    """A staged file whose closing fails."""

    def close(self) -> None:
        super().close()
        raise OSError("the upload failed")


class _Recorded(FileTransaction):
    """A storage's own transaction, recording its steps."""

    def __init__(self, path: str, file: io.IOBase) -> None:
        super().__init__(path)
        self.file = file
        self.steps: list[str] = []

    def start_staging(self) -> io.IOBase:
        self.steps.append("start_staging")
        return self.file

    def end_staging(self) -> None:
        self.steps.append("end_staging")
        self.file.close()

    def commit(self) -> None:
        self.steps.append("commit")

    def discard(self) -> None:
        self.steps.append("discard")


def test_a_transaction_of_its_own() -> None:
    t = _Recorded("f.bin", io.BytesIO())
    f = TransactionFile(t)
    assert t.steps == ["start_staging"]
    f.write(b"abc")
    f.commit()
    assert t.steps == ["start_staging", "end_staging", "commit"]
    assert t.file.closed

    t = _Recorded("g.bin", io.BytesIO())
    TransactionFile(t).discard()
    assert t.steps == ["start_staging", "discard"]


def test_a_failed_close_is_never_committed() -> None:
    t = _Recorded("f.bin", _Failing())
    f = TransactionFile(t)
    with pytest.raises(OSError, match="upload failed"):
        f.close()
    with pytest.raises(ValueError, match="failed"):
        f.commit()
    f.discard()
    assert t.steps == ["start_staging", "end_staging", "discard"]


class _TransactionalFS(SimulatedFileSystem):
    """Opens files for writing in the strategy set, inside fsspec's transactions."""

    strategy = "buffer"

    def _open(
        self, path: str, mode: str = "rb", autocommit: bool = True, **kwargs: Any
    ) -> Any:
        path = self._strip_protocol(path)
        if autocommit:
            return _Parts(self, path)
        if self.strategy == "temp":
            return _temp(self, path)
        if self.strategy == "adaptive":
            return TransactionFile(AdaptiveWritableTransaction(_Parts(self, path)))
        if self.strategy == "buffered":
            t = LocallyBufferedTransaction(_Whole(self, path, "wb"))
            return TransactionFile(t)
        t = LocalBufferTransaction(path, lambda p: _Parts(self, p))
        return TransactionFile(t)


@pytest.mark.parametrize("strategy", ["temp", "buffer", "adaptive", "buffered"])
def test_in_fsspec_transactions(strategy: str) -> None:
    fs = _TransactionalFS(hierarchical=True)
    fs.strategy = strategy
    with fs.transaction:
        for name in ("a", "b"):
            with fs.open(f"out/{name}.bin", "wb") as f:
                f.write(_BIG)
        # Nothing is visible before the transaction ends.
        assert not fs.exists("out/a.bin")
    assert fs.cat_file("out/a.bin") == fs.cat_file("out/b.bin") == _BIG

    with pytest.raises(RuntimeError), fs.transaction:
        with fs.open("out/c.bin", "wb") as f:
            f.write(_BIG)
        raise RuntimeError("the transaction failed")
    assert fs.ls("out", detail=False) == ["out/a.bin", "out/b.bin"]

    with fs.open("out/d.bin", "wb") as f:  # No transaction: written at once.
        f.write(b"d")
    assert fs.cat_file("out/d.bin") == b"d"
