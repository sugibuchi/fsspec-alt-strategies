"""Tests of the families' ``_transfer``, a copy driven by the source's family."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from fsspec_alt_strategies.common.operation import _collect
from fsspec_alt_strategies.flat.wrapper import AsyncFlatWrapper
from fsspec_alt_strategies.tree.wrapper import AsyncTreeWrapper
from fsspec_alt_strategies_common.profile import StorageProfile
from fsspec_alt_strategies_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _Reader:
    """A file of a simulated storage, read whole by one request."""

    def __init__(self, fs: SimulatedFileSystem, path: str) -> None:
        self.fs, self.path, self.done = fs, path, False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    async def read(self, size: int) -> bytes:
        if self.done:
            return b""
        self.done = True
        return await self.fs._cat_file(self.path)


class _Writer:
    """A file of a simulated storage, written on closing by one request."""

    def __init__(self, fs: SimulatedFileSystem, path: str) -> None:
        self.fs, self.path, self.data = fs, path, b""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, *exc: object) -> None:
        if exc_type is None:
            await self.fs._pipe_file(self.path, self.data)

    async def write(self, data: bytes) -> None:
        self.data += data


async def _open(fs: SimulatedFileSystem, path: str, mode: str) -> Any:
    return _Reader(fs, path) if "r" in mode else _Writer(fs, path)


class _Tree(AsyncTreeWrapper):
    """A simulated storage as a `TreeFileSystem`, with files to stream."""

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return await _open(self.fs, path, mode)


class _Flat(AsyncFlatWrapper):
    """A simulated storage as a `FlatFileSystem`, listed by its own ``_ils``."""

    def _ils(self, prefix: str) -> AsyncIterator[dict[str, Any]]:
        return self.fs._ils(prefix)

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return await _open(self.fs, path, mode)


_OPTIONS = {"operation_concurrency": 4, "io_concurrency": 2, "asynchronous": True}


def _tree(fs: SimulatedFileSystem, profile: StorageProfile) -> _Tree:
    return _Tree(fs, storage_profile=profile, **_OPTIONS)


def _flat(fs: SimulatedFileSystem, profile: StorageProfile) -> _Flat:
    return _Flat(fs, rm_batch_size=10, storage_profile=profile, **_OPTIONS)


_FILES = ["data/a/1.txt", "data/a/2.txt", "data/b/3.txt"]


@pytest.mark.parametrize("source", ["tree", "flat"])
@pytest.mark.parametrize("destination", ["tree", "flat"])
async def test_each_family_to_each(source: str, destination: str) -> None:
    src = SimulatedFileSystem(hierarchical=source == "tree", asynchronous=True)
    dst = SimulatedFileSystem(hierarchical=destination == "tree", asynchronous=True)
    for path in _FILES:
        src.add_file(path, path.encode())
    wrap = {"tree": _tree, "flat": _flat}
    # A flat destination needs no folders, and says so.
    profile = StorageProfile(implicit_folders=destination == "flat" or None)
    fs1 = wrap[source](src, StorageProfile())
    fs2 = wrap[destination](dst, profile)

    copied, errors = await _collect(
        fs1._transfer("simulated://data", fs2, "simulated://copied")
    )
    assert errors == []
    assert sorted(copied) == [(p, p.replace("data", "copied", 1)) for p in _FILES]
    for path in _FILES:
        assert dst._file_data[path.replace("data", "copied", 1)].data == path.encode()
    # The tree source lists a folder per request, the flat one the prefix once.
    lists = 3 if source == "tree" else 1
    assert src.stats.calls == {"list": lists, "read": 3}
    mkdirs = {"mkdir": 2} if destination == "tree" else {}
    assert dst.stats.calls == {"info": 1, **mkdirs, "write": 3}


async def test_a_plain_destination_is_unknown() -> None:
    src = SimulatedFileSystem(asynchronous=True)
    dst = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    src.add_file("data/a/1.txt", b"1")
    written: list[tuple[str, str]] = []

    async def copy_file(source: str, destination: str) -> None:
        # e.g. a server-side copy, which a router would plug in.
        written.append((source, destination))
        await dst._pipe_file(destination, src._file_data[source].data)

    fs1 = _flat(src, StorageProfile())
    op = fs1._transfer("data", dst, "copied/", copy_file=copy_file)
    copied, errors = await _collect(op)
    assert errors == []
    assert copied == written == [("data/a/1.txt", "copied/data/a/1.txt")]
    # Unknown, the folder is made first.
    assert dst.stats.calls == {"mkdir": 1, "write": 1}
    assert src.stats.calls == {"list": 1}


async def test_a_sync_destination_is_refused() -> None:
    fs1 = _tree(SimulatedFileSystem(asynchronous=True), StorageProfile())
    with pytest.raises(TypeError, match="async"):
        fs1._transfer("data", MemoryFileSystem(), "copied")  # type: ignore[arg-type]
