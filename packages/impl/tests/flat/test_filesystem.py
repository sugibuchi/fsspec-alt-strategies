"""Tests of `FlatFileSystem`, checked against fsspec's own methods on the same data."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem, sync
from fsspec.callbacks import Callback

from fsspec_alt_strategies.files.adaptive import (
    AdaptiveReadableFile,
    AdaptiveWritableFile,
)
from fsspec_alt_strategies.files.fsspec_async import AsyncFileAdapter
from fsspec_alt_strategies.flat.filesystem import FlatFileSystem
from fsspec_alt_strategies_common.monitor import Call, Monitor, monitoring
from fsspec_alt_strategies_common.profile import StorageProfile
from fsspec_alt_strategies_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _Reader(AdaptiveReadableFile):
    """A file of the simulated storage, read by byte ranges."""

    def __init__(self, inner: SimulatedFileSystem, path: str, size: int, **kwargs: Any):
        loop = kwargs.pop("loop", None) or asyncio.get_running_loop()
        super().__init__(path, loop, size, **kwargs)
        self.inner = inner

    async def _fetch_range(self, start: int, end: int) -> bytes:
        return await self.inner._cat_file(self.path, start=start, end=end)


class _Writer(AdaptiveWritableFile):
    """A file of the simulated storage, uploaded whole: its files are small."""

    def __init__(self, inner: SimulatedFileSystem, path: str, **kwargs: Any):
        loop = kwargs.pop("loop", None) or asyncio.get_running_loop()
        super().__init__(path, loop, **kwargs)
        self.inner = inner

    async def _upload_all(self, data: bytes) -> None:
        await self.inner._pipe_file(self.path, data)

    async def _start_multipart_upload(self) -> None:
        raise NotImplementedError

    async def _upload_part(
        self, data: bytes, start: int, end: int, index: int, last: bool
    ) -> Any:
        raise NotImplementedError

    async def _complete_multipart_upload(self, parts: list[Any]) -> None:
        raise NotImplementedError

    async def _abort_multipart_upload(self) -> None:
        raise NotImplementedError


async def _open_file(
    inner: SimulatedFileSystem, path: str, mode: str, **kwargs: Any
) -> Any:
    """Open a file, raising here if it is missing, or exists in mode "xb"."""
    if "r" in mode:
        size = (await inner._info(path))["size"]
        return _Reader(inner, path, size, **kwargs)
    if "x" in mode and await inner._exists(path):
        raise FileExistsError(path)
    return _Writer(inner, path, **kwargs)


class _Flat(FlatFileSystem):
    """A FlatFileSystem whose primitives are a simulated object store's."""

    protocol = "flattest"
    cachable = False
    storage_profile = StorageProfile()
    rm_batch_size = 2

    def __init__(self, inner: SimulatedFileSystem, **kwargs: Any) -> None:
        options = {
            "operation_concurrency": 4,
            "io_concurrency": 2,
        }
        super().__init__(**{**options, **kwargs})
        self.inner = inner

    def _ils(self, prefix: str) -> AsyncIterator[dict[str, Any]]:
        return self.inner._ils(prefix)

    async def _rm_files(self, paths: list[str]) -> None:
        await self.inner._rm_files(paths)

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        return await self.inner._ls(path, detail=detail)

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return await self.inner._info(path)

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        await self.inner._rm_file(path)

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        await self.inner._cp_file(path1, path2)

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        # The same file, for the sync API, on the filesystem's loop.
        path = self._strip_protocol(path)
        return sync(
            self.loop,
            _open_file,
            self.inner,
            path,
            mode,
            loop=self.loop,
            concurrency_limit=self.io_limit,
        )

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        path = self._strip_protocol(path)
        file = await _open_file(self.inner, path, mode, concurrency_limit=self.io_limit)
        return AsyncFileAdapter(file)


_FILES = [
    "data/2023/01/part-0.parquet",
    "data/2023/02/part-0.parquet",
    "data/2024/01/part-0.parquet",
    "data/2024/02/part-0.parquet",
    "data/README.md",
]


@pytest.fixture
def inner() -> SimulatedFileSystem:
    fs = SimulatedFileSystem(asynchronous=True)
    for path in _FILES:
        fs.add_file(path, b"x" * 10)
    fs.add_file("database.csv")  # Shares the text "data", not the folder.
    return fs


@pytest.fixture
def fs(inner: SimulatedFileSystem) -> _Flat:
    return _Flat(inner, asynchronous=True)


@pytest.mark.parametrize("kwargs", [{}, {"withdirs": True}, {"maxdepth": 2}])
@pytest.mark.parametrize(
    "path", ["data", "flattest://data/2024", "data/README.md", "missing"]
)
async def test_find_answers_as_fsspec(
    fs: _Flat, inner: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    expected = await AsyncFileSystem._find(inner, fs._strip_protocol(path), **kwargs)
    assert await fs._find(path, **kwargs) == expected


@pytest.mark.parametrize("path", ["data", "data/README.md", "missing"])
async def test_du_answers_as_fsspec(
    fs: _Flat, inner: SimulatedFileSystem, path: str
) -> None:
    assert await fs._du(path) == await AsyncFileSystem._du(inner, path)


@pytest.mark.parametrize(
    "path", ["data/*/01", "data/**/*.parquet", "data/*/", "data/README.md"]
)
async def test_glob_answers_as_fsspec(
    fs: _Flat, inner: SimulatedFileSystem, path: str
) -> None:
    expected = await AsyncFileSystem._glob(inner, path, detail=True)
    assert await fs._glob(path, detail=True) == expected


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"maxdepth": 1}, {"maxdepth": 2}, {"topdown": False}, {"detail": True}],
)
@pytest.mark.parametrize("path", ["data", "data/README.md", "missing"])
async def test_walk_answers_as_fsspec(
    fs: _Flat, inner: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    walked = [t async for t in fs._walk(path, **kwargs)]
    expected = [t async for t in AsyncFileSystem._walk(inner, path, **kwargs)]
    key = lambda t: t[0]  # noqa: E731
    assert sorted(walked, key=key) == sorted(expected, key=key)
    if kwargs.get("topdown") is False and len(walked) > 1:
        assert walked[-1][0] == "data"  # The root comes last.


async def test_walk_lists_once(inner: SimulatedFileSystem) -> None:
    fs = _Flat(inner, asynchronous=True)
    inner.reset_stats()
    assert len([t async for t in fs._walk("data")]) == 1 + 6
    assert inner.stats.calls == {"list": 1}  # fsspec's lists each of the 7 folders.


async def test_recursive_rm_copy_and_mv(fs: _Flat, inner: SimulatedFileSystem) -> None:
    await fs._copy("data", "copied", recursive=True)
    assert sorted(p for p in inner._file_data if p.startswith("copied/")) == [
        "copied" + p[len("data") :] for p in _FILES
    ]
    await fs._mv("copied", "moved", recursive=True)
    assert not any(p.startswith("copied/") for p in inner._file_data)
    inner.reset_stats()
    await fs._rm("moved", recursive=True)
    assert not any(p.startswith("moved/") for p in inner._file_data)
    # 5 keys in batches of 2, as the class sets: 3 deletions.
    assert inner.stats.calls["delete"] == 3


@pytest.mark.parametrize("renames_files", [None, True])
async def test_a_file_moves_as_the_profile_says_when_not_recursive(
    inner: SimulatedFileSystem, renames_files: bool | None
) -> None:
    renamed: list[tuple[str, str]] = []

    class Renaming(_Flat):
        storage_profile = StorageProfile(renames_files=renames_files)

        async def _mv_file(self, path1: str, path2: str, **kwargs: Any) -> None:
            renamed.append((path1, path2))
            await self.inner._cp_file(path1, path2)
            await self.inner._rm_file(path1)

    fs = Renaming(inner, asynchronous=True)
    inner.reset_stats()
    await fs._mv("data/README.md", "moved.md")
    assert renamed == ([("data/README.md", "moved.md")] if renames_files else [])
    assert inner._file_data["moved.md"].data == b"x" * 10
    assert "data/README.md" not in inner._file_data


async def test_data_goes_through_the_files(
    fs: _Flat, inner: SimulatedFileSystem
) -> None:
    assert await fs._cat_file("data/README.md", start=-3) == b"xxx"
    await fs._pipe_file("new.txt", b"hello")
    assert inner._file_data["new.txt"].data == b"hello"


def test_rm_batch_size_must_be_set(inner: SimulatedFileSystem) -> None:
    class Unset(_Flat):
        rm_batch_size = 0

    with pytest.raises(ValueError, match="rm_batch_size"):
        Unset(inner)


async def test_monitors_and_fsspec_callback_watch_the_calls(fs: _Flat) -> None:
    class Counting(Monitor):
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.items = 0

        async def started(self, call: Call, state: Any, op: Any) -> None:
            self.calls.append(call.method)

        async def yielded(self, call: Call, item: Any, state: Any, op: Any) -> None:
            self.items += 1

    monitor = Counting()
    callback = Callback()
    async with monitoring(monitor):
        assert len([t async for t in fs._walk("data")]) == 7
        await fs._rm("data", recursive=True, callback=callback)
    assert monitor.calls == ["walk", "rm"]
    assert callback.value == len(_FILES)
    # The walk's entries, then the files deleted.
    assert monitor.items == len(_FILES) + 6 + len(_FILES)


def test_a_subclass_missing_a_primitive_is_refused() -> None:
    class Incomplete(_Flat):
        _ils = FlatFileSystem._ils  # Still abstract, as if not written.

    with pytest.raises(TypeError, match="_ils"):
        Incomplete(SimulatedFileSystem(), asynchronous=True)


async def test_what_the_strategies_leave_to_fsspec(
    fs: _Flat, inner: SimulatedFileSystem
) -> None:
    expected = await AsyncFileSystem._du(inner, "data", withdirs=True)
    assert await fs._du("data", withdirs=True) == expected
    await fs._copy("data/README.md", "copied.md")
    await fs._copy(["data/README.md", "database.csv"], ["a.md", "b.csv"], True)
    await fs._mv("copied.md", "copied.md")  # Onto itself: nothing to do.
    await fs._mv("copied.md", "moved.md")
    await fs._rm(["moved.md", "a.md", "b.csv"])
    assert sorted(inner._file_data) == [
        "data/2023/01/part-0.parquet",
        "data/2023/02/part-0.parquet",
        "data/2024/01/part-0.parquet",
        "data/2024/02/part-0.parquet",
        "data/README.md",
        "database.csv",
    ]
