"""Tests of `FlatFileSystem`, checked against fsspec's own methods on the same data."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem, sync
from fsspec.callbacks import Callback

from fsspec_alt_strategies.monitor import Call, Monitor, monitoring
from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.files.adaptive import (
    AdaptiveReadableFile,
    AdaptiveWritableFile,
)
from fsspec_alt_strategies_impl.files.fsspec_async import AsyncFileAdapter
from fsspec_alt_strategies_impl.flat.filesystem import FlatFileSystem
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

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

    def _ils(
        self, path: str, recursive: bool = False, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        if recursive:
            return self.inner._list_prefix(f"{path}/{prefix}" if path else prefix)
        else:
            return self._ls_directory(path)

    async def _ls_directory(self, path: str) -> AsyncIterator[dict[str, Any]]:
        for entry in await self.inner._ls(path, detail=True):
            yield entry

    async def _rm_files(self, paths: list[str]) -> None:
        await self.inner._rm_files(paths)

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return await self.inner._info(path)

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        await self.inner._rm_file(path)

    async def _rmdir(self, path: str) -> None:
        await self.inner._rmdir(path)

    async def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        await self.inner._makedirs(path, exist_ok=exist_ok)

    async def _cp_file_entry(
        self, entry: dict[str, Any], path2: str, **kwargs: Any
    ) -> None:
        await self.inner._cp_file(entry["name"], path2)

    async def _mv_file_entry(
        self, entry: dict[str, Any], path2: str, **kwargs: Any
    ) -> None:
        await self.inner._mv_file(entry["name"], path2)

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
    fs.add_file("database.csv")  # Shares the text "data", not the directory.
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
    assert inner.stats.calls == {"list": 1}  # fsspec's lists each of the 7 directories.


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

        async def _mv_file_entry(
            self, entry: dict[str, Any], path2: str, **kwargs: Any
        ) -> None:
            renamed.append((entry["name"], path2))
            await self.inner._cp_file(entry["name"], path2)
            await self.inner._rm_file(entry["name"])

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
    # The walk's directories, then the files deleted.
    assert monitor.items == 7 + len(_FILES)


@pytest.mark.parametrize("primitive", ["_rm_files", "_rmdir", "_makedirs"])
def test_a_subclass_missing_a_primitive_is_refused(primitive: str) -> None:
    # Still abstract, as if not written: fsspec's _rmdir and _makedirs would do nothing.
    incomplete = type(
        "Incomplete", (_Flat,), {primitive: getattr(FlatFileSystem, primitive)}
    )

    with pytest.raises(TypeError, match=primitive):
        incomplete(SimulatedFileSystem(), asynchronous=True)


def test_buckets_are_made_and_deleted_by_the_subclass() -> None:
    inner = SimulatedFileSystem(hierarchical=True)
    fs = _Flat(inner)
    fs.mkdir("bucket")
    assert inner.isdir("bucket")
    with pytest.raises(FileExistsError):
        fs.mkdir("bucket")
    fs.rmdir("bucket")
    assert not inner.exists("bucket")


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


@pytest.mark.parametrize("maxdepth", [1, 2, 3])
async def test_rm_with_maxdepth_deletes_what_fsspecs_does(
    fs: _Flat, inner: SimulatedFileSystem, maxdepth: int
) -> None:
    expanded = await AsyncFileSystem._expand_path(
        inner, "data", recursive=True, maxdepth=maxdepth
    )
    expected = sorted(set(inner._file_data) - set(expanded))
    await fs._rm("data", recursive=True, maxdepth=maxdepth)
    assert sorted(inner._file_data) == expected


async def test_copy_and_mv_with_maxdepth(fs: _Flat, inner: SimulatedFileSystem) -> None:
    inner.add_file("data/2023/top.txt")
    await fs._copy("data", "copied", recursive=True, maxdepth=2)
    assert sorted(p for p in inner._file_data if p.startswith("copied/")) == [
        "copied/2023/top.txt",
        "copied/README.md",
    ]
    await fs._mv("data", "moved", recursive=True, maxdepth=1)
    assert sorted(p for p in inner._file_data if p.startswith("moved/")) == [
        "moved/README.md"
    ]
    # fsspec's own deletes the whole source: the keys deeper are kept here.
    assert "data/2023/top.txt" in inner._file_data


async def test_cp_file_asks_info_then_copies_the_entry(
    fs: _Flat, inner: SimulatedFileSystem
) -> None:
    inner.reset_stats()
    await fs._cp_file("data/README.md", "copied.md")
    # The default _cp_file asks the source's entry once, then hands it on.
    assert inner.stats.calls == {"info": 1, "copy": 1}
    assert inner._file_data["copied.md"].data == b"x" * 10


@pytest.mark.parametrize("path2", ["data/moved.md", "new/moved.md"])
async def test_a_single_file_moves_asking_each_path_once(
    inner: SimulatedFileSystem, path2: str
) -> None:
    class Renaming(_Flat):
        storage_profile = StorageProfile(renames_files=True)

    renaming = Renaming(inner, asynchronous=True)
    inner.reset_stats()
    await renaming._mv("data/README.md", path2)
    # The source's info goes on as its entry: the key is not listed. The
    # simulated object store's _mv_file is a copy, then a deletion, and its
    # _makedirs sends nothing.
    assert inner.stats.calls == {"info": 2, "copy": 1, "delete": 1}
    assert inner._file_data[path2].data == b"x" * 10


@pytest.mark.parametrize("path2", ["data/copied.md", "data/2024/", "new/copied.md"])
async def test_a_single_file_copies_as_fsspecs_asking_each_path_once(
    fs: _Flat, inner: SimulatedFileSystem, path2: str
) -> None:
    inner.reset_stats()
    await fs._copy("data/README.md", path2)
    # Asked once each, unless path2 ends with "/": the key is not listed.
    assert inner.stats.calls["info"] == (1 if path2.endswith("/") else 2)
    assert inner.stats.calls["copy"] == 1
    assert "list" not in inner.stats.calls
    reference = SimulatedFileSystem()
    for path, file in inner._file_data.items():
        if not path.startswith(("data/copied", "data/2024/README", "new/")):
            reference.add_file(path, file.data)
    reference.copy("data/README.md", path2)
    assert {p: f.data for p, f in inner._file_data.items()} == {
        p: f.data for p, f in reference._file_data.items()
    }
