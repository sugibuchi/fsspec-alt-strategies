"""Tests of `TreeFileSystem`, checked against fsspec's own methods on the same data."""

from __future__ import annotations

import asyncio
from typing import IO, TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem
from fsspec.callbacks import Callback

from fsspec_alt_strategies.files.buffered import LocallyBufferedFile
from fsspec_alt_strategies.files.fsspec_async import AsyncFileAdapter
from fsspec_alt_strategies.tree.filesystem import TreeFileSystem
from fsspec_alt_strategies_common.monitor import (
    Call,
    CallCancelled,
    Monitor,
    monitoring,
)
from fsspec_alt_strategies_common.profile import StorageProfile
from fsspec_alt_strategies_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from pathlib import Path


class _Whole(LocallyBufferedFile):
    """A file of the simulated storage, downloaded and uploaded whole."""

    def __init__(self, inner: SimulatedFileSystem, path: str, mode: str, **kwargs: Any):
        loop = kwargs.pop("loop", None) or asyncio.get_running_loop()
        super().__init__(path, loop, mode, **kwargs)
        self.inner = inner

    async def _download(self, target: IO[bytes]) -> None:
        target.write(await self.inner._cat_file(self.path))

    async def _upload(self, source: IO[bytes]) -> None:
        await self.inner._pipe_file(self.path, source.read())

    async def _exists(self) -> bool:
        return await self.inner._exists(self.path)


class _Tree(TreeFileSystem):
    """A TreeFileSystem whose primitives are a simulated storage's."""

    protocol = "treetest"
    cachable = False

    def __init__(self, inner: SimulatedFileSystem, **kwargs: Any) -> None:
        options = {
            "operation_concurrency": 4,
            "io_concurrency": 2,
            "storage_profile": StorageProfile(),
        }
        super().__init__(**{**options, **kwargs})
        self.inner = inner
        # The extra keyword arguments each listing was given, e.g. refresh=True.
        self.ls_kwargs: list[dict[str, Any]] = []

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        self.ls_kwargs.append(kwargs)
        return await self.inner._ls(path, detail=detail)

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return await self.inner._info(path)

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        await self.inner._rm_file(path)

    async def _rmdir(self, path: str) -> None:
        await self.inner._rmdir(path)

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        await self.inner._cp_file(path1, path2)

    async def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        await self.inner._makedirs(path, exist_ok=exist_ok)

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        # The same file, for the sync API, on the filesystem's loop.
        return _Whole(
            self.inner,
            self._strip_protocol(path),
            mode,
            loop=self.loop,
            concurrency_limit=self.io_limit,
        )

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        file = _Whole(
            self.inner,
            self._strip_protocol(path),
            mode,
            concurrency_limit=self.io_limit,
        )
        await file.aload()  # So that a missing file raises here.
        return AsyncFileAdapter(file)


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * 10)
    fs.add_file("data/README.md", b"readme")


@pytest.fixture
def inner() -> SimulatedFileSystem:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    return fs


@pytest.fixture
def fs(inner: SimulatedFileSystem) -> _Tree:
    return _Tree(inner, asynchronous=True)


@pytest.mark.parametrize("kwargs", [{}, {"withdirs": True}, {"maxdepth": 2}])
@pytest.mark.parametrize(
    "path", ["data", "treetest://data/2024", "data/README.md", "missing"]
)
async def test_find_answers_as_fsspec(
    fs: _Tree, inner: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    expected = await AsyncFileSystem._find(inner, fs._strip_protocol(path), **kwargs)
    assert await fs._find(path, **kwargs) == expected


@pytest.mark.parametrize("path", ["data", "data/README.md", "missing"])
async def test_du_answers_as_fsspec(
    fs: _Tree, inner: SimulatedFileSystem, path: str
) -> None:
    assert await fs._du(path) == await AsyncFileSystem._du(inner, path)
    assert await fs._du(path, total=False) == await AsyncFileSystem._du(
        inner, path, total=False
    )


@pytest.mark.parametrize(
    "path", ["data/*/01", "data/**/*.parquet", "data/*/", "data/README.md"]
)
async def test_glob_answers_as_fsspec(
    fs: _Tree, inner: SimulatedFileSystem, path: str
) -> None:
    expected = await AsyncFileSystem._glob(inner, path, detail=True)
    assert await fs._glob(path, detail=True) == expected


@pytest.mark.parametrize(
    "kwargs", [{}, {"maxdepth": 2}, {"topdown": False}, {"detail": True}]
)
@pytest.mark.parametrize("path", ["data", "data/README.md", "missing"])
async def test_walk_answers_as_fsspec(
    fs: _Tree, inner: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    walked = [t async for t in fs._walk(path, **kwargs)]
    expected = [t async for t in AsyncFileSystem._walk(inner, path, **kwargs)]
    key = lambda t: t[0]  # noqa: E731
    assert sorted(walked, key=key) == sorted(expected, key=key)
    if kwargs.get("topdown") is False and len(walked) > 1:
        assert walked[-1][0] == "data"  # The root comes last.


async def _walked(fs: _Tree, path: str, **kwargs: Any) -> Any:
    return [t async for t in fs._walk(path, **kwargs)]


@pytest.mark.parametrize(
    "method",
    [
        _walked,
        lambda fs, path, **kw: fs._find(path, **kw),
        lambda fs, path, **kw: fs._du(path, **kw),
        lambda fs, path, **kw: fs._glob(path + "/*/*", **kw),
    ],
    ids=["walk", "find", "du", "glob"],
)
async def test_extra_keyword_arguments_go_to_every_listing(
    fs: _Tree, method: Any
) -> None:
    # As fsspec's recursive methods pass them on to _ls, e.g. to skip a cache.
    await method(fs, "data", refresh=True)
    assert fs.ls_kwargs
    assert all(kwargs == {"refresh": True} for kwargs in fs.ls_kwargs)


class _Forbidding(SimulatedFileSystem):
    """Refuses to list one folder, as a storage without permission would."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


@pytest.mark.parametrize("on_error", ["omit", "callable", "raise"])
async def test_walk_handles_a_failed_folder_as_fsspec(on_error: str) -> None:
    inner = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(inner)
    fs = _Tree(inner, asynchronous=True)
    seen: list[Exception] = []
    handler: Any = seen.append if on_error == "callable" else on_error
    if on_error == "raise":
        with pytest.raises(PermissionError):
            [t async for t in fs._walk("data", on_error=handler)]
        return
    walked = [t async for t in fs._walk("data", on_error=handler)]
    fsspec_seen: list[Exception] = []
    fsspec_handler: Any = fsspec_seen.append if on_error == "callable" else on_error
    expected = [
        t async for t in AsyncFileSystem._walk(inner, "data", on_error=fsspec_handler)
    ]
    # The failed folder too, yielded empty.
    assert sorted(walked) == sorted(expected)
    assert ("data/2023", [], []) in walked
    # Unlike fsspec's, which passes on_error to the root's listing only, as its
    # walk recurses without it: here every folder's failure reaches it.
    assert [type(e) for e in seen] == (
        [PermissionError] if on_error == "callable" else []
    )
    assert fsspec_seen == []


async def test_recursive_rm_copy_and_mv(fs: _Tree, inner: SimulatedFileSystem) -> None:
    before = {p: f.data for p, f in inner._file_data.items()}
    await fs._copy("data", "copied", recursive=True)
    assert {
        p.replace("copied", "data", 1): inner._file_data[p].data
        for p in inner._file_data
        if p.startswith("copied/")
    } == before
    await fs._mv("copied", "moved", recursive=True)
    assert not await inner._exists("copied")
    assert sorted(p for p in inner._file_data if p.startswith("moved/")) == sorted(
        "moved" + p[len("data") :] for p in before
    )
    await fs._rm("moved", recursive=True)
    assert not await inner._exists("moved")


async def test_a_rename_moves_a_folder_whole(inner: SimulatedFileSystem) -> None:
    class Renaming(_Tree):
        async def _mv_file(self, path1: str, path2: str, **kwargs: Any) -> None:
            await self.inner._rename(path1, path2)

    profile = StorageProfile(renames_folders=True)
    renaming = Renaming(inner, storage_profile=profile, asynchronous=True)
    inner.reset_stats()
    await renaming._mv("data", "moved", recursive=True)
    assert inner.stats.calls == {"info": 1, "rename": 1}
    assert not await inner._exists("data")
    assert await inner._exists("moved/2024/01/part-0.parquet")


async def test_failures_are_raised(fs: _Tree, inner: SimulatedFileSystem) -> None:
    async def refused(path: str, **kwargs: Any) -> None:
        if path == "data/README.md":
            raise PermissionError(path)
        await inner._rm_file(path)

    fs._rm_file = refused  # type: ignore[method-assign]
    with pytest.raises(PermissionError):
        await fs._rm("data", recursive=True)
    assert await inner._exists("data/README.md")


async def test_data_goes_through_the_files(
    fs: _Tree, inner: SimulatedFileSystem, tmp_path: Path
) -> None:
    assert await fs._cat_file("data/README.md") == b"readme"
    assert await fs._cat_file("data/README.md", start=2, end=-1) == b"adm"
    await fs._pipe_file("new.txt", b"hello")
    assert inner._file_data["new.txt"].data == b"hello"
    with pytest.raises(FileExistsError):
        await fs._pipe_file("new.txt", b"again", mode="create")
    local = tmp_path / "local.txt"
    await fs._get_file("data/README.md", str(local))
    assert local.read_bytes() == b"readme"
    await fs._put_file(str(local), "put.txt")
    assert inner._file_data["put.txt"].data == b"readme"


async def test_operations_keep_to_their_limit() -> None:
    inner = SimulatedFileSystem(hierarchical=True, latency=0.002, asynchronous=True)
    for i in range(10):
        inner.add_file(f"root/{i}/file")
    fs = _Tree(inner, operation_concurrency=3, asynchronous=True)
    assert len(await fs._find("root")) == 10
    assert inner.stats.max_concurrency == 3


def test_the_sync_api_works() -> None:
    inner = SimulatedFileSystem(hierarchical=True)  # Sync, on fsspec's loop.
    _tree(inner)
    fs = _Tree(inner)
    assert fs.cat_file("data/README.md") == b"readme"
    fs.mv("data", "moved", recursive=True)
    assert fs.find("moved") == sorted(
        "moved" + p[len("data") :]
        for p in [
            "data/2023/01/part-0.parquet",
            "data/2023/02/part-0.parquet",
            "data/2024/01/part-0.parquet",
            "data/2024/02/part-0.parquet",
            "data/README.md",
        ]
    )


def test_concurrencies_must_be_positive(inner: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="io_concurrency"):
        _Tree(inner, io_concurrency=0)


class _Recording(Monitor):
    """Records what it is told, as (event, method, detail)."""

    def __init__(self, name: str = "", cancel_after: int | None = None) -> None:
        self.name = name
        self.cancel_after = cancel_after
        self.events: list[tuple[str, str, Any]] = []

    async def started(self, call: Call, state: Any, op: Any) -> None:
        self.events.append(("started", call.method, self.name))

    async def yielded(self, call: Call, item: Any, state: Any, op: Any) -> None:
        self.events.append(("yielded", call.method, item))
        if self.cancel_after is not None and len(self.events) > self.cancel_after:
            op.cancel()

    async def ended(
        self, call: Call, error: BaseException | None, state: Any, op: Any
    ) -> None:
        self.events.append(("ended", call.method, type(error)))


async def test_a_monitor_watches_the_calls_within_its_block(
    fs: _Tree, inner: SimulatedFileSystem
) -> None:
    await fs._rm("data/2023", recursive=True)  # Before the block: not watched.
    monitor = _Recording()
    async with monitoring(monitor):
        await fs._rm("data/2024", recursive=True)
    await fs._rm("data/README.md", recursive=True)  # After it: not watched.
    assert monitor.events[0] == ("started", "rm", "")
    assert monitor.events[-1] == ("ended", "rm", type(None))
    deleted = {item for event, _, item in monitor.events if event == "yielded"}
    assert {"data/2024/01/part-0.parquet", "data/2024"} <= deleted


async def test_monitors_nest_the_outer_first(fs: _Tree) -> None:
    outer, inner_monitor = _Recording("outer"), _Recording("inner")
    with monitoring(outer):
        await fs._find("data")
        with monitoring(inner_monitor):
            await fs._du("data")
    assert [e for e in outer.events if e[0] == "started"] == [
        ("started", "find", "outer"),
        ("started", "du", "outer"),
    ]
    assert [e for e in inner_monitor.events if e[0] == "started"] == [
        ("started", "du", "inner")
    ]


async def test_a_monitor_cancelling_ends_the_call(
    fs: _Tree, inner: SimulatedFileSystem
) -> None:
    monitor = _Recording(cancel_after=1)
    with monitoring(monitor), pytest.raises(CallCancelled) as raised:
        await fs._copy("data", "copied", recursive=True)
    assert raised.value.call == Call(fs, "copy", "data", "copied")
    assert monitor.events[-1] == ("ended", "copy", CallCancelled)
    # Only what was in flight was copied, not all five files.
    assert 0 < raised.value.state.files_copied < 5


async def test_a_monitor_sees_a_walk_stopped_early(fs: _Tree) -> None:
    monitor = _Recording()
    with monitoring(monitor):
        walk = fs._walk("data")
        await anext(walk)
        await walk.aclose()
    assert monitor.events[-1] == ("ended", "walk", type(None))


def test_a_monitor_watches_the_sync_api() -> None:
    inner = SimulatedFileSystem(hierarchical=True)  # Sync, on fsspec's loop.
    _tree(inner)
    fs = _Tree(inner)
    monitor = _Recording()
    # fsspec runs the call on its own loop, in a copy of this context.
    with monitoring(monitor):
        fs.mv("data", "moved", recursive=True)
    assert monitor.events[0] == ("started", "mv", "")
    assert monitor.events[-1] == ("ended", "mv", type(None))


async def test_fsspec_callback_counts_what_was_done(fs: _Tree) -> None:
    callback = Callback()
    await fs._copy("data", "copied", recursive=True, callback=callback)
    assert callback.value == 5  # The files copied.
    callback = Callback()
    await fs._rm("copied", recursive=True, callback=callback)
    assert callback.value == 5 + 7  # The files and the folders deleted.


async def test_fsspec_callback_counts_the_bytes_of_a_file(
    fs: _Tree, tmp_path: Path
) -> None:
    local = tmp_path / "local.txt"
    callback = Callback()
    await fs._get_file("data/README.md", str(local), callback=callback)
    assert (callback.size, callback.value) == (6, 6)
    callback = Callback()
    await fs._put_file(str(local), "put.txt", callback=callback)
    assert (callback.size, callback.value) == (6, 6)


def test_a_subclass_missing_a_primitive_is_refused() -> None:
    class Incomplete(_Tree):
        _rmdir = TreeFileSystem._rmdir  # Still abstract, as if not written.

    with pytest.raises(TypeError, match="_rmdir"):
        Incomplete(SimulatedFileSystem(), asynchronous=True)
