"""Tests of `TreeFileSystem`, checked against fsspec's own methods on the same data."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem, sync
from fsspec.callbacks import Callback

from fsspec_alt_strategies.monitor import (
    Call,
    CallCancelled,
    Monitor,
    monitoring,
)
from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.files.adaptive import (
    AdaptiveReadableFile,
    AdaptiveWritableFile,
)
from fsspec_alt_strategies_impl.files.fsspec_async import AsyncFileAdapter
from fsspec_alt_strategies_impl.tree.filesystem import TreeFileSystem
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


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


class _Tree(TreeFileSystem):
    """A TreeFileSystem whose primitives are a simulated storage's."""

    protocol = "treetest"
    cachable = False
    storage_profile = StorageProfile()

    def __init__(self, inner: SimulatedFileSystem, **kwargs: Any) -> None:
        options = {
            "operation_concurrency": 4,
            "io_concurrency": 2,
        }
        super().__init__(**{**options, **kwargs})
        self.inner = inner
        # The extra keyword arguments each listing was given, e.g. refresh=True.
        self.ls_kwargs: list[dict[str, Any]] = []

    async def _ils(self, path: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        self.ls_kwargs.append(kwargs)
        for entry in await self.inner._ls(path, detail=True):
            yield entry

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
    """Refuses to list one directory, as a storage without permission would."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


@pytest.mark.parametrize("on_error", ["omit", "callable", "raise"])
async def test_walk_handles_a_failed_dir_as_fsspec(on_error: str) -> None:
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
    # The failed directory too, yielded empty.
    assert sorted(walked) == sorted(expected)
    assert ("data/2023", [], []) in walked
    # Unlike fsspec's, which passes on_error to the root's listing only, as its
    # walk recurses without it: here every directory's failure reaches it.
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


async def test_a_rename_moves_a_dir_whole(inner: SimulatedFileSystem) -> None:
    class Renaming(_Tree):
        storage_profile = StorageProfile(renames_directories=True)

        async def _mv_file(self, path1: str, path2: str, **kwargs: Any) -> None:
            await self.inner._rename(path1, path2)

    renaming = Renaming(inner, asynchronous=True)
    inner.reset_stats()
    await renaming._mv("data", "moved", recursive=True)
    assert inner.stats.calls == {"info": 1, "rename": 1}
    assert not await inner._exists("data")
    assert await inner._exists("moved/2024/01/part-0.parquet")


@pytest.mark.parametrize("path2", ["moved.md", "data/2024/"])
async def test_a_file_moves_by_mv_file_when_not_recursive(
    inner: SimulatedFileSystem, path2: str
) -> None:
    class Renaming(_Tree):
        async def _mv_file(self, path1: str, path2: str, **kwargs: Any) -> None:
            await self.inner._rename(path1, path2)

    renaming = Renaming(inner, asynchronous=True)
    inner.reset_stats()
    await renaming._mv("data/README.md", path2)
    # One rename, no copy or deletion: fsspec's own would copy, then delete.
    assert inner.stats.calls.get("rename") == 1
    assert "copy" not in inner.stats.calls
    assert "delete" not in inner.stats.calls
    target = "moved.md" if path2 == "moved.md" else "data/2024/README.md"
    assert inner._file_data[target].data == b"readme"
    assert not await inner._exists("data/README.md")


async def test_rm_uses_a_recursive_delete_of_the_storage(
    inner: SimulatedFileSystem,
) -> None:
    class Deleting(_Tree):
        storage_profile = StorageProfile(deletes_directories=True)

        async def _rm_tree(self, path: str) -> None:
            await self.inner._rm(path, recursive=True)

    fs = Deleting(inner, asynchronous=True)
    callback = Callback()
    monitor = _Recording()
    inner.reset_stats()
    with monitoring(monitor):
        await fs._rm("data", recursive=True, callback=callback)
    assert inner.stats.calls == {"delete": 1}
    assert not await inner._exists("data")
    # Still an operation, watched as any: the root is its one path deleted.
    assert ("yielded", "rm", "data") in monitor.events
    assert callback.value == 1


async def test_the_profile_is_awaited_for_the_path_of_each_call(
    inner: SimulatedFileSystem,
) -> None:
    asked: list[str] = []

    class PerPath(_Tree):
        # E.g. a storage whose profile a request finds, per bucket.
        async def _storage_profile_of(self, path: str) -> StorageProfile:
            asked.append(path)
            await asyncio.sleep(0)
            return StorageProfile(deletes_directories=path.startswith("data"))

        async def _rm_tree(self, path: str) -> None:
            await self.inner._rm(path, recursive=True)

    fs = PerPath(inner, asynchronous=True)
    await fs._copy("data", "copied", recursive=True)
    # A copy asks of its destination, once.
    assert asked == ["copied"]

    inner.reset_stats()
    await fs._rm("data", recursive=True)
    assert asked[-1] == "data"
    assert inner.stats.calls == {"delete": 1}  # Deleted whole, as declared.

    inner.reset_stats()
    await fs._rm("copied", recursive=True)
    assert asked[-1] == "copied"
    assert inner.stats.calls["delete"] > 1  # Walked: not declared there.


async def test_mkdir_makes_a_directory(fs: _Tree, inner: SimulatedFileSystem) -> None:
    # fsspec's own _mkdir does nothing.
    await fs._mkdir("new/deep")
    assert await inner._isdir("new/deep")
    with pytest.raises(FileExistsError):
        await fs._mkdir("new/deep")
    with pytest.raises(FileNotFoundError):
        await fs._mkdir("none/deep", create_parents=False)
    await fs._mkdir("new/other", create_parents=False)
    assert await inner._isdir("new/other")


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


async def test_filesystems_share_a_semaphore_given() -> None:
    inner = SimulatedFileSystem(hierarchical=True, latency=0.002, asynchronous=True)
    for i in range(10):
        inner.add_file(f"a/{i}/file")
        inner.add_file(f"b/{i}/file")
    limit = asyncio.Semaphore(3)
    fs1 = _Tree(inner, operation_concurrency=limit, asynchronous=True)
    fs2 = _Tree(inner, operation_concurrency=limit, asynchronous=True)
    assert fs1.operation_limit is fs2.operation_limit is limit
    await asyncio.gather(fs1._find("a"), fs2._find("b"))
    # Both together keep to the one limit.
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
    assert callback.value == 5 + 7  # The files and the directories deleted.


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


def test_storage_profile_must_be_set(inner: SimulatedFileSystem) -> None:
    class Unset(_Tree):
        storage_profile = None  # type: ignore[assignment]

    with pytest.raises(ValueError, match="storage_profile"):
        Unset(inner)


def _copied(fs: SimulatedFileSystem) -> dict[str, bytes]:
    return {p: f.data for p, f in fs._file_data.items()}


def _twins() -> tuple[_Tree, SimulatedFileSystem, SimulatedFileSystem]:
    """A TreeFileSystem on a storage, and a second storage with the same data."""
    inner, reference = (
        SimulatedFileSystem(hierarchical=True, asynchronous=True) for _ in range(2)
    )
    _tree(inner)
    _tree(reference)
    return _Tree(inner, asynchronous=True), inner, reference


@pytest.mark.parametrize("maxdepth", [1, 2])
async def test_rm_with_maxdepth_deletes_the_files_fsspecs_does(maxdepth: int) -> None:
    fs, inner, reference = _twins()
    # The directories at the limit hold files, so deleting them fails, as fsspec's
    # own fails: it sends its directories to _rm_file, which refuses them here.
    with pytest.raises((OSError, ExceptionGroup)):
        await fs._rm("data", recursive=True, maxdepth=maxdepth)
    with pytest.raises(IsADirectoryError):
        await AsyncFileSystem._rm(reference, "data", recursive=True, maxdepth=maxdepth)
    assert _copied(inner) == _copied(reference)
    assert await inner._exists("data/2023/01/part-0.parquet")
    assert not await inner._exists("data/README.md")


async def test_copy_with_maxdepth_is_fsspecs() -> None:
    fs, inner, reference = _twins()
    inner.add_file("data/2023/top.txt")
    reference.add_file("data/2023/top.txt")
    await fs._copy("data", "copied", recursive=True, maxdepth=2)
    await AsyncFileSystem._copy(reference, "data", "copied", recursive=True, maxdepth=2)
    assert _copied(inner) == _copied(reference)
    assert not await inner._exists("copied/2023/01")


async def test_mv_with_maxdepth_moves_only_what_it_copies() -> None:
    fs, inner, _ = _twins()
    await fs._mv("data", "moved", recursive=True, maxdepth=1)
    assert sorted(p for p in inner._file_data if p.startswith("moved")) == [
        "moved/README.md"
    ]
    # fsspec's own deletes the whole source: the files deeper are kept here.
    assert await inner._exists("data/2023/01/part-0.parquet")


async def test_copy_to_a_list_is_fsspecs() -> None:
    fs, inner, reference = _twins()
    sources = ["data/README.md", "data/2023/01/part-0.parquet"]
    targets = ["one.md", "two.parquet"]
    await fs._copy(sources, targets)
    await AsyncFileSystem._copy(reference, sources, targets)
    assert _copied(inner) == _copied(reference)
    assert inner._file_data["one.md"].data == b"readme"


@pytest.mark.parametrize(
    ("path1", "path2"), [("data/README.md", "moved.md"), ("data/*.md", "moved/")]
)
async def test_a_non_recursive_mv_is_fsspecs(path1: str, path2: str) -> None:
    fs, inner, _ = _twins()
    await fs._mv(path1, path2)
    # fsspec's own mv, which only its sync API has, on the same data.
    reference = SimulatedFileSystem(hierarchical=True)
    _tree(reference)
    reference.mv(path1, path2)
    assert _copied(inner) == _copied(reference)


async def test_mv_onto_itself_sends_nothing(
    fs: _Tree, inner: SimulatedFileSystem
) -> None:
    before = _copied(inner)
    inner.reset_stats()
    await fs._mv("data", "data", recursive=True)
    assert inner.stats.calls == {}
    assert _copied(inner) == before


async def test_du_with_withdirs_is_fsspecs(
    fs: _Tree, inner: SimulatedFileSystem
) -> None:
    for total in (True, False):
        expected = await AsyncFileSystem._du(inner, "data", total=total, withdirs=True)
        assert await fs._du("data", total=total, withdirs=True) == expected


async def test_walk_of_an_empty_dir(fs: _Tree, inner: SimulatedFileSystem) -> None:
    inner.add_dir("empty")
    walked = [t async for t in fs._walk("empty")]
    assert walked == [t async for t in AsyncFileSystem._walk(inner, "empty")]


def test_the_sync_rmdir_deletes() -> None:
    # fsspec makes no sync rmdir of _rmdir: the class must, or rmdir does nothing.
    inner = SimulatedFileSystem(hierarchical=True)
    inner.add_dir("empty")
    fs = _Tree(inner)
    fs.rmdir("empty")
    assert not inner.exists("empty")
