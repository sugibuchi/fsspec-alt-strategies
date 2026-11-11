"""Tests of `AsyncRoutingFileSystem`, passing each call to the filesystem of its storage."""

from __future__ import annotations

import io
import pickle
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Self

import pytest
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.spec import AbstractFileSystem

from fsspec_alt_strategies.monitor import Monitor, monitoring
from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.router import AsyncRoutingFileSystem
from fsspec_alt_strategies_impl.wrapper import AsyncFlatWrapper, AsyncTreeWrapper
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


class _Reader:
    """A file of a simulated storage, read whole by one request."""

    def __init__(self, fs: SimulatedFileSystem, path: str) -> None:
        self.fs, self.path, self.done = fs, path, False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    async def read(self, size: int = -1) -> bytes:
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


class _File(io.BytesIO):
    """A sync file written on closing, or in a transaction on commit."""

    def __init__(self, fs: SimulatedFileSystem, path: str, autocommit: bool) -> None:
        super().__init__()
        self.fs, self.path, self.autocommit = fs, path, autocommit
        self.data = b""
        self.outcome: str | None = None

    def close(self) -> None:
        if not self.closed:
            self.data = self.getvalue()
            if self.autocommit:
                self.fs.add_file(self.path, self.data)
        super().close()

    def commit(self) -> None:
        self.fs.add_file(self.path, self.data)
        self.outcome = "committed"

    def discard(self) -> None:
        self.outcome = "discarded"


class _Files:
    """The file objects of the simulated storage a wrapper wraps."""

    fs: SimulatedFileSystem
    closed = 0

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return _Reader(self.fs, path) if "r" in mode else _Writer(self.fs, path)

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return _File(self.fs, path, kwargs.get("autocommit", True))

    def close(self) -> None:
        _Files.closed += 1


class _Plain(SimulatedFileSystem):
    """A plain async fsspec filesystem, with no strategies, but files to stream."""

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return _Reader(self, path) if "r" in mode else _Writer(self, path)


class _Tree(_Files, AsyncTreeWrapper):
    pass


class _Flat(_Files, AsyncFlatWrapper):
    def _ils(
        self, path: str, recursive: bool = False, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        if recursive:
            return self.fs._list_prefix(f"{path}/{prefix}" if path else prefix)
        else:
            return super()._ils(path, prefix=prefix, **kwargs)


_OPTIONS = {"operation_concurrency": 4, "io_concurrency": 2, "asynchronous": True}


class _Router(AsyncRoutingFileSystem):
    """Paths "lake/<backend>/<path>": a tree, a flat and a plain storage."""

    protocol = "routed"
    root_marker = ""
    cachable = False

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("asynchronous", True)
        super().__init__(transfer_concurrency=2, **kwargs)
        self.tree = SimulatedFileSystem(hierarchical=True, asynchronous=True)
        self.flat = SimulatedFileSystem(asynchronous=True)
        self.plain = _Plain(asynchronous=True)
        self.makers = {
            "tree": lambda: _Tree(
                self.tree, storage_profile=StorageProfile(), **_OPTIONS
            ),
            "flat": lambda: _Flat(
                self.flat,
                rm_batch_size=10,
                storage_profile=StorageProfile(implicit_directories=True),
                **_OPTIONS,
            ),
            # A plain async fsspec filesystem, without strategies.
            "plain": lambda: self.plain,
        }
        self.names: dict[int, str] = {}

    def _route(self, path: str) -> tuple[AsyncFileSystem, str] | None:
        parts = path.split("/", 2)
        if parts[0] != "lake" or len(parts) == 1:
            if parts[0] in ("", "lake"):
                return None
            raise FileNotFoundError(path)
        if parts[1] == "":
            return None
        if parts[1] not in self.makers:
            raise FileNotFoundError(path)
        fs = self._backend(parts[1], self.makers[parts[1]])
        self.names[id(fs)] = parts[1]
        return fs, parts[2] if len(parts) == 3 else ""

    def _outer(self, fs: AsyncFileSystem, path: str) -> str:
        return f"lake/{self.names[id(fs)]}/{path}".rstrip("/")

    async def _roots(self, path: str) -> list[dict[str, Any]]:
        if path == "":
            names = ["lake"]
        elif path == "lake":
            names = [f"lake/{n}" for n in self.makers]
        else:
            raise FileNotFoundError(path)
        return [{"name": n, "size": 0, "type": "directory"} for n in names]

    def backend(self, name: str) -> Any:
        return self._route(f"lake/{name}/x")[0]  # type: ignore[index]


_FILES = ["data/a/1.txt", "data/a/2.txt", "data/b/3.txt"]


def _router(**kwargs: Any) -> _Router:
    router = _Router(**kwargs)
    for sim in (router.tree, router.flat, router.plain):
        for path in _FILES:
            sim.add_file(path, path.encode())
    return router


@pytest.fixture
def router() -> _Router:
    return _router()


def _reset(router: _Router) -> None:
    for sim in (router.tree, router.flat, router.plain):
        sim.reset_stats()


# Routing, and the names mapped back.


async def test_info_and_ls_are_the_routers_paths(router: _Router) -> None:
    info = await router._info("routed://lake/tree/data/a/1.txt")
    assert info["name"] == "lake/tree/data/a/1.txt"
    assert info["size"] == len(b"data/a/1.txt")
    assert await router._ls("lake/flat/data", detail=False) == [
        "lake/flat/data/a",
        "lake/flat/data/b",
    ]


async def test_above_the_storages_lists_the_roots(router: _Router) -> None:
    assert await router._ls("", detail=False) == ["lake"]
    assert await router._ls("lake", detail=False) == [
        "lake/tree",
        "lake/flat",
        "lake/plain",
    ]
    assert (await router._info("lake"))["type"] == "directory"
    assert (await router._info(""))["type"] == "directory"
    assert await router._isdir("lake")
    assert not await router._isfile("lake")
    with pytest.raises(FileNotFoundError):
        await router._info("sea")
    assert not await router._exists("sea/x")
    assert not await router._exists("lake/nope/x")
    assert await router._exists("lake/tree")


class _Stripping(_Router):
    """A router whose path conversion drops a trailing "/"."""

    def _route(self, path: str) -> tuple[AsyncFileSystem, str] | None:
        target = super()._route(path)
        return target and (target[0], target[1].rstrip("/"))


def test_a_trailing_slash_is_kept_on_the_inner_path() -> None:
    router = _Stripping()
    assert router._target("lake/tree/data/")[1] == "data/"  # type: ignore[index]
    assert router._target("lake/tree/data")[1] == "data"  # type: ignore[index]
    assert router._target("lake/tree/")[1] == ""  # type: ignore[index]


async def test_a_call_passed_on_keeps_the_slash() -> None:
    router = _Stripping()
    tree = router.backend("tree")
    given: list[str] = []

    async def copy(path1: str, path2: str, *args: Any, **kwargs: Any) -> None:
        given.extend([path1, path2])

    tree._copy = copy
    await router._copy("lake/tree/a/", "lake/tree/dest/")
    assert given == ["a/", "dest/"]


@pytest.mark.parametrize("method", ["_copy", "_mv"])
@pytest.mark.parametrize("source", ["tree", "flat", "plain"])
@pytest.mark.parametrize("destination", ["tree", "flat", "plain"])
@pytest.mark.parametrize("form", ["path", "glob", "list"])
async def test_a_file_goes_into_a_missing_dir_ending_in_a_slash(
    method: str, source: str, destination: str, form: str
) -> None:
    # Within a storage and across two, as fsspec's copy: never a file "new".
    router = _Stripping()
    for sim in (router.tree, router.flat, router.plain):
        sim.add_file("data/a/1.txt", b"1")
    path = {
        "path": f"lake/{source}/data/a/1.txt",
        "glob": f"lake/{source}/data/a/1.*",
        "list": [f"lake/{source}/data/a/1.txt"],
    }[form]
    await getattr(router, method)(path, f"lake/{destination}/new/")
    sim = getattr(router, destination)
    assert sim._file_data["new/1.txt"].data == b"1"
    assert "new" not in sim._file_data


class _RootDescribing(_Router):
    async def _root_info(self, path: str) -> dict[str, Any]:
        return {"name": path, "size": 0, "type": "directory", "owner": "me"}


async def test_info_above_the_storages_is_the_root_info() -> None:
    router = _RootDescribing()
    assert (await router._info("routed://"))["owner"] == "me"
    assert (await router._info("lake"))["owner"] == "me"
    assert await router._isdir("lake")
    router.tree.add_file("a.txt", b"a")
    assert "owner" not in await router._info("lake/tree/a.txt")


async def test_an_errors_note_names_the_routers_path(router: _Router) -> None:
    with pytest.raises(FileNotFoundError) as info:
        await router._cat_file("lake/tree/missing")
    assert "'lake/tree/missing' was routed to" in "".join(info.value.__notes__)


async def test_changes_above_the_storages_are_refused(router: _Router) -> None:
    with pytest.raises(PermissionError):
        await router._rm("lake", recursive=True)
    with pytest.raises(PermissionError):
        await router._pipe_file("lake", b"")
    await router._makedirs("lake", exist_ok=True)
    with pytest.raises(FileExistsError):
        await router._makedirs("lake")


async def test_a_sync_filesystem_is_refused(router: _Router) -> None:
    router.makers["memory"] = MemoryFileSystem
    with pytest.raises(TypeError, match="SyncTreeWrapper"):
        await router._info("lake/memory/x")


# The primitives, each the storage's own.


async def test_the_primitives_are_forwarded(router: _Router, tmp_path: Path) -> None:
    assert await router._cat_file("lake/tree/data/a/1.txt", start=5) == b"a/1.txt"
    await router._pipe_file("lake/flat/new.txt", b"new")
    assert router.flat._file_data["new.txt"].data == b"new"
    await router._cp_file("lake/tree/data/a/1.txt", "lake/tree/c.txt")
    assert router.tree.stats.calls["copy"] == 1
    await router._mv_file("lake/tree/c.txt", "lake/tree/d.txt")
    assert "c.txt" not in router.tree._file_data
    await router._rm_file("lake/flat/new.txt")
    assert "new.txt" not in router.flat._file_data
    await router._makedirs("lake/tree/m")
    await router._makedirs("lake/tree/n/o", exist_ok=True)
    await router._rmdir("lake/tree/m")
    await router._mkdir("lake/plain/p")
    assert await router._isdir("lake/tree/n/o")
    assert not await router._isdir("lake/tree/m")
    assert await router._isfile("lake/tree/d.txt")
    assert await router._size("lake/tree/d.txt") == len(b"data/a/1.txt")

    # The simulated storage moves no local files: the calls are recorded.
    moved: list[tuple[str, ...]] = []
    tree = router.backend("tree")

    async def get_file(rpath: str, lpath: str, **kwargs: Any) -> None:
        moved.append(("get", rpath, lpath))

    async def put_file(lpath: str, rpath: str, **kwargs: Any) -> None:
        moved.append(("put", lpath, rpath))

    tree._get_file, tree._put_file = get_file, put_file
    local = str(tmp_path / "1.txt")
    await router._get_file("lake/tree/data/a/1.txt", local)
    await router._put_file(local, "lake/tree/put.txt")
    assert moved == [("get", "data/a/1.txt", local), ("put", local, "put.txt")]

    f = await router.open_async("lake/flat/data/b/3.txt")
    assert await f.read() == b"data/b/3.txt"
    with router.open("lake/tree/w.txt", "wb") as w:
        w.write(b"w")
    assert router.tree._file_data["w.txt"].data == b"w"


def test_the_sync_methods_fsspec_does_not_make(router: _Router) -> None:
    sync_router = _Router(asynchronous=False)
    sync_router.tree.add_file("x/1.txt", b"1")
    sync_router.tree.add_dir("empty")
    sync_router.rmdir("lake/tree/empty")
    sync_router.mv("lake/tree/x", "lake/tree/y", recursive=True)
    assert set(sync_router.tree._file_data) == {"y/1.txt"}


def test_created_modified_and_sign_are_the_storages(router: _Router) -> None:
    tree = router.backend("tree")
    tree.created = lambda path: ("created", path)
    tree.modified = lambda path: ("modified", path)
    tree.sign = lambda path, expiration=100: f"signed {path} {expiration}"
    assert router.created("lake/tree/a") == ("created", "a")
    assert router.modified("lake/tree/a") == ("modified", "a")
    assert router.sign("lake/tree/a", expiration=5) == "signed a 5"


# The recursive methods, each the storage's own strategy.


async def test_find_runs_the_storages_strategy(router: _Router) -> None:
    expected = [f"lake/tree/{p}" for p in _FILES]
    assert await router._find("lake/tree/data") == expected
    # tree_find: a listing per directory.
    assert router.tree.stats.calls == {"list": 3}
    assert await router._find("lake/flat/data") == [f"lake/flat/{p}" for p in _FILES]
    # flat_find: one listing of the prefix.
    assert router.flat.stats.calls == {"list": 1}
    detail = await router._find("lake/flat/data", detail=True)
    assert all(name == e["name"] for name, e in detail.items())


async def test_find_above_the_storages_walks_each(router: _Router) -> None:
    found = await router._find("lake")
    assert found == sorted(
        f"lake/{b}/{p}" for b in ("tree", "flat", "plain") for p in _FILES
    )
    assert found == await AsyncFileSystem._find(router, "lake")
    assert await router._du("lake") == 3 * sum(len(p) for p in _FILES)


async def test_du_and_walk_are_the_storages(router: _Router) -> None:
    assert await router._du("lake/tree/data", total=False) == {
        f"lake/tree/{p}": len(p) for p in _FILES
    }
    walked = [item async for item in router._walk("lake/flat/data")]
    assert walked[0] == ("lake/flat/data", ["a", "b"], [])
    assert router.flat.stats.calls == {"list": 1}  # flat_walk
    above = [root async for root, _, _ in router._walk("lake", maxdepth=2)]
    assert above[:2] == ["lake", "lake/tree"]


async def test_a_glob_in_one_storage_goes_whole_to_it(router: _Router) -> None:
    assert await router._glob("lake/flat/data/*/1.txt") == ["lake/flat/data/a/1.txt"]
    assert router.flat.stats.calls == {"list": 1}  # flat_glob
    assert router.tree.stats.calls == {}


async def test_a_glob_across_storages_lists_the_roots(router: _Router) -> None:
    matches = await router._glob("lake/*/data/b/*")
    assert matches == [f"lake/{b}/data/b/3.txt" for b in ("flat", "plain", "tree")]
    assert matches == await AsyncFileSystem._glob(router, "lake/*/data/b/*")


async def test_expand_path(router: _Router) -> None:
    assert await router._expand_path("lake/tree/data/a", recursive=True) == [
        "lake/tree/data/a",
        "lake/tree/data/a/1.txt",
        "lake/tree/data/a/2.txt",
    ]


async def test_rm_groups_by_storage(router: _Router) -> None:
    await router._rm(["lake/tree/data", "lake/flat/data"], recursive=True)
    assert router.tree._file_data == {}
    assert router.flat._file_data == {}
    # flat_rm deletes in one batch, tree_rm a file per request.
    assert router.flat.stats.calls["list"] == 1  # flat_rm
    assert router.tree.stats.calls["list"] == 3  # tree_rm
    assert router.plain._file_data


async def test_rm_of_a_pattern_across_storages(router: _Router) -> None:
    await router._rm("lake/*/data/b/3.txt")
    for sim in (router.tree, router.flat, router.plain):
        assert "data/b/3.txt" not in sim._file_data


# Copies and moves.


async def test_copy_within_a_storage_is_its_own(router: _Router) -> None:
    await router._copy("lake/tree/data", "lake/tree/copied", recursive=True)
    assert router.tree.stats.calls["copy"] == 3
    assert {p for p in router.tree._file_data if p.startswith("copied/")} == {
        p.replace("data", "copied", 1) for p in _FILES
    }
    await router._cp_file("lake/tree/data/a/1.txt", "lake/tree/one.txt")
    await router._mv_file("lake/tree/one.txt", "lake/tree/two.txt")
    assert router.tree.stats.calls["copy"] == 5  # The move copies too.
    assert router.tree._file_data["two.txt"].data == b"data/a/1.txt"
    assert "one.txt" not in router.tree._file_data


@pytest.mark.parametrize(
    ("source", "destination"), [("tree", "flat"), ("flat", "tree")]
)
async def test_copy_across_storages_streams(
    router: _Router, source: str, destination: str
) -> None:
    await router._copy(
        f"lake/{source}/data", f"lake/{destination}/copied", recursive=True
    )
    src, dst = getattr(router, source), getattr(router, destination)
    for path in _FILES:
        assert dst._file_data[path.replace("data", "copied", 1)].data == path.encode()
    assert src.stats.calls["read"] == 3
    assert dst.stats.calls["write"] == 3


@pytest.mark.parametrize("move", [False, True])
async def test_a_list_across_storages_goes_into_the_destination(
    router: _Router, move: bool
) -> None:
    router.tree.add_file("other/b/4.txt", b"4")
    sources = ["lake/tree/data/a", "lake/flat/data/b", "lake/tree/other/b"]
    with pytest.raises(FileExistsError) as info:
        if move:
            await router._mv(sources, "lake/plain/dest", recursive=True)
        else:
            await router._copy(sources, "lake/plain/dest", recursive=True)
    # Each root under its name, a single one in its storage too. Of the two
    # named "b", the first in path order goes, as tree_cp places a list.
    assert info.value.args == ("lake/plain/dest/b",)
    assert "'lake/tree/other/b'" in "".join(info.value.__notes__)
    assert sorted(p for p in router.plain._file_data if p.startswith("dest/")) == [
        "dest/a/1.txt",
        "dest/a/2.txt",
        "dest/b/3.txt",
    ]
    assert "other/b/4.txt" in router.tree._file_data


class _ServerCopying(_Router):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.server_copies: list[tuple[str, str]] = []
        # The size of each entry handed over, which a server copy may need.
        self.sizes: dict[str, int] = {}

    async def _cross_fs_cp_entry(
        self,
        src_fs: AsyncFileSystem,
        src_entry: dict[str, Any],
        dest_fs: AsyncFileSystem,
        dest_path: str,
    ) -> None:
        src_path = src_entry["name"]
        if src_path.endswith("3.txt"):
            # e.g. not possible for this one: streamed.
            await super()._cross_fs_cp_entry(src_fs, src_entry, dest_fs, dest_path)
            return
        self.server_copies.append((src_path, dest_path))
        self.sizes[src_path] = src_entry["size"]
        data = src_fs.fs._file_data[src_path].data  # type: ignore[attr-defined]
        dest_fs.fs.add_file(dest_path, data)  # type: ignore[attr-defined]


async def test_copy_across_storages_tries_the_server_first() -> None:
    router = _ServerCopying()
    for path in _FILES:
        router.tree.add_file(path, path.encode())
    await router._copy("lake/tree/data", "lake/flat/copied", recursive=True)
    assert sorted(router.server_copies) == [
        ("data/a/1.txt", "copied/a/1.txt"),
        ("data/a/2.txt", "copied/a/2.txt"),
    ]
    # Each entry as the source's listing gave it, with its size.
    assert router.sizes == {"data/a/1.txt": 12, "data/a/2.txt": 12}
    assert router.tree.stats.calls["read"] == 1
    assert router.flat.stats.calls["write"] == 1
    assert len(router.flat._file_data) == 3


async def test_maxdepth_across_storages_runs_the_strategy() -> None:
    router = _ServerCopying()
    for path in [*_FILES, "data/top.txt"]:
        router.tree.add_file(path, path.encode())
    await router._copy("lake/tree/data", "lake/flat/copied", recursive=True, maxdepth=1)
    # Each file by _cross_fs_cp_entry, which fsspec's own copy never calls.
    assert router.server_copies == [("data/top.txt", "copied/top.txt")]
    assert sorted(router.flat._file_data) == ["copied/top.txt"]

    await router._mv("lake/tree/data", "lake/flat/moved", recursive=True, maxdepth=1)
    assert "moved/top.txt" in router.flat._file_data
    # What is deeper stays, as _tree_mv keeps it.
    assert sorted(router.tree._file_data) == sorted(_FILES)


async def test_copy_from_a_plain_storage_is_fsspecs(router: _Router) -> None:
    await router._copy("lake/plain/data", "lake/tree/copied", recursive=True)
    assert router.tree._file_data["copied/a/1.txt"].data == b"data/a/1.txt"
    assert router.plain.stats.calls["read"] == 3


async def test_copy_of_a_file_across_storages(router: _Router) -> None:
    await router._cp_file("lake/tree/data/a/1.txt", "lake/flat/one.txt")
    assert router.flat._file_data["one.txt"].data == b"data/a/1.txt"


@pytest.mark.parametrize("source", ["tree", "flat"])
async def test_mv_across_storages_deletes_each_source(
    router: _Router, source: str
) -> None:
    destination = "flat" if source == "tree" else "tree"
    deleted_while_copying: list[int] = []
    src, dst = getattr(router, source), getattr(router, destination)
    write = dst._pipe_file

    async def pipe_file(path: str, value: bytes, **kwargs: Any) -> None:
        deleted_while_copying.append(src.stats.calls["delete"])
        await write(path, value, **kwargs)

    dst._pipe_file = pipe_file
    await router._mv(f"lake/{source}/data", f"lake/{destination}/moved", recursive=True)
    assert src._file_data == {}
    assert sorted(p for p in dst._file_data if p.startswith("moved/")) == sorted(
        p.replace("data", "moved", 1) for p in _FILES
    )
    if source == "tree":
        # The sources were deleted as their copies ended, not after them all.
        # A flat source's go in batches, as flat_mv deletes them.
        assert deleted_while_copying[-1] > 0
        assert not await router._exists("lake/tree/data")


async def test_mv_within_a_storage_is_its_own(router: _Router) -> None:
    await router._mv("lake/flat/data", "lake/flat/moved", recursive=True)
    assert sorted(router.flat._file_data) == sorted(
        p.replace("data", "moved", 1) for p in _FILES
    )
    await router._mv("lake/plain/data/a/1.txt", "lake/plain/one.txt")
    assert "one.txt" in router.plain._file_data


# Monitors, transactions and the lifecycle.


class _Calls(Monitor):
    def __init__(self) -> None:
        self.calls: list[Any] = []

    async def started(self, call: Any, state: Any, op: Any) -> None:
        self.calls.append(call)


async def test_monitors_see_the_router_in_via(router: _Router) -> None:
    monitor = _Calls()
    async with monitoring(monitor):
        await router._find("lake/tree/data")
        async for _ in router._walk("lake/flat/data"):
            pass
        await router._copy("lake/tree/data", "lake/flat/copied", recursive=True)
    find, walk, copy = monitor.calls
    assert (find.method, find.path, find.via) == ("find", "data", (router,))
    assert find.fs is router.tree
    assert (walk.method, walk.via) == ("walk", (router,))
    # A copy across storages is the router's own call.
    assert (copy.fs, copy.method, copy.path, copy.via) == (
        router,
        "copy",
        "lake/tree/data",
        (),
    )


@pytest.mark.parametrize("commit", [True, False])
def test_one_transaction_across_storages(router: _Router, commit: bool) -> None:
    files = []
    with pytest.raises(RuntimeError) if not commit else nullcontext():
        with router.transaction:
            for path in ("lake/tree/t.txt", "lake/flat/t.txt"):
                with router.open(path, "wb") as f:
                    f.write(b"t")
                files.append(f)
            assert "t.txt" not in router.tree._file_data
            if not commit:
                raise RuntimeError
    assert [f.outcome for f in files] == ["committed" if commit else "discarded"] * 2
    assert ("t.txt" in router.tree._file_data) is commit
    assert ("t.txt" in router.flat._file_data) is commit


async def test_pipe_and_put_join_the_transaction(
    router: _Router, tmp_path: Path
) -> None:
    local = tmp_path / "put.txt"
    local.write_bytes(b"put")
    router.start_transaction()
    await router._pipe_file("lake/tree/piped.txt", b"piped")
    await router._put_file(str(local), "lake/flat/put.txt")
    # Written through open, on commit: no request yet.
    assert "piped.txt" not in router.tree._file_data
    assert "put.txt" not in router.flat._file_data
    assert len(router.transaction.files) == 2
    router.end_transaction()
    assert router.tree._file_data["piped.txt"].data == b"piped"
    assert router.flat._file_data["put.txt"].data == b"put"


async def test_open_async_takes_no_part_in_the_transaction(router: _Router) -> None:
    # As in fsspec, whose open_async knows no transaction: written on closing.
    router.start_transaction()
    async with await router.open_async("lake/tree/t.txt", "wb") as f:
        await f.write(b"t")
    assert router.tree._file_data["t.txt"].data == b"t"
    assert not router.transaction.files
    router.end_transaction()
    assert router.tree._file_data["t.txt"].data == b"t"


def test_close_closes_the_storages_made(router: _Router) -> None:
    router.backend("tree")
    router.backend("flat")
    before = _Files.closed
    router.close()
    assert _Files.closed == before + 2
    assert router._backends == {}


def test_pickling_keeps_the_arguments(router: _Router) -> None:
    router.backend("tree")
    copy = pickle.loads(pickle.dumps(router))
    assert type(copy) is _Router
    assert copy._backends == {}
    assert copy.transfer_concurrency == router.transfer_concurrency


def test_transfer_concurrency_is_checked() -> None:
    with pytest.raises(ValueError, match="transfer_concurrency"):
        AsyncRoutingFileSystem.__init__(  # type: ignore[misc]
            object.__new__(_Router), transfer_concurrency=0
        )


# Every public method of fsspec is either passed on, or inherited on purpose.

# Sync methods fsspec makes from the router's async ones, or aliases of others.
_MIRRORED = {
    "cat_file", "copy", "cp", "cp_file", "delete", "disk_usage", "download", "du",
    "exists", "expand_path", "find", "get_file", "glob", "info", "isdir", "isfile",
    "lexists", "listdir", "ls", "makedir", "makedirs", "mkdir", "mkdirs", "move",
    "pipe_file", "put_file", "rename", "rm", "rm_file", "size", "stat", "upload",
    "walk",
}  # fmt: skip
# fsspec's own, built on the primitives passed on, as on any filesystem.
_BUILT_ON_PRIMITIVES = {
    "_cat", "_cat_ranges", "_get", "_pipe", "_put", "_size", "_sizes", "cat",
    "cat_ranges", "checksum", "get", "head", "open", "pipe", "put", "read_block",
    "read_bytes", "read_text", "sizes", "tail", "touch", "tree", "ukey",
    "write_bytes", "write_text",
}  # fmt: skip
# The router's own as a filesystem: its paths, its instance, its transaction.
_ROUTERS_OWN = {
    "_get_kwargs_from_urls", "_get_pyarrow_filesystem", "_isfilestore",
    "_ls_from_cache", "_parent", "_process_limits", "_strip_protocol",
    "clear_instance_cache", "current", "end_transaction", "from_dict", "from_json",
    "get_mapper", "start_transaction", "to_dict", "to_json", "transaction_type",
    "unstrip_protocol",
}  # fmt: skip
_INHERITED = _MIRRORED | _BUILT_ON_PRIMITIVES | _ROUTERS_OWN


def _public(cls: type) -> set[str]:
    return {n for n in dir(cls) if not n.startswith("__") and callable(getattr(cls, n))}


def test_every_fsspec_method_is_passed_on_or_inherited_on_purpose() -> None:
    methods = _public(AbstractFileSystem) | _public(AsyncFileSystem)
    passed_on = set(vars(AsyncRoutingFileSystem))
    assert methods - passed_on - _INHERITED == set()
    assert not passed_on & _INHERITED
