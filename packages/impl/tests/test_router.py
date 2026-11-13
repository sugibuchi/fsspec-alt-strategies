"""Tests of `RoutingFileSystem`, passing each call to the filesystem of its storage."""

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
from fsspec_alt_strategies.profile import UNKNOWN, StorageProfile
from fsspec_alt_strategies_impl.router import RoutingFileSystem, RoutingHandler
from fsspec_alt_strategies_impl.wrapper.flat import FsspecFlatHandler
from fsspec_alt_strategies_impl.wrapper.tree import FsspecTreeHandler
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
    """The file objects of the simulated storage a handler wraps, and its releases counted."""

    fs: SimulatedFileSystem
    closed = 0

    async def open_async_file(self, path: str, mode: str, **kwargs: Any) -> Any:
        return _Reader(self.fs, path) if "r" in mode else _Writer(self.fs, path)

    def open_sync_file(self, path: str, mode: str, **kwargs: Any) -> Any:
        return _File(self.fs, path, kwargs.get("autocommit", True))

    def release(self) -> None:
        _Files.closed += 1


class _Plain(SimulatedFileSystem):
    """A plain async fsspec filesystem, with no strategies, but files to stream."""

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return _Reader(self, path) if "r" in mode else _Writer(self, path)


class _TreeHandler(_Files, FsspecTreeHandler):
    pass


class _FlatHandler(_Files, FsspecFlatHandler):
    def ils_recursive(  # type: ignore[override]
        self, entry: dict[str, Any], *, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        path = entry["name"]
        return self.fs._list_prefix(f"{path}/{prefix}" if path else prefix)


_OPTIONS = {"operation_concurrency": 4, "io_concurrency": 2}


class _Lake(RoutingHandler):
    """Paths "lake/<storage>/<path>": a tree, a flat and a plain storage.

    It refuses a path holding ":", and notes each path it checks.
    """

    protocol = "routed"

    def __init__(self) -> None:
        super().__init__(transfer_concurrency=2)
        self.tree = SimulatedFileSystem(hierarchical=True, asynchronous=True)
        self.flat = SimulatedFileSystem(asynchronous=True)
        self.plain = _Plain(asynchronous=True)
        self.makers: dict[str, Any] = {
            "tree": lambda: _TreeHandler(
                self.tree, storage_profile=StorageProfile(), **_OPTIONS
            ),
            "flat": lambda: _FlatHandler(
                self.flat,
                rm_batch_size=10,
                storage_profile=StorageProfile(implicit_directories=True),
                **_OPTIONS,
            ),
            # A plain async fsspec filesystem, wrapped with an unknown profile.
            "plain": lambda: FsspecTreeHandler(
                self.plain, storage_profile=UNKNOWN, **_OPTIONS
            ),
        }
        self.names: dict[int, str] = {}
        self.checked: list[str] = []

    def verify_path(self, path: str) -> None:
        self.checked.append(path)
        if ":" in path:
            raise ValueError(f"A name holds ':': {path!r}")

    def route(self, path: str) -> tuple[Any, str] | None:
        parts = path.split("/", 2)
        if parts[0] != "lake" or len(parts) == 1:
            if parts[0] in ("", "lake"):
                return None
            raise FileNotFoundError(path)
        if parts[1] not in self.makers:
            raise FileNotFoundError(path)
        handler = self.inner(parts[1], self.makers[parts[1]])
        self.names[id(handler)] = parts[1]
        return handler, parts[2] if len(parts) == 3 else ""

    def outer(self, handler: Any, path: str) -> str:
        return f"lake/{self.names[id(handler)]}/{path}".rstrip("/")

    async def roots(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        path = entry["name"]
        if path == "":
            names = ["lake"]
        elif path == "lake":
            names = [f"lake/{n}" for n in self.makers]
        else:
            raise FileNotFoundError(path)
        return [{"name": n, "size": 0, "type": "directory"} for n in names]


class _Router(RoutingFileSystem):
    """A router of a lake, made from its lake's class, as from storage options."""

    protocol = _Lake.protocol
    root_marker = _Lake.root
    _strip_protocol = _Lake.strip_protocol

    def __init__(self, lake: type[_Lake] = _Lake, **kwargs: Any) -> None:
        super().__init__(lake(), **kwargs)


_FILES = ["data/a/1.txt", "data/a/2.txt", "data/b/3.txt"]


def _router(
    lake: type[_Lake] = _Lake, *, asynchronous: bool = True, filled: bool = True
) -> Any:
    """A router on a lake of ``lake``'s class, its three storages holding `_FILES` if ``filled``.

    Its handler's storages are reached as ``router.tree``, ``router.flat`` and
    ``router.plain`` too, for the tests.
    """
    router = _Router(lake, asynchronous=asynchronous)
    for name in ("tree", "flat", "plain"):
        sim = getattr(router.handler, name)
        setattr(router, name, sim)
        for path in _FILES if filled else ():
            sim.add_file(path, path.encode())
    return router


def _backend(router: Any, name: str) -> Any:
    """The filesystem the router made for a storage, routing to it."""
    return router._route(f"lake/{name}/x")[0]


@pytest.fixture
def router() -> Any:
    return _router()


def _reset(router: Any) -> None:
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


class _Stripping(_Lake):
    """A lake whose path conversion drops a trailing "/"."""

    def route(self, path: str) -> tuple[Any, str] | None:
        target = super().route(path)
        return target and (target[0], target[1].rstrip("/"))


def test_a_trailing_slash_is_kept_on_the_inner_path() -> None:
    router = _router(_Stripping)
    assert router._target("lake/tree/data/")[1] == "data/"  # type: ignore[index]
    assert router._target("lake/tree/data")[1] == "data"  # type: ignore[index]
    assert router._target("lake/tree/")[1] == ""  # type: ignore[index]


async def test_a_call_passed_on_keeps_the_slash() -> None:
    router = _router(_Stripping)
    tree = _backend(router, "tree")
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
    router = _router(_Stripping)
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


class _RootDescribing(_Lake):
    async def root_info(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {**entry, "size": 0, "type": "directory", "owner": "me"}


async def test_info_above_the_storages_is_the_root_info() -> None:
    router = _router(_RootDescribing)
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


async def test_a_filesystem_routed_to_is_refused(router: _Router) -> None:
    router.handler.makers["memory"] = MemoryFileSystem
    with pytest.raises(TypeError, match="FsspecTreeHandler"):
        await router._info("lake/memory/x")


async def test_each_path_is_checked_by_the_router_then_the_storage() -> None:
    class Strict(_TreeHandler):
        def verify_path(self, path: str) -> None:
            if path.endswith(".tmp"):
                raise ValueError(f"No temporary file here: {path!r}")

    router = _router()
    router.handler.makers["tree"] = lambda: Strict(
        router.tree, storage_profile=StorageProfile(), **_OPTIONS
    )
    assert await router._cat_file("lake/tree/data/a/1.txt") == b"data/a/1.txt"
    assert "lake/tree/data/a/1.txt" in router.handler.checked
    with pytest.raises(ValueError, match="holds ':'"):
        await router._info("lake/tree/a:b")
    with pytest.raises(ValueError, match="temporary"):
        await router._info("lake/tree/x.tmp")


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
    tree = _backend(router, "tree")

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
    sync_router = _router(asynchronous=False, filled=False)
    sync_router.tree.add_file("x/1.txt", b"1")
    sync_router.tree.add_dir("empty")
    sync_router.rmdir("lake/tree/empty")
    sync_router.mv("lake/tree/x", "lake/tree/y", recursive=True)
    assert set(sync_router.tree._file_data) == {"y/1.txt"}


def test_created_modified_and_sign_are_the_storages(router: _Router) -> None:
    tree = _backend(router, "tree")
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


class _ServerCopying(_Lake):
    def __init__(self) -> None:
        super().__init__()
        self.server_copies: list[tuple[str, str]] = []
        # The size of each entry handed over, which a server copy may need.
        self.sizes: dict[str, int] = {}

    async def cp_across(self, src: Any, entry: Any, dst: Any, path: Any) -> None:
        src_path = entry["name"]
        if src_path.endswith("3.txt"):
            # e.g. not possible for this one: streamed.
            await super().cp_across(src, entry, dst, path)
            return
        self.server_copies.append((src_path, path))
        self.sizes[src_path] = entry["size"]
        dst.fs.add_file(path, src.fs._file_data[src_path].data)


async def test_copy_across_storages_tries_the_server_first() -> None:
    router = _router(_ServerCopying, filled=False)
    for path in _FILES:
        router.tree.add_file(path, path.encode())
    await router._copy("lake/tree/data", "lake/flat/copied", recursive=True)
    assert sorted(router.handler.server_copies) == [
        ("data/a/1.txt", "copied/a/1.txt"),
        ("data/a/2.txt", "copied/a/2.txt"),
    ]
    # Each entry as the source's listing gave it, with its size.
    assert router.handler.sizes == {"data/a/1.txt": 12, "data/a/2.txt": 12}
    assert router.tree.stats.calls["read"] == 1
    assert router.flat.stats.calls["write"] == 1
    assert len(router.flat._file_data) == 3


async def test_maxdepth_across_storages_runs_the_strategy() -> None:
    router = _router(_ServerCopying, filled=False)
    for path in [*_FILES, "data/top.txt"]:
        router.tree.add_file(path, path.encode())
    await router._copy("lake/tree/data", "lake/flat/copied", recursive=True, maxdepth=1)
    # Each file by cp_across, which fsspec's own copy never calls.
    assert router.handler.server_copies == [("data/top.txt", "copied/top.txt")]
    assert sorted(router.flat._file_data) == ["copied/top.txt"]

    await router._mv("lake/tree/data", "lake/flat/moved", recursive=True, maxdepth=1)
    assert "moved/top.txt" in router.flat._file_data
    # What is deeper stays, as _tree_mv keeps it.
    assert sorted(router.tree._file_data) == sorted(_FILES)


async def test_copy_from_a_wrapped_storage_runs_the_tree_copy(router: _Router) -> None:
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
    _backend(router, "tree")
    _backend(router, "flat")
    before = _Files.closed
    router.close()
    assert _Files.closed == before + 2
    assert router._storages == {}
    # Made again by the next call.
    assert _backend(router, "tree") is not None


async def test_the_handlers_async_close_closes_the_storages_made(
    router: _Router,
) -> None:
    _backend(router, "tree")
    before = _Files.closed
    await router.handler.close()
    assert _Files.closed == before + 1


def test_pickling_keeps_the_arguments() -> None:
    # Made again from its arguments, as fsspec pickles any filesystem.
    router = _Router(asynchronous=True)
    _backend(router, "tree")
    copy = pickle.loads(pickle.dumps(router))
    assert type(copy) is _Router
    assert type(copy.handler) is _Lake
    assert copy._storages == {}
    assert copy.transfer_concurrency == router.transfer_concurrency


def test_transfer_concurrency_is_checked() -> None:
    with pytest.raises(ValueError, match="transfer_concurrency"):
        RoutingHandler.__init__(object.__new__(_Lake), transfer_concurrency=0)


def test_a_router_needs_a_handler() -> None:
    with pytest.raises(TypeError, match="handler"):
        RoutingFileSystem()  # type: ignore[call-arg]


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
    "_cat", "_get", "_pipe", "_put", "_size", "_sizes", "cat",
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
}  # fmt: skip
_INHERITED = _MIRRORED | _BUILT_ON_PRIMITIVES | _ROUTERS_OWN


def _public(cls: type) -> set[str]:
    return {n for n in dir(cls) if not n.startswith("__") and callable(getattr(cls, n))}


def test_every_fsspec_method_is_passed_on_or_inherited_on_purpose() -> None:
    methods = _public(AbstractFileSystem) | _public(AsyncFileSystem)
    passed_on = set(vars(RoutingFileSystem))
    assert methods - passed_on - _INHERITED == set()
    assert not passed_on & _INHERITED


async def test_cat_ranges_gives_each_storage_its_ranges(router: _Router) -> None:
    data = bytes(range(256)) * 400
    for sim in (router.tree, router.flat, router.plain):
        sim.add_file("big", data)
    starts = [i * 1000 for i in range(10)]
    paths, all_starts, all_ends = [], [], []
    for backend in ("tree", "flat", "plain"):
        paths += [f"lake/{backend}/big"] * 10
        all_starts += starts
        all_ends += [s + 100 for s in starts]
    paths.append("lake/missing/big")
    all_starts.append(0)
    all_ends.append(10)
    _reset(router)

    out = await router._cat_ranges(paths, all_starts, all_ends)

    assert out[:-1] == [
        data[s:e] for s, e in zip(all_starts[:-1], all_ends[:-1], strict=True)
    ]
    assert isinstance(out[-1], FileNotFoundError)
    # Each storage's filesystem merges its ten ranges, the wrapped one's too.
    for sim in (router.tree, router.flat, router.plain):
        assert sim.stats.calls == {"read": 1}
    with pytest.raises(FileNotFoundError):
        await router._cat_ranges(paths, all_starts, all_ends, on_error="raise")


@pytest.mark.parametrize("move", [False, True])
async def test_transfers_across_storages_load_each_sides_profile_with_its_path(
    router: _Router, move: bool
) -> None:
    # The destination's, which its jobs read, loaded by its handler as it
    # opens. The source's family is its handler's class's.
    loaded: list[str] = []
    for name in ("tree", "flat"):
        fs = _backend(router, name).handler
        original = fs.load_profile

        async def note(name: str = name, original: Any = original) -> Any:
            loaded.append(name)
            return await original()

        fs.load_profile = note
    method = router._mv if move else router._copy
    await method("lake/tree/data", "lake/flat/moved", recursive=True)
    assert loaded == ["flat"]


# What a handler may leave out, a sync router, and routers nested.


class _Everywhere(RoutingHandler):
    """Paths "<storage>/<path>", each naming its storage: none is above them."""

    def __init__(self, *, asynchronous: bool = True) -> None:
        super().__init__(transfer_concurrency=2)
        self.asynchronous = asynchronous
        self.made: list[Any] = []

    def verify_path(self, path: str) -> None:
        """Nothing: any path names its storage."""

    def route(self, path: str) -> tuple[Any, str]:
        name, _, inner = path.partition("/")
        if not name:
            raise FileNotFoundError(path)
        return self.inner(name, self._make), inner

    def _make(self) -> FsspecTreeHandler:
        sim = SimulatedFileSystem(hierarchical=True, asynchronous=self.asynchronous)
        handler = FsspecTreeHandler(sim, storage_profile=UNKNOWN, **_OPTIONS)
        self.made.append(handler)
        return handler

    def outer(self, handler: Any, path: str) -> str:
        return f"s{self.made.index(handler)}/{path}".rstrip("/")


async def test_a_router_whose_every_path_routes_needs_no_roots() -> None:
    router = RoutingFileSystem(_Everywhere(), asynchronous=True)
    with pytest.raises(FileNotFoundError):
        await router._ls("", detail=False)
    fs, _ = router._route("s0/x.txt")
    fs.handler.fs.add_file("x.txt", b"x")
    assert await router._cat_file("s0/x.txt") == b"x"
    assert await router._ls("s0", detail=False) == ["s0/x.txt"]


def test_a_sync_router_runs_its_storages_on_its_loop() -> None:
    router = RoutingFileSystem(_Everywhere(asynchronous=False), asynchronous=False)
    fs, _ = router._route("s0/x.txt")
    # Made in the router's mode, the storage's filesystem runs on its loop.
    assert fs.loop is router.loop
    fs.handler.fs.add_file("x.txt", b"x")
    with router.open("s0/x.txt", "rb") as f:
        assert f.read() == b"x"
    assert router.cat_file("s0/x.txt") == b"x"


class _Region(RoutingHandler):
    """Paths "<region>/<lake path>": each region a lake, its own router."""

    def __init__(self) -> None:
        super().__init__(transfer_concurrency=2)
        self.checked: list[str] = []
        self.lakes: dict[str, _Lake] = {}

    def verify_path(self, path: str) -> None:
        self.checked.append(path)
        if path.startswith("/"):
            raise ValueError(f"Not a region's path: {path!r}")

    def route(self, path: str) -> tuple[Any, str] | None:
        region, _, inner = path.partition("/")
        if not region:
            return None
        if region not in ("eu", "us"):
            raise FileNotFoundError(path)
        lake = self.inner(region, _Lake)
        self.lakes[region] = lake
        return lake, inner

    def outer(self, handler: Any, path: str) -> str:
        region = next(r for r, lake in self.lakes.items() if lake is handler)
        return f"{region}/{path}".rstrip("/")

    async def roots(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        if entry["name"] != "":
            raise FileNotFoundError(entry["name"])
        return [{"name": r, "size": 0, "type": "directory"} for r in ("eu", "us")]


async def test_a_router_may_route_to_another() -> None:
    router = RoutingFileSystem(_Region(), asynchronous=True)
    assert await router._ls("", detail=False) == ["eu", "us"]
    # Above the storages of the inner router: its roots, named by the outer's paths.
    assert await router._ls("eu/lake", detail=False) == [
        "eu/lake/tree",
        "eu/lake/flat",
        "eu/lake/plain",
    ]
    assert (await router._info("eu/lake/tree"))["type"] == "directory"

    eu = router.handler.lakes["eu"]
    eu.tree.add_file("data/x.txt", b"x")
    assert await router._find("eu/lake/tree") == ["eu/lake/tree/data/x.txt"]
    assert await router._cat_file("eu/lake/tree/data/x.txt") == b"x"
    # Each router checks its own path: the outer one, then the inner one.
    assert "eu/lake/tree/data/x.txt" in router.handler.checked
    assert "lake/tree/data/x.txt" in eu.checked
    with pytest.raises(ValueError, match="holds ':'"):
        await router._info("eu/lake/tree/a:b")


async def test_a_copy_between_storages_of_two_nested_routers() -> None:
    router = RoutingFileSystem(_Region(), asynchronous=True)
    router._route("eu/lake/tree")
    router._route("us/lake/flat")
    router.handler.lakes["eu"].tree.add_file("data/x.txt", b"x")
    await router._copy("eu/lake/tree/data", "us/lake/flat/copied", recursive=True)
    us = router.handler.lakes["us"]
    assert us.flat._file_data["copied/x.txt"].data == b"x"
