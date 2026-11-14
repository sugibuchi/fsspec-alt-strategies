"""Tests of `common.filesystem` and `wrapper`: what the families' classes share.

The families test their filesystems through fsspec's methods. These test the
data methods a filesystem passes on to a handler wrapping another filesystem,
and the helpers shaping fsspec's answers.
"""

from __future__ import annotations

import io
import random
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.filesystem import (
    _BaseFileSystem,
    _copy_errors,
    _failed_dir,
    _raise,
)
from fsspec_alt_strategies_impl.common.handler import StorageHandler
from fsspec_alt_strategies_impl.common.paths import fixed_prefix
from fsspec_alt_strategies_impl.files.fsspec_async import SyncFileAdapter
from fsspec_alt_strategies_impl.wrapper.flat import (
    FlatFileSystemWrapper,
    FsspecFlatHandler,
)
from fsspec_alt_strategies_impl.wrapper.tree import (
    FsspecTreeHandler,
    TreeFileSystemWrapper,
)
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

_LIMITS: dict[str, Any] = {
    "operation_concurrency": 2,
    "io_concurrency": 1,
    "storage_profile": StorageProfile(),
}


def _tree(fs: Any, handler: type = FsspecTreeHandler, **options: Any) -> Any:
    """A `TreeFileSystemWrapper` on a handler of `fs`, async: it serves `fs`'s paths."""
    return TreeFileSystemWrapper(
        handler(fs, **{**_LIMITS, **options}), asynchronous=True
    )


class _Transfers(SimulatedFileSystem):
    """A simulated storage noting the transfers passed on to it, and the files opened."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.transfers: list[tuple[str, str, str]] = []

    async def _get_file(self, rpath: str, lpath: str, **kwargs: Any) -> None:
        self.transfers.append(("get", rpath, lpath))

    async def _put_file(self, lpath: str, rpath: str, **kwargs: Any) -> None:
        self.transfers.append(("put", lpath, rpath))

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return ("opened", path, mode)

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        return ("open", path, mode)


async def test_a_handler_of_an_async_filesystem_passes_data_methods_on() -> None:
    inner = _Transfers(asynchronous=True)
    inner.add_file("a", b"0123456789")
    tree = _tree(inner)
    assert await tree._cat_file("a", start=2, end=5) == b"234"
    await tree._pipe_file("b", b"new")
    assert await inner._cat_file("b") == b"new"
    await tree._get_file("a", "local")
    await tree._put_file("local", "c")
    assert inner.transfers == [("get", "a", "local"), ("put", "local", "c")]
    assert await tree.open_async("a") == ("opened", "a", "rb")
    assert tree._open("a", "wb") == ("open", "a", "wb")
    assert tree.handler.io_limit._value == 1


async def test_a_handler_of_a_sync_filesystem_passes_data_methods_on(
    tmp_path: Path,
) -> None:
    inner = MemoryFileSystem()
    inner.pipe_file("/w/a", b"0123456789")
    tree = _tree(inner)
    assert await tree._cat_file("/w/a", start=2, end=5) == b"234"
    await tree._pipe_file("/w/b", b"new")
    assert inner.cat_file("/w/b") == b"new"
    local = tmp_path / "a"
    await tree._get_file("/w/a", str(local))
    assert local.read_bytes() == b"0123456789"
    await tree._put_file(str(local), "/w/c")
    assert inner.cat_file("/w/c") == b"0123456789"
    with tree._open("/w/c", "rb") as f:
        assert f.read() == b"0123456789"
    await tree._cp_file("/w/c", "/w/e")
    await tree._mv_file("/w/c", "/w/d")
    assert sorted(inner.ls("/w", detail=False)) == ["/w/a", "/w/b", "/w/d", "/w/e"]
    assert tree.unstrip_protocol("/w/d") == "memory:///w/d"
    assert tree.handler.io_limit._value == 1


class _Unseekable(io.BytesIO):
    """An SDK's handle streaming its data: it cannot seek, and notes its reads."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.reads: list[int] = []

    def seekable(self) -> bool:
        return False

    def seek(self, *args: Any) -> int:
        raise io.UnsupportedOperation("seek")

    def read(self, size: int | None = -1) -> bytes:
        self.reads.append(-1 if size is None else size)
        return super().read(size)


class _Streaming(FsspecTreeHandler):
    """A storage whose files cannot seek, read through them: by the default `cat_file`."""

    cat_file = StorageHandler.cat_file

    def __init__(self, fs: Any, **options: Any) -> None:
        super().__init__(fs, **options)
        self.data = b"0123456789"
        self.opened: list[_Unseekable] = []

    async def open_async_file(
        self, entry: dict[str, Any], mode: str = "rb", **kwargs: Any
    ) -> Any:
        self.opened.append(_Unseekable(self.data))
        return SyncFileAdapter(self.opened[-1], mode=mode, size=len(self.data))


async def test_a_range_of_an_unseekable_file_skips_its_start() -> None:
    fs = _tree(MemoryFileSystem(), _Streaming)
    handler = fs.handler
    handler.transfer_size = 3
    cat = _BaseFileSystem._cat_file
    assert await cat(fs, "f", start=7, end=9) == b"78"
    # Read and dropped in chunks no bigger than a transfer.
    assert handler.opened[-1].reads == [3, 3, 1, 2]
    assert await cat(fs, "f", start=-3) == b"789"
    assert await cat(fs, "f", start=4, end=-4) == b"45"
    assert await cat(fs, "f", start=20, end=30) == b""
    assert await cat(fs, "f", start=6, end=2) == b""


async def test_a_local_dir_is_no_transfer(tmp_path: Path) -> None:
    # Both return before opening a file: fsspec makes local directories itself.
    tree = _tree(MemoryFileSystem())
    assert await _BaseFileSystem._get_file(tree, "/w/a", str(tmp_path)) is None
    assert await _BaseFileSystem._put_file(tree, str(tmp_path), "/w/a") is None


def test_paths_as_given_keep_a_trailing_slash() -> None:
    tree = _tree(MemoryFileSystem())
    assert tree._as_given(["memory://w/a/", "w/b"]) == ["/w/a/", "/w/b"]
    handler = tree.handler
    assert handler.verified("memory://w/a/", strip_only_protocol=True) == "/w/a/"
    assert handler.entry("w/a/", strip_only_protocol=True) == {"name": "/w/a/"}


def test_a_path_as_given_is_verified_a_pattern_too() -> None:
    class Strict(FsspecTreeHandler):
        def verify_path(self, path: str) -> None:
            if "~" in path:
                raise ValueError(f"Refused: {path!r}")

    handler = Strict(MemoryFileSystem(), **_LIMITS)
    assert handler.verified("memory://w/*.txt", strip_only_protocol=True) == "/w/*.txt"
    for path in ("memory://w/~/", "memory://w/~*"):
        with pytest.raises(ValueError, match="Refused"):
            handler.verified(path, strip_only_protocol=True)


@pytest.mark.parametrize(
    ("path", "prefix"),
    [
        ("abfs://data/2026-*/x.csv", "abfs://data"),
        (
            "abfss://c@acct.dfs.core.windows.net/a/*.csv",
            "abfss://c@acct.dfs.core.windows.net/a",
        ),
        ("abfs://data-*/x.csv", "abfs://"),
        ("file:///*.txt", "file:///"),
        ("/home/me/f?.txt", "/home/me"),
        ("/[ab]", "/"),
        ("a/b/[!c]/d", "a/b"),
        ("a*", ""),
        ("data/x.csv", "data/x.csv"),
    ],
)
def test_the_fixed_prefix_of_a_pattern(path: str, prefix: str) -> None:
    assert fixed_prefix(path) == prefix


def test_a_handler_verifies_a_patterns_fixed_prefix() -> None:
    class Names(FsspecTreeHandler):
        """A storage whose names refuse "*": a glob's pattern is verified by its prefix."""

        def verify_path(self, path: str) -> None:
            if "*" in fixed_prefix(path):
                raise ValueError(f"Refused: {path!r}")

    handler = Names(MemoryFileSystem(), **_LIMITS)
    assert handler.verified("memory://w/*.txt", strip_only_protocol=True) == "/w/*.txt"


def test_a_copy_drops_sources_gone_unless_errors_raise() -> None:
    gone, other = FileNotFoundError("a"), OSError("b")
    assert _copy_errors([gone, other], None) == [other]
    assert _copy_errors([gone, other], "raise") == [gone, other]


def test_errors_are_raised_alone_or_in_a_group() -> None:
    _raise([], "Deleting")
    with pytest.raises(OSError, match=r"^a$"):
        _raise([OSError("a")], "Deleting")
    with pytest.raises(ExceptionGroup, match="Deleting failed for 2 paths"):
        _raise([OSError("a"), OSError("b")], "Deleting")


def test_a_dir_failing_to_list_is_raised_reported_or_empty() -> None:
    error = OSError("a")
    with pytest.raises(OSError, match=r"^a$"):
        _failed_dir("a", error, "raise", detail=False)
    reported: list[Exception] = []
    assert _failed_dir("a", error, reported.append, detail=True) == ("a", {}, {})
    assert reported == [error]
    assert _failed_dir("a", error, "omit", detail=False) == ("a", [], [])


class _Flat(FsspecFlatHandler):
    """A simulated storage as a flat handler, listed by its own prefix listing."""

    def ils_recursive(  # type: ignore[override]
        self, entry: dict[str, Any], *, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        path = entry["name"]
        return self.fs._list_prefix(f"{path}/{prefix}" if path else prefix)


@pytest.mark.parametrize("family", ["tree", "flat"])
@pytest.mark.parametrize("method", ["_copy", "_mv"])
@pytest.mark.parametrize("recursive", [False, True])
@pytest.mark.parametrize("path", ["data/a/1.txt", "data/a/1.*", ["data/a/1.txt"]])
async def test_a_file_copied_or_moved_into_a_missing_dir(
    family: str, method: str, recursive: bool, path: str | list[str]
) -> None:
    sim = SimulatedFileSystem(hierarchical=family == "tree", asynchronous=True)
    sim.add_file("data/a/1.txt", b"1")
    if family == "tree":
        fs: Any = _tree(sim)
    else:
        fs = FlatFileSystemWrapper(
            _Flat(sim, rm_batch_size=10, **_LIMITS), asynchronous=True
        )

    await getattr(fs, method)(path, "new/", recursive=recursive)
    assert sim._file_data["new/1.txt"].data == b"1"
    assert "new" not in sim._file_data


@pytest.mark.parametrize("batch_size", [None, 5])
async def test_many_paths_go_io_concurrency_at_a_time(batch_size: int | None) -> None:
    # The filesystem holds a permit of io_limit around each data call of its
    # handler: a handler need take none, and no batch goes beyond it.
    inner = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    fs = _tree(inner, io_concurrency=3)
    paths = [f"many/{i}" for i in range(20)]

    await fs._pipe(dict.fromkeys(paths, b"x"), batch_size=batch_size)
    piped = inner.stats.max_concurrency
    inner.reset_stats()
    assert await fs._cat(paths, batch_size=batch_size) == dict.fromkeys(paths, b"x")

    # io_concurrency, whatever the call's batch.
    assert piped == inner.stats.max_concurrency == 3


def test_fsspecs_options_are_named() -> None:
    fs = TreeFileSystemWrapper(
        FsspecTreeHandler(MemoryFileSystem(), **_LIMITS),
        use_listings_cache=False,
        listings_expiry_time=5,
        max_paths=3,
        skip_instance_cache=True,  # fsspec's metaclass takes it, before __init__.
    )
    cache = fs.dircache
    assert (cache.use_listings_cache, cache.listings_expiry_time) == (False, 5)
    assert cache.max_paths == 3


@pytest.mark.parametrize("option", ["asynchronus", "batch_size", "anon"])
def test_an_option_not_named_is_refused(option: str) -> None:
    # fsspec's own constructors would ignore it, silently.
    handler = FsspecTreeHandler(MemoryFileSystem(), **_LIMITS)
    with pytest.raises(TypeError, match=option):
        TreeFileSystemWrapper(handler, **{option: True})


def _ranges_fs(**kwargs: Any) -> tuple[SimulatedFileSystem, Any]:
    inner = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    for i, size in enumerate([0, 1000, 300_000, 3 * 1024 * 1024]):
        inner.add_file(
            f"r/{i}", bytes(range(256)) * (size // 256) + b"x" * (size % 256)
        )
    return inner, _tree(inner, **kwargs)


async def test_cat_ranges_answers_as_fsspec() -> None:
    inner, fs = _ranges_fs()
    rng = random.Random(4)
    paths, starts, ends = [], [], []
    for _ in range(300):
        paths.append(rng.choice(["r/0", "r/1", "r/2", "r/3", "r/missing"]))
        start = rng.choice([None, 0, -50, rng.randrange(400_000)])
        starts.append(start)
        ends.append(
            rng.choice(
                [
                    None,
                    -10,
                    rng.randrange(3_200_000),
                    (start or 0) + rng.randrange(5000),
                ]
            )
        )
    ours = await fs._cat_ranges(paths, starts, ends)
    reference = await AsyncFileSystem._cat_ranges(inner, paths, starts, ends)
    for got, expected in zip(ours, reference, strict=True):
        if isinstance(expected, Exception):
            assert type(got) is type(expected)
        else:
            assert got == expected
    with pytest.raises(FileNotFoundError):
        await fs._cat_ranges(paths, starts, ends, on_error="raise")


async def test_cat_ranges_merges_nearby_ranges() -> None:
    inner, fs = _ranges_fs()
    # 100 ranges of 1000 bytes, 2000 apart, in one file: one request, not 100.
    starts = [i * 3000 for i in range(100)]
    ends = [s + 1000 for s in starts]
    inner.reset_stats()
    ours = await fs._cat_ranges(["r/2"] * 100, starts, ends)
    assert ours == [
        inner._file_data["r/2"].data[s:e] for s, e in zip(starts, ends, strict=True)
    ]
    assert inner.stats.calls == {"read": 1}
    # Apart beyond max_gap: one request each.
    inner.reset_stats()
    await fs._cat_ranges(
        ["r/2"] * 3, [0, 100_000, 200_000], [10, 100_010, 200_010], max_gap=1000
    )
    assert inner.stats.calls == {"read": 3}


async def test_cat_ranges_fetches_a_large_merged_range_in_parts() -> None:
    inner, fs = _ranges_fs()
    fs.ranges_split_size = 1024 * 1024
    inner.reset_stats()
    # 3 MiB merged from two ranges: in 3 parts of 1 MiB, at once.
    size = 3 * 1024 * 1024
    ours = await fs._cat_ranges(["r/3", "r/3"], [0, size // 2], [size // 2, size])
    assert b"".join(ours) == inner._file_data["r/3"].data
    assert inner.stats.calls == {"read": 3}


# The profile, loaded once, by the operations that need it.


class _Noting:
    """Counts the loads of the profile."""

    loaded = 0

    async def load_profile(self) -> StorageProfile:
        self.loaded += 1
        return StorageProfile()


class _NotingTree(_Noting, FsspecTreeHandler):
    pass


class _NotingFlat(_Noting, FsspecFlatHandler):
    pass


@pytest.mark.parametrize("family", ["tree", "flat"])
@pytest.mark.parametrize(
    ("call", "loads"),
    [
        # A tree deletion reads its storage's profile; a flat one none.
        (lambda fs: fs._rm("data", recursive=True), {"tree": 1, "flat": 0}),
        # A copy or a move reads the destination's: here, the same.
        (lambda fs: fs._copy("data", "copied", recursive=True), 1),
        (lambda fs: fs._mv("data", "moved", recursive=True), 1),
        # A strategy reading no profile loads none.
        (lambda fs: fs._find("data"), 0),
    ],
)
async def test_strategies_load_the_profile_once(
    family: str, call: Any, loads: Any
) -> None:
    if isinstance(loads, dict):
        loads = loads[family]
    inner = SimulatedFileSystem(hierarchical=family == "tree", asynchronous=True)
    inner.add_file("data/a.txt", b"a")
    inner.add_file("data/b/c.txt", b"c")
    limits = {k: v for k, v in _LIMITS.items() if k != "storage_profile"}
    handler: Any
    if family == "tree":
        handler = _NotingTree(inner, storage_profile=StorageProfile(), **limits)
        fs: Any = TreeFileSystemWrapper(handler, asynchronous=True)
    else:
        handler = _NotingFlat(
            inner, rm_batch_size=10, storage_profile=StorageProfile(), **limits
        )
        fs = FlatFileSystemWrapper(handler, asynchronous=True)
    await call(fs)
    assert handler.loaded == loads


async def test_the_profile_set_is_the_handlers() -> None:
    fs = _tree(SimulatedFileSystem(asynchronous=True))
    await fs.handler.open()
    assert fs.handler.profile == StorageProfile(fs_type="tree")
