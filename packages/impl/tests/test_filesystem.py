"""Tests of `common.filesystem` and `wrapper`: what the families' classes share.

The families test their filesystems through fsspec's methods. These test the
data methods the wrappers pass on, and the helpers shaping fsspec's answers.
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
    _strip_only_protocol,
)
from fsspec_alt_strategies_impl.files.fsspec_async import SyncFileAdapter
from fsspec_alt_strategies_impl.wrapper.flat import AsyncFlatWrapper
from fsspec_alt_strategies_impl.wrapper.tree import AsyncTreeWrapper, SyncTreeWrapper
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

_LIMITS: dict[str, Any] = {
    "operation_concurrency": 2,
    "io_concurrency": 1,
    "storage_profile": StorageProfile(),
}


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


async def test_an_async_wrapper_passes_data_methods_on() -> None:
    inner = _Transfers(asynchronous=True)
    inner.add_file("a", b"0123456789")
    tree = AsyncTreeWrapper(inner, **_LIMITS)
    assert await tree._cat_file("a", start=2, end=5) == b"234"
    await tree._pipe_file("b", b"new")
    assert await inner._cat_file("b") == b"new"
    await tree._get_file("a", "local")
    await tree._put_file("local", "c")
    assert inner.transfers == [("get", "a", "local"), ("put", "local", "c")]
    assert await tree.open_async("a") == ("opened", "a", "rb")
    assert tree._open("a", "wb") == ("open", "a", "wb")
    assert tree.io_limit._value == 1


async def test_a_sync_wrapper_passes_data_methods_on(tmp_path: Path) -> None:
    inner = MemoryFileSystem()
    inner.pipe_file("/w/a", b"0123456789")
    tree = SyncTreeWrapper(inner, asynchronous=True, **_LIMITS)
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
    assert tree.io_limit._value == 1


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


class _Streaming(SyncTreeWrapper):
    """A storage whose files cannot seek."""

    def __init__(self, data: bytes) -> None:
        super().__init__(MemoryFileSystem(), asynchronous=True, **_LIMITS)
        self.data = data
        self.opened: list[_Unseekable] = []

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        self.opened.append(_Unseekable(self.data))
        return SyncFileAdapter(self.opened[-1], mode=mode, size=len(self.data))


async def test_a_range_of_an_unseekable_file_skips_its_start() -> None:
    fs = _Streaming(b"0123456789")
    fs.transfer_size = 3
    cat = _BaseFileSystem._cat_file
    assert await cat(fs, "f", start=7, end=9) == b"78"
    # Read and dropped in chunks no bigger than a transfer.
    assert fs.opened[-1].reads == [3, 3, 1, 2]
    assert await cat(fs, "f", start=-3) == b"789"
    assert await cat(fs, "f", start=4, end=-4) == b"45"
    assert await cat(fs, "f", start=20, end=30) == b""
    assert await cat(fs, "f", start=6, end=2) == b""


async def test_a_local_dir_is_no_transfer(tmp_path: Path) -> None:
    # Both return before opening a file: fsspec makes local directories itself.
    tree = SyncTreeWrapper(MemoryFileSystem(), asynchronous=True, **_LIMITS)
    assert await _BaseFileSystem._get_file(tree, "/w/a", str(tmp_path)) is None
    assert await _BaseFileSystem._put_file(tree, str(tmp_path), "/w/a") is None


def test_plain_paths_keep_a_trailing_slash() -> None:
    tree = SyncTreeWrapper(MemoryFileSystem(), **_LIMITS)
    assert _strip_only_protocol(tree, ["memory://w/a/", "w/b"]) == ["/w/a/", "/w/b"]


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


class _Flat(AsyncFlatWrapper):
    """A simulated storage as a `FlatFileSystem`, listed by its own prefix listing."""

    def _ils(
        self, path: str, recursive: bool = False, prefix: str = "", **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        if recursive:
            return self.fs._list_prefix(f"{path}/{prefix}" if path else prefix)
        else:
            return super()._ils(path, prefix=prefix, **kwargs)


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
        fs: Any = AsyncTreeWrapper(sim, asynchronous=True, **_LIMITS)
    else:
        fs = _Flat(sim, rm_batch_size=10, asynchronous=True, **_LIMITS)

    await getattr(fs, method)(path, "new/", recursive=recursive)
    assert sim._file_data["new/1.txt"].data == b"1"
    assert "new" not in sim._file_data


class _NoPermit(AsyncTreeWrapper):
    """A subclass whose data methods send their request holding no permit of io_limit."""

    async def _cat_file(
        self, path: str, start: Any = None, end: Any = None, **kwargs: Any
    ) -> bytes:
        return await self.fs._cat_file(path, start=start, end=end)

    async def _pipe_file(self, path: str, value: bytes, **kwargs: Any) -> None:
        await self.fs._pipe_file(path, value)


@pytest.mark.parametrize("batch_size", [None, 5])
async def test_many_paths_go_io_concurrency_at_a_time(batch_size: int | None) -> None:
    inner = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    fs = _NoPermit(inner, asynchronous=True, **{**_LIMITS, "io_concurrency": 3})
    paths = [f"many/{i}" for i in range(20)]

    await fs._pipe(dict.fromkeys(paths, b"x"), batch_size=batch_size)
    piped = inner.stats.max_concurrency
    inner.reset_stats()
    assert await fs._cat(paths, batch_size=batch_size) == dict.fromkeys(paths, b"x")

    # io_concurrency by default, though the subclass takes no permit; the call's own if given.
    assert piped == inner.stats.max_concurrency == (batch_size or 3)


def _ranges_fs(**kwargs: Any) -> tuple[SimulatedFileSystem, AsyncTreeWrapper]:
    inner = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    for i, size in enumerate([0, 1000, 300_000, 3 * 1024 * 1024]):
        inner.add_file(
            f"r/{i}", bytes(range(256)) * (size // 256) + b"x" * (size % 256)
        )
    return inner, AsyncTreeWrapper(inner, asynchronous=True, **{**_LIMITS, **kwargs})


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


# The profile, loaded with the paths of the operation first.


class _Noting:
    """Notes each path the profile is ensured with, before a strategy loads it with none."""

    ensured: list[Any]

    async def _ensure_storage_profile(self, path: Any) -> StorageProfile:
        self.ensured.append(path)
        return await super()._ensure_storage_profile(path)  # type: ignore[misc]


class _NotingTree(_Noting, AsyncTreeWrapper):
    pass


class _NotingFlat(_Noting, AsyncFlatWrapper):
    pass


async def _walked(fs: Any, path: str) -> list[Any]:
    return [item async for item in fs._walk(path)]


@pytest.mark.parametrize("family", ["tree", "flat"])
@pytest.mark.parametrize(
    ("call", "paths"),
    [
        (lambda fs: _walked(fs, "data"), ["data"]),
        (lambda fs: fs._find("data"), ["data"]),
        (lambda fs: fs._du("data"), ["data"]),
        (lambda fs: fs._glob("data/*"), ["data/*"]),
        (lambda fs: fs._rm("data", recursive=True), ["data"]),
        (lambda fs: fs._copy("data", "copied", recursive=True), ["data", "copied"]),
        (lambda fs: fs._mv("data", "moved", recursive=True), ["data", "moved"]),
        # fsspec's own, starting no strategy.
        (lambda fs: fs._rm("data/a.txt"), []),
    ],
)
async def test_methods_starting_a_strategy_ensure_the_profile_first(
    family: str, call: Any, paths: list[str]
) -> None:
    inner = SimulatedFileSystem(hierarchical=family == "tree", asynchronous=True)
    inner.add_file("data/a.txt", b"a")
    inner.add_file("data/b/c.txt", b"c")
    if family == "tree":
        fs: Any = _NotingTree(inner, **_LIMITS)
    else:
        fs = _NotingFlat(inner, rm_batch_size=10, **_LIMITS)
    fs.ensured = []
    await call(fs)
    assert fs.ensured == paths


async def test_ensure_storage_profile_is_the_profile_by_default() -> None:
    fs = AsyncTreeWrapper(SimulatedFileSystem(asynchronous=True), **_LIMITS)
    assert await fs._ensure_storage_profile("any/path") == StorageProfile(
        fs_type="tree"
    )
