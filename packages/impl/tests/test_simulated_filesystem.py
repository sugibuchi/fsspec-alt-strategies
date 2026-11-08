"""Tests of `SimulatedFileSystem`, which the strategies' tests and benchmarks run on."""

from __future__ import annotations

import asyncio

import pytest

from fsspec_alt_strategies_impl_testing import SimulatedFileSystem


def _tree(fs: SimulatedFileSystem) -> None:
    fs.add_file("a/1.txt", b"one")
    fs.add_file("a/b/2.txt", b"two")
    fs.add_file("a/b/c/3.txt", b"three")


# --- Files and folders ------------------------------------------------------


async def test_read_and_list(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    assert await any_fs._cat_file("a/b/2.txt") == b"two"
    assert await any_fs._cat_file("a/b/c/3.txt", start=1, end=3) == b"hr"
    assert await any_fs._ls("a", detail=False) == ["a/1.txt", "a/b"]
    info = await any_fs._info("a/1.txt")
    assert (info["type"], info["size"]) == ("file", 3)
    assert (await any_fs._info("a/b"))["type"] == "directory"


async def test_paths_take_the_protocol_and_a_leading_slash(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    assert await any_fs._cat_file("simulated:///a/1.txt") == b"one"
    assert await any_fs._ls("/", detail=False) == ["a"]


async def test_errors(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await any_fs._cat_file("a/missing.txt")
    with pytest.raises(FileNotFoundError):
        await any_fs._ls("missing")
    with pytest.raises(IsADirectoryError):
        await any_fs._cat_file("a/b")
    with pytest.raises(NotADirectoryError):
        await any_fs._pipe_file("a/1.txt/x", b"")


async def test_write_and_copy(any_fs: SimulatedFileSystem) -> None:
    await any_fs._pipe_file("x/y.txt", b"y")
    await any_fs._cp_file("x/y.txt", "z/y.txt")
    assert await any_fs._cat_file("z/y.txt") == b"y"
    assert any_fs.stats.calls == {"write": 1, "copy": 1, "read": 1}


async def test_flat_folders_exist_only_with_files(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    await flat_fs._mkdir("empty")
    assert not await flat_fs._exists("empty")
    await flat_fs._rm_file("a/b/c/3.txt")
    assert not await flat_fs._exists("a/b/c")
    with pytest.raises(ValueError, match="no empty folders"):
        flat_fs.add_dir("empty")


async def test_hierarchical_folders_are_real(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    await hierarchical_fs._mkdir("empty")
    assert await hierarchical_fs._isdir("empty")
    await hierarchical_fs._rm_file("a/b/c/3.txt")
    assert await hierarchical_fs._isdir("a/b/c")
    with pytest.raises(OSError, match="not empty"):
        await hierarchical_fs._rmdir("a/b")
    with pytest.raises(FileExistsError):
        await hierarchical_fs._mkdir("empty")


# --- Requests ---------------------------------------------------------------


async def test_listing_counts_a_request_per_page() -> None:
    fs = SimulatedFileSystem(page_size=10, asynchronous=True)
    for i in range(25):
        fs.add_file(f"d/{i:02}.txt")
    assert fs.stats.total == 0  # Setting up makes no request.
    assert len(await fs._ls("d")) == 25
    assert fs.stats.calls == {"list": 3}


@pytest.mark.parametrize("hierarchical", [False, True])
async def test_recursive_listing(hierarchical: bool) -> None:
    fs = SimulatedFileSystem(hierarchical=hierarchical, asynchronous=True)
    _tree(fs)
    files = ["a/1.txt", "a/b/2.txt", "a/b/c/3.txt"]
    assert await fs._find("a") == files
    assert fs.stats.calls == {"list": 1}
    assert await fs._find("a", withdirs=True) == sorted([*files, "a", "a/b", "a/b/c"])
    assert await fs._find("a/1.txt") == ["a/1.txt"]
    assert await fs._find("missing") == []


async def test_without_recursive_listing_find_walks() -> None:
    fs = SimulatedFileSystem(
        hierarchical=True, recursive_listing=False, asynchronous=True
    )
    _tree(fs)
    assert await fs._find("a") == ["a/1.txt", "a/b/2.txt", "a/b/c/3.txt"]
    assert fs.stats.calls["list"] == 3  # a, a/b and a/b/c


async def test_flat_rm_lists_then_deletes_each_file(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    await flat_fs._rm("a", recursive=True)
    assert flat_fs.stats.calls == {"list": 1, "delete": 3}
    assert await flat_fs._ls("", detail=False) == []


async def test_hierarchical_rm_deletes_a_folder_at_once(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    await hierarchical_fs._rm("a/b", recursive=True)
    assert hierarchical_fs.stats.calls == {"delete": 1}
    assert await hierarchical_fs._find("", withdirs=True) == ["a", "a/1.txt"]
    with pytest.raises(PermissionError):
        await hierarchical_fs._rm("", recursive=True)


async def test_requests_sent_together_overlap() -> None:
    fs = SimulatedFileSystem(latency=0.01, asynchronous=True)
    for i in range(5):
        fs.add_file(f"{i}.txt")
    await asyncio.gather(*(fs._cat_file(f"{i}.txt") for i in range(5)))
    assert fs.stats.max_concurrency == 5
    fs.reset_stats()
    assert fs.stats.total == 0


async def test_bandwidth_delays_requests_by_their_bytes() -> None:
    # 1 MB at 10 MB/s takes 0.1 s, on top of the latency.
    fs = SimulatedFileSystem(bandwidth=10_000_000, asynchronous=True)
    fs.add_file("big.bin", bytes(1_000_000))
    loop = asyncio.get_running_loop()
    begin = loop.time()
    await fs._cat_file("big.bin")
    assert loop.time() - begin >= 0.09
    begin = loop.time()
    await fs._cat_file("big.bin", start=0, end=1000)
    assert loop.time() - begin < 0.05
    await fs._pipe_file("copy.bin", bytes(500))
    assert fs.stats.transferred == {"read": 1_001_000, "write": 500}
    with pytest.raises(ValueError, match="bandwidth"):
        SimulatedFileSystem(bandwidth=0)


async def test_rm_keeps_to_batch_size(flat_fs: SimulatedFileSystem) -> None:
    for i in range(10):
        flat_fs.add_file(f"d/{i}.txt")
    await flat_fs._rm("d", recursive=True, batch_size=2)
    assert flat_fs.stats.calls["delete"] == 10
    assert flat_fs.stats.max_concurrency == 2


async def test_ils_lists_a_raw_prefix_a_page_at_a_time() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    for name in ("data/2024-01.csv", "data/2024-02.csv", "data/2024-03.csv"):
        fs.add_file(name)
    fs.add_file("data/2025-01.csv")
    listing = fs._ils("data/2024-")
    assert (await anext(listing))["name"] == "data/2024-01.csv"
    assert fs.stats.calls == {"list": 1}  # The next page only once needed.
    names = [e["name"] async for e in listing]
    assert names == ["data/2024-02.csv", "data/2024-03.csv"]
    assert fs.stats.calls == {"list": 2}
    # Exactly a page full asks for no other page.
    fs.reset_stats()
    assert len([e async for e in fs._ils("data/2024-0") if e]) == 3
    assert [e async for e in fs._ils("missing")] == []
    assert fs.stats.calls == {"list": 3}


async def test_rm_files_deletes_several_in_one_request(
    flat_fs: SimulatedFileSystem,
) -> None:
    for name in ("d/a", "d/b", "d/c"):
        flat_fs.add_file(name)
    await flat_fs._rm_files(["d/a", "d/b", "missing"])  # A missing key is no error.
    assert flat_fs.stats.calls == {"delete": 1}
    assert list(flat_fs._file_data) == ["d/c"]


def test_sync_methods() -> None:
    # fsspec makes the sync methods from the async ones, run in its IO thread.
    fs = SimulatedFileSystem(hierarchical=True)
    fs.pipe_file("a/b.txt", b"b")
    assert fs.cat_file("a/b.txt") == b"b"
    fs.rm("a", recursive=True)
    assert fs.stats.calls == {"write": 1, "read": 1, "delete": 1}
