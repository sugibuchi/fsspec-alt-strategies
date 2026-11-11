"""Tests of the flat wrappers: existing filesystems, as a `FlatFileSystem`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.local import LocalFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.wrapper import AsyncFlatWrapper, SyncFlatWrapper
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from pathlib import Path

_OPTIONS: dict[str, Any] = {
    "rm_batch_size": 2,
    "operation_concurrency": 4,
    "io_concurrency": 2,
    "storage_profile": StorageProfile(),
}
_FILES = ["data/2023/01/a.txt", "data/2023/02/b.txt", "data/2024/c.txt", "data/d.txt"]


@pytest.fixture
def inner() -> SimulatedFileSystem:
    fs = SimulatedFileSystem(asynchronous=True)
    for path in _FILES:
        fs.add_file(path, path.encode())
    fs.add_file("database.csv")  # Shares the text "data", not the directory.
    return fs


async def test_an_async_filesystem_answers_as_fsspec(
    inner: SimulatedFileSystem,
) -> None:
    flat = AsyncFlatWrapper(inner, **_OPTIONS)
    assert await flat._find("data") == await AsyncFileSystem._find(inner, "data")
    assert await flat._glob("data/*/*.txt") == await AsyncFileSystem._glob(
        inner, "data/*/*.txt"
    )
    listed = flat._ils("data", recursive=True, prefix="202")
    assert [e["name"] async for e in listed] == [
        p for p in sorted(_FILES) if p.startswith("data/202")
    ]
    await flat._mv("data/2023", "moved", recursive=True)
    await flat._rm("data", recursive=True)
    assert sorted(inner._file_data) == [
        "database.csv",
        "moved/01/a.txt",
        "moved/02/b.txt",
    ]


async def test_a_key_gone_meanwhile_is_no_error(inner: SimulatedFileSystem) -> None:
    flat = AsyncFlatWrapper(inner, **_OPTIONS)
    await flat._rm_files(["data/d.txt", "missing.txt"])
    assert not await inner._exists("data/d.txt")


def test_rm_batch_size_must_be_positive(inner: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="rm_batch_size"):
        AsyncFlatWrapper(inner, **{**_OPTIONS, "rm_batch_size": 0})


def test_a_sync_filesystem_answers_as_fsspec(tmp_path: Path) -> None:
    local = LocalFileSystem()
    root = local._strip_protocol(str(tmp_path))
    for path in _FILES:
        local.makedirs(f"{root}/{path}".rpartition("/")[0], exist_ok=True)
        local.pipe_file(f"{root}/{path}", path.encode())
    flat = SyncFlatWrapper(local, **_OPTIONS)
    assert flat.find(f"{root}/data") == local.find(f"{root}/data")
    assert flat.cat_file(f"{root}/data/d.txt") == b"data/d.txt"
    flat.rm(f"{root}/data/2023", recursive=True)
    assert local.find(f"{root}/data") == [
        f"{root}/data/2024/c.txt",
        f"{root}/data/d.txt",
    ]


async def test_a_key_gone_from_a_batch_deletes_the_rest(
    inner: SimulatedFileSystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refused(paths: list[str], **kwargs: Any) -> None:
        raise FileNotFoundError(paths[0])

    monkeypatch.setattr(inner, "_rm", refused)
    flat = AsyncFlatWrapper(inner, **_OPTIONS)
    await flat._rm_files(["data/d.txt", "missing.txt"])
    assert not await inner._exists("data/d.txt")


async def test_a_sync_key_gone_from_a_batch_deletes_the_rest(tmp_path: Path) -> None:
    local = LocalFileSystem()
    root = local._strip_protocol(str(tmp_path))
    local.pipe_file(f"{root}/a.txt", b"a")
    flat = SyncFlatWrapper(local, asynchronous=True, **_OPTIONS)
    # The local rm raises for the missing key: the rest goes one by one.
    await flat._rm_files([f"{root}/a.txt", f"{root}/missing.txt"])
    assert not local.exists(f"{root}/a.txt")
