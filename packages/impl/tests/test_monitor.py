"""Tests of `monitored`: a filesystem as it is, or wrapped as its entry says."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from fsspec_alt_strategies.monitor import Call, Monitor, monitoring
from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.monitor import monitored
from fsspec_alt_strategies_impl.wrapper.adlfs import AdlfsWrapper
from fsspec_alt_strategies_impl.wrapper.flat import AsyncFlatWrapper
from fsspec_alt_strategies_impl.wrapper.gcsfs import GcsfsWrapper
from fsspec_alt_strategies_impl.wrapper.s3fs import S3fsWrapper
from fsspec_alt_strategies_impl.wrapper.tree import AsyncTreeWrapper, SyncTreeWrapper
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from pathlib import Path

_SIMULATED = "fsspec_alt_strategies_impl_testing.SimulatedFileSystem"


def test_a_monitored_filesystem_is_returned_as_it_is() -> None:
    tree = AsyncTreeWrapper(
        SimulatedFileSystem(),
        operation_concurrency=1,
        io_concurrency=1,
        storage_profile=StorageProfile(),
    )
    assert monitored(tree) is tree


def test_a_filesystem_is_wrapped_as_its_entry_says() -> None:
    fs = SimulatedFileSystem()
    profile = StorageProfile(fs_type="flat", implicit_directories=True)
    flat = monitored(
        fs,
        {
            _SIMULATED: {
                "storage_profile": profile,
                "operation_concurrency": 3,
                "io_concurrency": 2,
                "rm_batch_size": 100,
            }
        },
    )
    assert isinstance(flat, AsyncFlatWrapper)
    assert flat.fs is fs
    assert (flat.operation_concurrency, flat.io_concurrency) == (3, 2)
    assert (flat.rm_batch_size, flat.storage_profile) == (100, profile)


def test_a_subclass_takes_the_entry_of_its_class() -> None:
    class Mine(SimulatedFileSystem):
        pass

    # Of no fs_type: a tree, the way right on any storage.
    entry: Any = {
        "storage_profile": StorageProfile(),
        "operation_concurrency": 2,
        "io_concurrency": 1,
    }
    assert isinstance(monitored(Mine(), {_SIMULATED: entry}), AsyncTreeWrapper)


def test_an_unknown_filesystem_is_not_wrapped_by_guess() -> None:
    with pytest.raises(ValueError, match="SimulatedFileSystem"):
        monitored(SimulatedFileSystem(), {})


def test_the_known_local_filesystem(tmp_path: Path) -> None:
    local = LocalFileSystem()
    root = local._strip_protocol(str(tmp_path))
    local.makedirs(f"{root}/data")
    local.pipe_file(f"{root}/data/a.txt", b"a")
    tree = monitored(local)
    assert isinstance(tree, SyncTreeWrapper)

    class Calls(Monitor):
        def __init__(self) -> None:
            self.calls: list[Call] = []

        async def started(self, call: Call, state: Any, op: Any) -> None:
            self.calls.append(call)

    calls = Calls()
    with monitoring(calls):
        tree.mv(f"{root}/data", f"{root}/moved", recursive=True)
    assert [(c.method, c.fs) for c in calls.calls] == [("mv", local)]
    assert local.cat_file(f"{root}/moved/a.txt") == b"a"


def test_the_known_memory_filesystem() -> None:
    memory = MemoryFileSystem()
    # Its files are shared by every instance: a path of this test's own.
    memory.pipe_file("/test_monitored/data/a.txt", b"a")
    try:
        tree = monitored(memory)
        assert isinstance(tree, SyncTreeWrapper)
        assert tree.find("/test_monitored") == ["/test_monitored/data/a.txt"]
    finally:
        memory.rm("/test_monitored", recursive=True)


@pytest.mark.parametrize(
    ("name", "own"),
    [
        ("s3fs.core.S3FileSystem", S3fsWrapper),
        ("gcsfs.core.GCSFileSystem", GcsfsWrapper),
        ("adlfs.spec.AzureBlobFileSystem", AdlfsWrapper),
    ],
)
def test_s3fs_gcsfs_and_adlfs_get_their_own_wrappers(name: str, own: type) -> None:
    module, _, qualname = name.rpartition(".")
    known = type(qualname, (SimulatedFileSystem,), {"__module__": module})
    known.__qualname__ = qualname

    # A subclass of theirs, as gcsfs's own ExtendedGcsFileSystem is.
    class Extended(known):  # type: ignore[valid-type, misc]
        pass

    assert type(monitored(Extended())) is own
    # Another family than the one it is written for: the generic wrapper.
    other = "tree" if issubclass(own, AsyncFlatWrapper) else "flat"
    entry: Any = {
        "storage_profile": StorageProfile(fs_type=other),
        "operation_concurrency": 1,
        "io_concurrency": 1,
        "rm_batch_size": 1,
    }
    assert type(monitored(Extended(), {name: entry})) in {
        AsyncTreeWrapper,
        AsyncFlatWrapper,
    }


@pytest.mark.parametrize("asynchronous", [True, False])
def test_an_unknown_filesystem_is_wrapped_if_asked(asynchronous: bool) -> None:
    fs = SimulatedFileSystem() if asynchronous else LocalFileSystem()
    tree = monitored(fs, {}, wrap_unknown_fs=True)
    assert isinstance(tree, AsyncTreeWrapper if asynchronous else SyncTreeWrapper)
    assert (tree.operation_concurrency, tree.io_concurrency) == (1, 1)
    assert tree.storage_profile == StorageProfile(fs_type="tree")
