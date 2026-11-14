"""Tests of `monitored`: a filesystem as it is, or on a handler wrapping it, as its entry says."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from fsspec_alt_strategies.monitor import Call, Monitor, Monitored, monitoring
from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies.wrapper import known_implementations
from fsspec_alt_strategies_impl.common.filesystem import BaseFileSystem
from fsspec_alt_strategies_impl.monitor import monitored
from fsspec_alt_strategies_impl.wrapper.adlfs import AdlfsHandler
from fsspec_alt_strategies_impl.wrapper.filesystem import FileSystemWrapper
from fsspec_alt_strategies_impl.wrapper.flat import FsspecFlatHandler
from fsspec_alt_strategies_impl.wrapper.gcsfs import GcsfsHandler
from fsspec_alt_strategies_impl.wrapper.s3fs import S3fsHandler
from fsspec_alt_strategies_impl.wrapper.tree import FsspecTreeHandler
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from pathlib import Path

_SIMULATED = "fsspec_alt_strategies_impl_testing.SimulatedFileSystem"


def test_a_monitored_filesystem_is_returned_as_it_is() -> None:
    tree = FileSystemWrapper(
        FsspecTreeHandler(
            SimulatedFileSystem(),
            operation_concurrency=1,
            io_concurrency=1,
            storage_profile=StorageProfile(),
        )
    )
    assert monitored(tree) is tree


def test_a_filesystem_is_wrapped_as_its_entry_says() -> None:
    fs = SimulatedFileSystem()
    profile = StorageProfile(implicit_directories=True)
    flat = monitored(
        fs,
        {
            _SIMULATED: {
                "handler": "FsspecFlatHandler",
                "storage_profile": profile,
                "operation_concurrency": 3,
                "io_concurrency": 2,
                "rm_batch_size": 100,
            }
        },
    )
    assert isinstance(flat, BaseFileSystem)
    assert isinstance(flat._handler, FsspecFlatHandler)
    assert flat._handler.fs is fs
    assert (flat._handler.operation_concurrency, flat._handler.io_concurrency) == (3, 2)
    assert (flat._handler.rm_batch_size, flat._handler.profile) == (100, profile)


def test_a_subclass_takes_the_entry_of_its_class() -> None:
    class Mine(SimulatedFileSystem):
        pass

    entry: Any = {
        "handler": "FsspecTreeHandler",
        "storage_profile": StorageProfile(),
        "operation_concurrency": 2,
        "io_concurrency": 1,
    }
    tree = monitored(Mine(), {_SIMULATED: entry})
    assert isinstance(tree, BaseFileSystem)
    assert type(tree._handler) is FsspecTreeHandler


def test_an_unknown_filesystem_is_not_wrapped_by_guess() -> None:
    with pytest.raises(ValueError, match="SimulatedFileSystem"):
        monitored(SimulatedFileSystem(), {})


def test_the_known_local_filesystem(tmp_path: Path) -> None:
    local = LocalFileSystem()
    root = local._strip_protocol(str(tmp_path))
    local.makedirs(f"{root}/data")
    local.pipe_file(f"{root}/data/a.txt", b"a")
    tree = monitored(local)
    assert isinstance(tree, BaseFileSystem)
    assert not tree._handler.asynchronous  # type: ignore[attr-defined]

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
        assert isinstance(tree, BaseFileSystem)
        assert not tree._handler.asynchronous  # type: ignore[attr-defined]
        assert tree.find("/test_monitored") == ["/test_monitored/data/a.txt"]
    finally:
        memory.rm("/test_monitored", recursive=True)


@pytest.mark.parametrize(
    ("name", "own"),
    [
        ("s3fs.core.S3FileSystem", S3fsHandler),
        ("gcsfs.core.GCSFileSystem", GcsfsHandler),
        ("adlfs.spec.AzureBlobFileSystem", AdlfsHandler),
    ],
)
def test_s3fs_gcsfs_and_adlfs_get_their_own_handlers(name: str, own: type) -> None:
    module, _, qualname = name.rpartition(".")
    known = type(qualname, (SimulatedFileSystem,), {"__module__": module})
    known.__qualname__ = qualname

    # A subclass of theirs, as gcsfs's own ExtendedGcsFileSystem is.
    class Extended(known):  # type: ignore[valid-type, misc]
        pass

    assert type(monitored(Extended())._handler) is own


def _entry(handler: str, **rm_batch_size: int) -> Any:
    return {
        "handler": handler,
        "storage_profile": StorageProfile(),
        "operation_concurrency": 1,
        "io_concurrency": 1,
        **rm_batch_size,
    }


@pytest.mark.parametrize(
    ("handler", "cls"),
    [
        ("S3fsHandler", S3fsHandler),
        ("fsspec_alt_strategies_impl.wrapper.gcsfs.GcsfsHandler", GcsfsHandler),
        ("FsspecTreeHandler", FsspecTreeHandler),
    ],
)
def test_an_entry_names_its_handler_built_in_or_by_its_full_name(
    handler: str, cls: type
) -> None:
    flat = {"rm_batch_size": 1} if handler != "FsspecTreeHandler" else {}
    fs = monitored(SimulatedFileSystem(), {_SIMULATED: _entry(handler, **flat)})
    assert type(fs._handler) is cls


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        ("NoSuchHandler", ValueError),
        ("fsspec_alt_strategies_impl.wrapper.s3fs.NoSuchHandler", ValueError),
        ("no_such_module.Handler", ImportError),
        # A class, but none running the strategies.
        ("fsspec.implementations.local.LocalFileSystem", ValueError),
    ],
)
def test_an_entry_naming_no_file_operation_handler_raises(
    handler: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        monitored(SimulatedFileSystem(), {_SIMULATED: _entry(handler)})


def test_every_handler_the_known_implementations_name_is_built_in() -> None:
    named = {e["handler"] for e in known_implementations.values()}
    assert named == {
        "FsspecTreeHandler",
        "S3fsHandler",
        "GcsfsHandler",
        "AdlfsHandler",
    }
    for entry in known_implementations.values():
        flat = (
            {"rm_batch_size": entry["rm_batch_size"]}
            if "rm_batch_size" in entry
            else {}
        )
        own = _entry(entry["handler"], **flat)
        assert isinstance(
            monitored(SimulatedFileSystem(), {_SIMULATED: own}), Monitored
        )


@pytest.mark.parametrize("asynchronous", [True, False])
def test_an_unknown_filesystem_is_wrapped_if_asked(asynchronous: bool) -> None:
    fs = SimulatedFileSystem() if asynchronous else LocalFileSystem()
    tree = monitored(fs, {}, wrap_unknown_fs=True)
    assert isinstance(tree, BaseFileSystem)
    assert tree._handler.asynchronous is asynchronous  # type: ignore[attr-defined]
    assert (tree._handler.operation_concurrency, tree._handler.io_concurrency) == (1, 1)
    assert type(tree._handler) is FsspecTreeHandler
    assert tree._handler.profile == StorageProfile()
