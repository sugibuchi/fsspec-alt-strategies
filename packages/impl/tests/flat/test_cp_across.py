"""Tests of `flat_cp` between two storages, copying the keys under a prefix between two storages."""

from __future__ import annotations

import asyncio
from typing import Any, Self

import pytest
from fsspec.utils import other_paths

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.flat.cp import CpState, flat_cp
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem


class _Reader:
    """A file of a simulated storage read in ranges, a request per read."""

    def __init__(self, fs: SimulatedFileSystem, path: str) -> None:
        self.fs, self.path, self.position = fs, path, 0
        self.size = len(fs._file_data[path].data)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    async def read(self, size: int) -> bytes:
        if self.position >= self.size:
            return b""
        end = self.position + size
        data = await self.fs._cat_file(self.path, start=self.position, end=end)
        self.position += len(data)
        return data


class _Writer:
    """A file of a simulated storage written on closing, by one request."""

    def __init__(self, fs: SimulatedFileSystem, path: str) -> None:
        self.fs, self.path, self.data = fs, path, b""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, *exc: object) -> None:
        if exc_type is None:
            await self.fs._pipe_file(self.path, self.data)

    async def write(self, data: bytes) -> None:
        self.data += data


def streaming_copy(
    open_source: Any, open_destination: Any, *, block_size: int = 1024
) -> Any:
    """A ``cp_file`` between two storages, a block at a time, as the router's."""

    async def copy_file(source: str, destination: str) -> None:
        reader = await open_source(source, "rb")
        async with reader:
            writer = await open_destination(destination, "wb")
            async with writer:
                while block := await reader.read(block_size):
                    await writer.write(block)

    return copy_file


def _opener(fs: SimulatedFileSystem, kind: type[Any]) -> Any:
    async def open_async(path: str, mode: str) -> Any:
        return kind(fs, path)

    return open_async


def _across(
    src: SimulatedFileSystem,
    dst: SimulatedFileSystem,
    path1: Any,
    path2: str,
    **kwargs: Any,
) -> Any:
    """`flat_cp` between two filesystems, streaming each file."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "profile": StorageProfile(),
        "ils": src._ils,
        "info": src._info,
        "cp_file": streaming_copy(
            _opener(src, _Reader), _opener(dst, _Writer), block_size=4
        ),
        "dst_info": dst._info,
        "dst_makedirs": dst._makedirs,
        "dst_rm_file": dst._rm_file,
        "same_storage": False,
    }
    return flat_cp(path1, path2, **{**operations, **kwargs})


_FILES = [
    "data/2023/01/part-0.parquet",
    "data/2023/02/part-0.parquet",
    "data/2024/01/part-0.parquet",
    "data/README.md",
]


def _tree(fs: SimulatedFileSystem, prefix: str = "") -> None:
    for path in _FILES:
        fs.add_file(prefix + path, path.encode())


@pytest.mark.parametrize(
    ("path1", "path2", "dst_dirs"),
    [
        ("data", "copied", []),  # Onto a new path.
        ("data", "archive", ["archive"]),  # Into an existing folder.
        ("data", "new/", []),  # Into a folder, by its trailing "/".
        ("data/", "archive", ["archive"]),  # Its contents onto a folder.
        ("data/README.md", "archive", ["archive"]),  # A file, into a folder.
        ("data", "data", []),  # The same path, on another storage.
    ],
)
async def test_places_as_fsspec(
    flat_fs: SimulatedFileSystem,
    any_fs: SimulatedFileSystem,
    path1: str,
    path2: str,
    dst_dirs: list[str],
) -> None:
    # From an object store, to each kind of destination.
    src, dst = flat_fs, any_fs
    _tree(src, "bucket/")
    for folder in dst_dirs:
        dst.add_file(f"bucket/{folder}/old", b"")
    path1, path2 = f"bucket/{path1}", f"bucket/{path2}"
    root = path1.rstrip("/")
    files = sorted(p for p in src._file_data if p.startswith(root + "/"))
    paths = [root, *files] if files else [root]
    exists = (bool(dst_dirs) or path2.endswith("/")) and not path1.endswith("/")
    expected = dict(zip(paths, other_paths(paths, path2, exists=exists), strict=True))
    data = {p: src._file_data[p].data for p in paths if p in src._file_data}

    copied, errors = await _collect_with_errors(_across(src, dst, path1, path2))
    assert errors == []
    assert sorted(copied) == sorted((p, expected[p]) for p in data)
    for source, content in data.items():
        assert dst._file_data[expected[source]].data == content


async def test_patterns_and_lists_place_as_flat_cp(
    flat_fs: SimulatedFileSystem,
) -> None:
    dst = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(flat_fs)
    copied, errors = await _collect_with_errors(
        _across(flat_fs, dst, "data/*/01", "copied")
    )
    assert errors == []
    assert sorted(copied) == [
        ("data/2023/01/part-0.parquet", "copied/2023/01/part-0.parquet"),
        ("data/2024/01/part-0.parquet", "copied/2024/01/part-0.parquet"),
    ]
    copied, errors = await _collect_with_errors(
        _across(flat_fs, dst, ["data/2023/01", "data/2024/01"], "listed")
    )
    # Two roots of one name: the first is copied, the other reported.
    assert copied == [("data/2023/01/part-0.parquet", "listed/01/part-0.parquet")]
    assert [type(e) for e in errors] == [FileExistsError]


async def test_requests_to_a_tree(flat_fs: SimulatedFileSystem) -> None:
    dst = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(flat_fs)
    op = _across(flat_fs, dst, "data", "copied")
    copied, _ = await _collect_with_errors(op)
    assert len(copied) == 4
    assert op.state == CpState(current=op.state.current, files_found=4, files_copied=4)
    # One listing, and the reads in blocks of 4 bytes: 27 or 14 bytes each.
    assert flat_fs.stats.calls == {"list": 1, "read": 3 * 7 + 4}
    # A folder made once for each folder with keys, however its keys come.
    assert dst.stats.calls == {"info": 1, "mkdir": 4, "write": 4}


async def test_requests_to_an_object_store(flat_fs: SimulatedFileSystem) -> None:
    dst = SimulatedFileSystem(asynchronous=True)
    _tree(flat_fs)
    profile = StorageProfile(implicit_folders=True)
    _, errors = await _collect_with_errors(
        _across(flat_fs, dst, "data", "copied/", profile=profile)
    )
    assert errors == []
    assert dst.stats.calls == {"write": 4}
    assert dst._file_data["copied/data/README.md"].data == b"data/README.md"


async def test_a_marker_makes_its_folder() -> None:
    keys = {"data/": b"", "data/empty/": b"", "data/a.txt": b"a"}
    dst = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    written: list[tuple[str, str]] = []

    async def ils(prefix: str) -> Any:
        for name, data in keys.items():
            if name.startswith(prefix):
                yield {"name": name, "size": len(data), "type": "file"}

    async def info(path: str) -> dict[str, Any]:
        raise FileNotFoundError(path)

    async def copy_file(source: str, destination: str) -> None:
        written.append((source, destination))
        await dst._pipe_file(destination, keys[source])

    profile = StorageProfile(parents_required=False)
    copied, errors = await _collect_with_errors(
        flat_cp(
            "data",
            "copied",
            ils=ils,
            info=info,
            cp_file=copy_file,
            dst_info=dst._info,
            dst_makedirs=dst._makedirs,
            dst_rm_file=dst._rm_file,
            same_storage=False,
            profile=profile,
            concurrency_limit=asyncio.Semaphore(4),
        )
    )
    assert errors == []
    assert copied == written == [("data/a.txt", "copied/a.txt")]
    assert "copied/empty" in dst._folders
    # The markers' folders only: the key's needs none, by the profile.
    assert dst.stats.calls == {"info": 1, "mkdir": 2, "write": 1}


async def test_a_destination_refusing_to_overwrite(
    flat_fs: SimulatedFileSystem,
) -> None:
    dst = SimulatedFileSystem(hierarchical=True, overwrite=False, asynchronous=True)
    _tree(flat_fs)
    dst.add_file("target.md", b"old")
    _, errors = await _collect_with_errors(
        _across(flat_fs, dst, "data/README.md", "target.md")
    )
    assert errors == []
    assert dst._file_data["target.md"].data == b"data/README.md"
    # The refused write, the old file found a file, its deletion, the write again.
    assert dst.stats.calls == {"info": 2, "write": 2, "delete": 1}


async def test_failures_are_yielded_and_the_rest_is_copied(
    flat_fs: SimulatedFileSystem,
) -> None:
    dst = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(flat_fs)
    stream = streaming_copy(_opener(flat_fs, _Reader), _opener(dst, _Writer))

    async def copy_file(source: str, destination: str) -> None:
        if source == "data/README.md":
            raise PermissionError(source)
        await stream(source, destination)

    async def makedirs(path: str, exist_ok: bool = False) -> None:
        if path == "copied/2023/02":
            raise PermissionError(path)
        await dst._makedirs(path, exist_ok=exist_ok)

    copied, errors = await _collect_with_errors(
        _across(
            flat_fs, dst, "data", "copied", cp_file=copy_file, dst_makedirs=makedirs
        )
    )
    assert sorted(e.__notes__[0] for e in errors) == [
        "Copying 'data/README.md' to 'copied/README.md' failed.",
        "Creating the folder 'copied/2023/02' failed.",
    ]
    # The folder failed, but its key is still copied, as the storage allows.
    assert len(copied) == 3
    assert "copied/README.md" not in dst._file_data


async def test_a_missing_source_raises(flat_fs: SimulatedFileSystem) -> None:
    dst = SimulatedFileSystem(asynchronous=True)
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(_across(flat_fs, dst, "missing", "copied"))
    assert dst.stats.calls == {"info": 1}


async def test_cancel_yields_the_copies_sent() -> None:
    src = SimulatedFileSystem(latency=0.01, asynchronous=True)
    dst = SimulatedFileSystem(asynchronous=True)
    for i in range(10):
        src.add_file(f"root/file-{i}", b"x")
    op = _across(
        src,
        dst,
        "root",
        "copied",
        profile=StorageProfile(implicit_folders=True),
        concurrency_limit=asyncio.Semaphore(3),
    )
    copied = []
    async for pair in op:
        copied.append(pair)
        op.cancel()
    assert len(copied) == dst.stats.calls["write"] == 3
    assert op.cancelled()
