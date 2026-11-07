"""Tests of `tree_mv`, placing paths as fsspec's ``mv`` does."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fsspec.utils import other_paths

from fsspec_alt_strategies.control import collect
from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies.tree.mv import MvState, tree_mv
from fsspec_alt_strategies_testing import SimulatedFileSystem


def _mv(fs: SimulatedFileSystem, path1: Any, path2: Any, **kwargs: Any) -> Any:
    """`tree_mv` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "ls": fs._ls,
        "info": fs._info,
        "cp_file": fs._cp_file,
        "makedirs": fs._makedirs,
        "rm_file": fs._rm_file,
        "rm_folder": fs._rmdir,
    }
    return tree_mv(path1, path2, **{**operations, **kwargs})


_FILES = [
    "data/2023/01/part-0.parquet",
    "data/2023/02/part-0.parquet",
    "data/2024/01/part-0.parquet",
    "data/README.md",
]


def _tree(fs: SimulatedFileSystem) -> None:
    for path in _FILES:
        fs.add_file(path, path.encode())
    fs.add_file("archive/old.txt")
    fs.add_file("other.txt", b"other")


def _expected(fs: SimulatedFileSystem, path1: str, path2: str) -> dict[str, str]:
    """Where fsspec's copy puts each file under ``path1``, from its own rules."""
    root = path1.rstrip("/")
    files = sorted(p for p in fs._file_data if p.startswith(root + "/"))
    # fsspec expands the source with the folder itself first, as here.
    paths = [root, *files] if files else [root]
    dest_is_dir = path2.endswith("/") or path2.rstrip("/") in fs._folders
    exists = dest_is_dir and not path1.endswith("/")
    return dict(zip(paths, other_paths(paths, path2, exists=exists), strict=True))


@pytest.mark.parametrize(
    ("path1", "path2"),
    [
        ("data", "moved"),  # Onto a new path.
        ("data", "archive"),  # Into an existing folder.
        ("data", "new/"),  # Into a folder, by its trailing "/".
        ("data/", "archive"),  # Its contents onto an existing folder.
        ("data/2023", "data/2025"),  # Within the same parent.
        ("data/README.md", "renamed.md"),  # A file, onto a new path.
        ("data/README.md", "archive"),  # A file, into a folder.
    ],
)
async def test_places_as_fsspec(
    any_fs: SimulatedFileSystem, path1: str, path2: str
) -> None:
    # In a bucket: fsspec's other_paths cannot place paths at the top level of
    # a filesystem whose root is "", as it finds no "/" to split them at.
    for path in _FILES:
        any_fs.add_file(f"bucket/{path}", path.encode())
    any_fs.add_file("bucket/archive/old.txt")
    path1, path2 = f"bucket/{path1}", f"bucket/{path2}"
    expected = _expected(any_fs, path1, path2)
    data = {p: any_fs._file_data[p].data for p in expected if p in any_fs._file_data}
    moved, errors = await collect(_mv(any_fs, path1, path2))
    assert errors == []
    for source, destination in expected.items():
        if source in data:
            assert any_fs._file_data[destination].data == data[source]
            assert (source, destination) in moved
    assert not await any_fs._exists(path1.rstrip("/"))


async def test_a_pattern_moves_its_matches(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    moved, errors = await collect(_mv(any_fs, "data/202*", "moved"))
    assert errors == []
    # Relative to the matches' common prefix, "data", as fsspec places them.
    assert await any_fs._exists("moved/2023/01/part-0.parquet")
    assert await any_fs._exists("moved/2024/01/part-0.parquet")
    assert not await any_fs._exists("data/2023")
    assert await any_fs._exists("data/README.md")
    assert ("data/2024/01/part-0.parquet", "moved/2024/01/part-0.parquet") in moved


async def test_a_list_moves_each_under_its_name(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    paths = ["data/2023", "data/2024/01", "other.txt", "data/2024/01/part-0.parquet"]
    moved, errors = await collect(_mv(any_fs, paths, "moved"))
    # A path inside another in the list moves with it.
    assert errors == []
    assert await any_fs._exists("moved/2023/02/part-0.parquet")
    assert await any_fs._exists("moved/01/part-0.parquet")
    assert any_fs._file_data["moved/other.txt"].data == b"other"
    assert ("data/2023", "moved/2023") in moved  # The folder, once empty.
    assert not await any_fs._exists("data/2023")
    assert not await any_fs._exists("data/2024/01")
    assert not await any_fs._exists("other.txt")
    assert await any_fs._exists("data/README.md")


async def test_a_list_leaves_a_path_whose_name_is_taken(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    _, errors = await collect(_mv(any_fs, ["data/*/01"], "moved"))
    assert [type(e) for e in errors] == [FileExistsError]
    assert any_fs._file_data["moved/01/part-0.parquet"].data == (
        b"data/2023/01/part-0.parquet"
    )
    assert not await any_fs._exists("data/2023/01")
    assert await any_fs._exists("data/2024/01/part-0.parquet")


async def test_requests(hierarchical_fs: SimulatedFileSystem) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/empty")
    op = _mv(hierarchical_fs, "data", "moved")
    moved, _ = await collect(op)
    # 4 files, and 7 folders: data, 2023, its 01 and 02, 2024 and its 01, empty.
    assert op.state == MvState(
        current=op.state.current, files_found=4, files_moved=4, folders_moved=7
    )
    assert len(moved) == 4 + 7
    assert moved[-1] == ("data", "moved")
    assert hierarchical_fs.stats.calls == {
        "info": 1,  # Whether "moved" is a folder.
        "list": 7,
        # The folders with files, and the empty one: 2023 and 2024 hold only
        # folders, which create them.
        "mkdir": 5,
        "copy": 4,
        "delete": 4 + 7,
    }
    assert await hierarchical_fs._exists("moved/empty")


async def test_profile_saves_the_mkdirs(hierarchical_fs: SimulatedFileSystem) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/empty")
    profile = StorageProfile(parents_required=False)
    await collect(_mv(hierarchical_fs, "data", "moved", profile=profile))
    # Only the empty folder needs one: the copies create the others.
    assert hierarchical_fs.stats.calls["mkdir"] == 1
    assert await hierarchical_fs._exists("moved/empty")


async def test_a_storage_requiring_parents() -> None:
    fs = SimulatedFileSystem(
        hierarchical=True, parents_required=True, asynchronous=True
    )
    _tree(fs)
    _, errors = await collect(_mv(fs, "data", "new/deep/moved"))
    assert errors == []
    assert await fs._exists("new/deep/moved/2023/01/part-0.parquet")
    # Wrongly declared, the profile makes each copy fail, and keeps the source.
    wrong = StorageProfile(parents_required=False)
    _, errors = await collect(_mv(fs, "new/deep/moved", "elsewhere", profile=wrong))
    assert {type(e) for e in errors} == {FileNotFoundError}
    assert await fs._exists("new/deep/moved/README.md")


async def test_a_storage_refusing_to_overwrite() -> None:
    fs = SimulatedFileSystem(hierarchical=True, overwrite=False, asynchronous=True)
    _tree(fs)
    fs.add_file("moved/README.md", b"old")
    _, errors = await collect(_mv(fs, "data", "moved/"))
    assert errors == []
    assert fs._file_data["moved/data/README.md"].data == b"data/README.md"
    fs.reset_stats()
    fs.add_file("target.txt", b"old")
    await collect(_mv(fs, "other.txt", "target.txt"))
    assert fs._file_data["target.txt"].data == b"other"
    # The refused copy, the old file's deletion, the copy again, the deletion.
    assert fs.stats.calls == {"info": 1, "list": 1, "copy": 2, "delete": 2}


class _Locked(SimulatedFileSystem):
    """Refuses to copy one file, as a storage without permission would."""

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        if self._strip_protocol(path1) == "data/2023/01/part-0.parquet":
            raise PermissionError(path1)
        await super()._cp_file(path1, path2, **kwargs)


async def test_a_failure_keeps_its_source_and_the_rest_moves() -> None:
    fs = _Locked(hierarchical=True, asynchronous=True)
    _tree(fs)
    moved, errors = await collect(_mv(fs, "data", "moved"))
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == [
        "Moving 'data/2023/01/part-0.parquet' to 'moved/2023/01/part-0.parquet' failed."
    ]
    assert await fs._exists("data/2023/01/part-0.parquet")
    assert not await fs._exists("data/2023/02")
    assert not await fs._exists("data/2024")
    assert await fs._exists("moved/2024/01/part-0.parquet")
    assert sum(source in _FILES for source, _ in moved) == 3


class _Undeletable(SimulatedFileSystem):
    """Refuses to delete one file, as a storage without permission would."""

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        if self._strip_protocol(path) == "data/README.md":
            raise PermissionError(path)
        await super()._rm_file(path, **kwargs)


async def test_a_file_copied_but_not_deleted_is_in_both_places() -> None:
    fs = _Undeletable(hierarchical=True, asynchronous=True)
    _tree(fs)
    _, errors = await collect(_mv(fs, "data", "moved"))
    assert [type(e) for e in errors] == [PermissionError]
    assert "the file is in both places" in errors[0].__notes__[0]
    assert await fs._exists("data/README.md")
    assert await fs._exists("moved/README.md")


async def test_closing_waits_for_the_moves_sent() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/file-{i}")
    profile = StorageProfile(parents_required=False)
    async with _mv(
        fs, "root", "moved", profile=profile, concurrency_limit=asyncio.Semaphore(10)
    ) as op:
        async for _ in op:
            break  # The first file: the nine others were sent with it.
    assert fs._in_flight == 0
    # Not yielded, but counted, and none left half done.
    assert op.state.files_moved == 10
    assert fs.stats.calls["copy"] == fs.stats.calls["delete"] == 10


async def test_paths_it_refuses(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(ValueError, match="into itself"):
        await collect(_mv(any_fs, "data", "data/inner"))
    with pytest.raises(FileNotFoundError):
        await collect(_mv(any_fs, "missing", "moved"))
    any_fs.reset_stats()
    assert await collect(_mv(any_fs, "data", "data")) == ([], [])
    assert any_fs.stats.calls == {}


async def test_cancel_yields_the_moves_sent() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/file-{i}")
    profile = StorageProfile(parents_required=False)
    op = _mv(
        fs, "root", "moved", profile=profile, concurrency_limit=asyncio.Semaphore(10)
    )
    moved = []
    async for pair in op:
        moved.append(pair)
        op.cancel()
    # All ten were sent together: each is still yielded, and none after.
    assert len(moved) == fs.stats.calls["copy"] == 10
    assert await fs._exists("root")
    assert op.cancelled()
