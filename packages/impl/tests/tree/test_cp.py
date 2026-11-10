"""Tests of `_tree_cp`, placing paths as fsspec's ``copy`` does."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import other_paths

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies.states import CpState
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.cp import _tree_cp
from fsspec_alt_strategies_impl_testing import (
    SimulatedFileSystem,
    TreeView,
)

_PRIMITIVES = {"ils", "info", "rm_file", "rm_dir", "rm_tree", "cp_file", "mv_file"}
"""The keyword arguments that replace a primitive of the filesystem."""


_DESTINATION = {
    "dst_info": "info",
    "dst_makedirs": "makedirs",
    "dst_rm_file": "rm_file",
}
"""The keyword arguments that replace a primitive of the destination, by its name."""


def _views(
    fs: SimulatedFileSystem, kwargs: dict[str, Any]
) -> tuple[TreeView, TreeView]:
    """The source and the destination, both ``fs``, with the primitives ``kwargs`` replace.

    The destination is the source unless a primitive of its own is replaced.
    The limit is the source's, and the profile the destination's.
    """
    limit = {}
    if "concurrency_limit" in kwargs:
        limit["operation_limit"] = kwargs.pop("concurrency_limit")
    profile = {}
    if "profile" in kwargs:
        profile["storage_profile"] = kwargs.pop("profile")

    primitives = {k: kwargs.pop(k) for k in _PRIMITIVES & kwargs.keys()}
    replaced = {
        _DESTINATION[k]: kwargs.pop(k) for k in _DESTINATION.keys() & kwargs.keys()
    }
    if not replaced:
        source = TreeView(fs, **primitives, **limit, **profile)
        return source, source
    source = TreeView(fs, **primitives, **limit)
    return source, TreeView(fs, **replaced, **profile)


def _cp(fs: SimulatedFileSystem, path1: Any, path2: Any, **kwargs: Any) -> Any:
    """`_tree_cp` within ``fs``, whose primitives ``kwargs`` may replace."""
    source, destination = _views(fs, kwargs)
    return _tree_cp(
        source, destination, path1, path2, **{"same_storage": True, **kwargs}
    )


_FILES = [
    "data/2023/01/part-0.parquet",
    "data/2023/02/part-0.parquet",
    "data/2024/01/part-0.parquet",
    "data/README.md",
]


def _tree(fs: SimulatedFileSystem, prefix: str = "") -> None:
    for path in _FILES:
        fs.add_file(prefix + path, path.encode())
    fs.add_file(prefix + "archive/old.txt")


@pytest.mark.parametrize(
    ("path1", "path2"),
    [
        ("data", "copied"),  # Onto a new path.
        ("data", "archive"),  # Into an existing directory.
        ("data", "new/"),  # Into a directory, by its trailing "/".
        ("data/", "archive"),  # Its contents onto an existing directory.
        ("data/README.md", "copied.md"),  # A file, onto a new path.
        ("data/README.md", "archive"),  # A file, into a directory.
    ],
)
async def test_places_as_fsspec(
    any_fs: SimulatedFileSystem, path1: str, path2: str
) -> None:
    # In a bucket: fsspec's other_paths cannot place paths at the top level of
    # a filesystem whose root is "", as it finds no "/" to split them at.
    _tree(any_fs, "bucket/")
    path1, path2 = f"bucket/{path1}", f"bucket/{path2}"
    root = path1.rstrip("/")
    files = sorted(p for p in any_fs._file_data if p.startswith(root + "/"))
    paths = [root, *files] if files else [root]
    dest_is_dir = path2.endswith("/") or path2.rstrip("/") in any_fs._dirs
    exists = dest_is_dir and not path1.endswith("/")
    expected = dict(zip(paths, other_paths(paths, path2, exists=exists), strict=True))
    data = {p: any_fs._file_data[p].data for p in paths if p in any_fs._file_data}

    copied, errors = await _collect_with_errors(_cp(any_fs, path1, path2))
    assert errors == []
    assert sorted(copied) == sorted((p, expected[p]) for p in data)
    for source, content in data.items():
        assert any_fs._file_data[expected[source]].data == content
        assert any_fs._file_data[source].data == content  # The source is kept.


@pytest.mark.parametrize(
    ("pattern", "path2"),
    [
        ("data/*/01", "copied"),  # Directories, relative to their common prefix.
        ("data/20*", "copied"),
        ("data/**/*.parquet", "copied"),  # Files only.
        ("data/*.md", "archive"),  # A single file goes into the destination.
        ("data/2024/*", "copied/"),  # A single directory goes onto it.
    ],
)
async def test_a_pattern_places_as_fsspec(
    any_fs: SimulatedFileSystem, pattern: str, path2: str
) -> None:
    _tree(any_fs, "bucket/")
    pattern, path2 = f"bucket/{pattern}", f"bucket/{path2}"
    paths = await AsyncFileSystem._expand_path(any_fs, pattern, recursive=True)
    # As fsspec's copy: a single path expanded goes into the destination.
    destinations = other_paths(paths, path2, exists=len(paths) == 1)
    expected = dict(zip(paths, destinations, strict=True))
    files = [p for p in paths if p in any_fs._file_data]

    copied, errors = await _collect_with_errors(_cp(any_fs, pattern, path2))
    assert errors == []
    assert sorted(copied) == sorted((p, expected[p]) for p in files)


@pytest.mark.parametrize(
    ("paths", "path2"),
    [
        (["data/README.md", "data/2023/01/part-0.parquet"], "copied"),
        (["data/*.md", "archive/old.txt"], "new/"),
        (["missing", "data/README.md"], "copied"),  # A missing path is skipped.
    ],
)
async def test_a_list_of_files_places_as_fsspec(
    any_fs: SimulatedFileSystem, paths: list[str], path2: str
) -> None:
    _tree(any_fs)
    files = await AsyncFileSystem._expand_path(any_fs, paths, recursive=True)
    expected = dict(zip(files, other_paths(files, path2, flatten=True), strict=True))

    copied, errors = await _collect_with_errors(_cp(any_fs, paths, path2))
    assert errors == []
    assert sorted(copied) == sorted(expected.items())


async def test_a_list_keeps_the_trees_of_its_dirs(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    copied, errors = await _collect_with_errors(
        _cp(any_fs, ["data/2023", "data/2024/*", "data/README.md"], "copied")
    )
    assert errors == []
    # As cp -r places them, where fsspec's copy puts every file in "copied".
    assert sorted(copied) == [
        ("data/2023/01/part-0.parquet", "copied/2023/01/part-0.parquet"),
        ("data/2023/02/part-0.parquet", "copied/2023/02/part-0.parquet"),
        ("data/2024/01/part-0.parquet", "copied/01/part-0.parquet"),
        ("data/README.md", "copied/README.md"),
    ]


@pytest.mark.parametrize(
    ("paths", "destination"),
    [
        (
            ["data/2024/01/part-0.parquet", "data/2023/01/part-0.parquet"],
            "copied/part-0.parquet",
        ),
        (["data/*/01"], "copied/01/part-0.parquet"),  # Directories.
    ],
)
async def test_a_list_copies_only_the_first_of_a_name(
    any_fs: SimulatedFileSystem, paths: list[str], destination: str
) -> None:
    _tree(any_fs)
    copied, errors = await _collect_with_errors(_cp(any_fs, paths, "copied"))
    # The first in path order is copied, as GNU's cp does, never overwritten.
    assert copied == [("data/2023/01/part-0.parquet", destination)]
    assert any_fs._file_data[copied[0][1]].data == b"data/2023/01/part-0.parquet"
    assert [type(e) for e in errors] == [FileExistsError]
    assert "data/2024/01" in errors[0].__notes__[0]


async def test_a_list_matching_nothing_raises(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(_cp(any_fs, ["missing", "data/*/13"], "copied"))


async def test_a_pattern_onto_its_own_matches_is_refused(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    with pytest.raises(ValueError, match="into itself"):
        await _collect_with_errors(_cp(any_fs, "data/*", "data"))


async def test_requests(hierarchical_fs: SimulatedFileSystem) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/empty")
    op = _cp(hierarchical_fs, "data", "copied")
    copied, _ = await _collect_with_errors(op)
    assert len(copied) == 4
    assert op.state == CpState(current=op.state.current, files_found=4, files_copied=4)
    assert hierarchical_fs.stats.calls == {
        "info": 1,  # Whether "copied" is a directory.
        "list": 7,
        # The directories with files, and the empty one: 2023 and 2024 hold only
        # directories, which create them.
        "mkdir": 5,
        "copy": 4,
    }
    assert await hierarchical_fs._exists("copied/empty")

    hierarchical_fs.reset_stats()
    profile = StorageProfile(parents_required=False)
    await _collect_with_errors(_cp(hierarchical_fs, "data", "again", profile=profile))
    assert hierarchical_fs.stats.calls["mkdir"] == 1  # Only the empty directory.


async def test_a_storage_requiring_parents() -> None:
    fs = SimulatedFileSystem(
        hierarchical=True, parents_required=True, asynchronous=True
    )
    _tree(fs)
    _, errors = await _collect_with_errors(_cp(fs, "data", "new/deep/copied"))
    assert errors == []
    assert await fs._exists("new/deep/copied/2023/01/part-0.parquet")
    _, errors = await _collect_with_errors(_cp(fs, "data/README.md", "other/README.md"))
    assert errors == []


async def test_a_storage_refusing_to_overwrite() -> None:
    fs = SimulatedFileSystem(hierarchical=True, overwrite=False, asynchronous=True)
    _tree(fs)
    fs.add_file("target.md", b"old")
    await _collect_with_errors(_cp(fs, "data/README.md", "target.md"))
    assert fs._file_data["target.md"].data == b"data/README.md"
    # The refused copy, the old file found a file, its deletion, the copy again.
    assert fs.stats.calls == {"info": 2, "list": 1, "copy": 2, "delete": 1}


async def test_a_dir_refusing_a_copy_is_not_deleted() -> None:
    fs = SimulatedFileSystem(hierarchical=True, overwrite=False, asynchronous=True)
    fs.add_file("data/README.md")
    fs.add_dir("target/README.md")  # A directory where the file would go.
    deleted: list[str] = []

    async def cp_file(path1: str, path2: str) -> None:
        raise FileExistsError(path2)  # As a storage refusing a directory's path.

    async def rm_file(path: str) -> None:
        deleted.append(path)

    copied, errors = await _collect_with_errors(
        _cp(fs, "data/README.md", "target/", cp_file=cp_file, dst_rm_file=rm_file)
    )
    assert copied == []
    assert [type(e) for e in errors] == [FileExistsError]
    # rm_file is given files only: the directory at the destination is kept.
    assert deleted == []


class _Locked(SimulatedFileSystem):
    """Refuses to copy one file, as a storage without permission would."""

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        if self._strip_protocol(path1) == "data/2023/01/part-0.parquet":
            raise PermissionError(path1)
        await super()._cp_file(path1, path2, **kwargs)


async def test_a_failure_is_yielded_and_the_rest_is_copied() -> None:
    fs = _Locked(hierarchical=True, asynchronous=True)
    _tree(fs)
    copied, errors = await _collect_with_errors(_cp(fs, "data", "copied"))
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == [
        "Copying 'data/2023/01/part-0.parquet' to"
        " 'copied/2023/01/part-0.parquet' failed."
    ]
    assert len(copied) == 3


async def test_paths_it_refuses(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(ValueError, match="Cannot copy 'data' into itself"):
        await _collect_with_errors(_cp(any_fs, "data", "data/inner"))
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(_cp(any_fs, "missing", "copied"))


async def test_cancel_yields_the_copies_sent() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/file-{i}")
    profile = StorageProfile(parents_required=False)
    op = _cp(
        fs, "root", "copied", profile=profile, concurrency_limit=asyncio.Semaphore(10)
    )
    copied = []
    async for pair in op:
        copied.append(pair)
        op.cancel()
    assert len(copied) == fs.stats.calls["copy"] == 10
    assert op.cancelled()


async def test_a_dir_failing_to_be_created_keeps_its_files() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)

    async def makedirs(path: str, exist_ok: bool = False) -> None:
        if path == "copied/2023/01":
            raise PermissionError(path)
        await fs._makedirs(path, exist_ok=exist_ok)

    copied, errors = await _collect_with_errors(
        _cp(fs, "data", "copied", dst_makedirs=makedirs)
    )
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == ["Creating the directory 'copied/2023/01' failed."]
    # The files waiting on the directory are not copied; the others are.
    assert not any(source.startswith("data/2023/01/") for source, _ in copied)
    assert not await fs._exists("copied/2023/01")
    assert await fs._exists("copied/2023/02/part-0.parquet")
    assert await fs._exists("copied/2024/01/part-0.parquet")


def _shallow(fs: SimulatedFileSystem) -> None:
    """A tree with files at each level, so that fsspec places by the root."""
    for path in ["a.txt", "b.txt", "2023/x.txt", "2023/01/y.txt", "2023/01/02/z.txt"]:
        fs.add_file(f"bucket/data/{path}", path.encode())
    fs.add_file("bucket/archive/old.txt")


@pytest.mark.parametrize("maxdepth", [1, 2])
@pytest.mark.parametrize(
    ("path1", "path2"),
    [("data", "copied"), ("data", "archive"), ("data/*", "copied")],
)
async def test_maxdepth_copies_as_fsspec(
    any_fs: SimulatedFileSystem, path1: str, path2: str, maxdepth: int
) -> None:
    reference = SimulatedFileSystem(hierarchical=any_fs.hierarchical, asynchronous=True)
    _shallow(any_fs)
    _shallow(reference)
    path1, path2 = f"bucket/{path1}", f"bucket/{path2}"
    await AsyncFileSystem._copy(
        reference, path1, path2, recursive=True, maxdepth=maxdepth
    )

    _, errors = await _collect_with_errors(_cp(any_fs, path1, path2, maxdepth=maxdepth))
    assert errors == []
    assert {p: f.data for p, f in any_fs._file_data.items()} == {
        p: f.data for p, f in reference._file_data.items()
    }


async def test_maxdepth_requests(hierarchical_fs: SimulatedFileSystem) -> None:
    _shallow(hierarchical_fs)
    hierarchical_fs.add_dir("bucket/data/empty")
    op = _cp(hierarchical_fs, "bucket/data", "bucket/copied", maxdepth=2)
    copied, errors = await _collect_with_errors(op)
    assert errors == []
    assert sorted(source for source, _ in copied) == [
        "bucket/data/2023/x.txt",
        "bucket/data/a.txt",
        "bucket/data/b.txt",
    ]
    # As fsspec's copy with maxdepth, no empty directory is made.
    assert not await hierarchical_fs._exists("bucket/copied/empty")
    hierarchical_fs.stats.calls.pop("info")
    # data and its two directories listed, each directory with files made, a copy per file.
    assert hierarchical_fs.stats.calls == {"list": 3, "mkdir": 2, "copy": 3}


async def test_maxdepth_below_1_is_refused(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="maxdepth"):
        _cp(any_fs, "data", "copied", maxdepth=0)


async def test_a_list_yields_a_name_taken_when_the_rest_is_in_place(
    any_fs: SimulatedFileSystem,
) -> None:
    any_fs.add_file("a/x", b"a")
    any_fs.add_file("b/x", b"b")
    copied, errors = await _collect_with_errors(_cp(any_fs, ["a/x", "b/x"], "a"))
    assert copied == []
    assert [type(e) for e in errors] == [FileExistsError]
    assert any_fs._file_data["a/x"].data == b"a"


def test_a_copy_between_two_storages_needs_cp_file(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    other = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    with pytest.raises(ValueError, match="cp_file"):
        _tree_cp(TreeView(hierarchical_fs), TreeView(other), "data", "copied")
