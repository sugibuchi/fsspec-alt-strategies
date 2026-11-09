"""Tests of `_flat_cp`, placing paths as fsspec's ``copy`` does."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import other_paths

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.flat.cp import CpState, _flat_cp
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem

# A flat storage's: its folders are implicit, and none is made.
_FLAT = StorageProfile(implicit_folders=True)


async def _nothing(*args: Any, **kwargs: Any) -> None:
    pass


_SOURCE = {"iexpand", "info", "rm_files", "mv_file"}
"""The keyword arguments that replace a primitive of the source."""

_DESTINATION = {
    "dst_info": "info",
    "dst_makedirs": "makedirs",
    "dst_rm_file": "rm_file",
}
"""The keyword arguments that replace a primitive of the destination, by its name."""


def _views(fs: Any, kwargs: dict[str, Any]) -> tuple[FlatView, FlatView]:
    """The source and the destination, both ``fs``, with the primitives ``kwargs`` replace.

    The destination is the source unless a primitive of its own is replaced.
    The limit and the batch size are the source's, and the profile the
    destination's.
    """
    source = {k: kwargs.pop(k) for k in _SOURCE & kwargs.keys()}
    if "concurrency_limit" in kwargs:
        source["operation_limit"] = kwargs.pop("concurrency_limit")
    if "batch_size" in kwargs:
        source["rm_batch_size"] = kwargs.pop("batch_size")
    profile = {}
    if "profile" in kwargs:
        profile["storage_profile"] = kwargs.pop("profile")

    replaced = {
        _DESTINATION[k]: kwargs.pop(k) for k in _DESTINATION.keys() & kwargs.keys()
    }
    if not replaced:
        view = FlatView(fs, **source, **profile)
        return view, view
    return FlatView(fs, **source), FlatView(fs, **replaced, **profile)


def _cp(fs: SimulatedFileSystem, path1: Any, path2: Any, **kwargs: Any) -> Any:
    """`_flat_cp` within ``fs``, whose primitives ``kwargs`` may replace."""
    kwargs = {"profile": _FLAT, **kwargs}
    source, destination = _views(fs, kwargs)
    return _flat_cp(
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
        ("data", "archive"),  # Into an existing folder.
        ("data", "new/"),  # Into a folder, by its trailing "/".
        ("data/", "archive"),  # Its contents onto an existing folder.
        ("data/README.md", "copied.md"),  # A file, onto a new path.
        ("data/README.md", "archive"),  # A file, into a folder.
    ],
)
async def test_places_as_fsspec(
    flat_fs: SimulatedFileSystem, path1: str, path2: str
) -> None:
    # In a bucket: fsspec's other_paths cannot place paths at the top level of
    # a filesystem whose root is "", as it finds no "/" to split them at.
    _tree(flat_fs, "bucket/")
    path1, path2 = f"bucket/{path1}", f"bucket/{path2}"
    root = path1.rstrip("/")
    files = sorted(p for p in flat_fs._file_data if p.startswith(root + "/"))
    paths = [root, *files] if files else [root]
    dest_is_dir = path2.endswith("/") or path2.rstrip("/") in flat_fs._folders
    exists = dest_is_dir and not path1.endswith("/")
    expected = dict(zip(paths, other_paths(paths, path2, exists=exists), strict=True))
    data = {p: flat_fs._file_data[p].data for p in paths if p in flat_fs._file_data}

    copied, errors = await _collect_with_errors(_cp(flat_fs, path1, path2))
    assert errors == []
    assert sorted(copied) == sorted((p, expected[p]) for p in data)
    for source, content in data.items():
        assert flat_fs._file_data[expected[source]].data == content
        assert flat_fs._file_data[source].data == content  # The source is kept.


@pytest.mark.parametrize(
    ("pattern", "path2"),
    [
        ("data/*/01", "copied"),  # Folders, relative to their common prefix.
        ("data/20*", "copied"),
        ("data/**/*.parquet", "copied"),  # Files only.
        ("data/*.md", "archive"),  # A single file goes into the destination.
        ("data/2024/*", "copied/"),  # A single folder goes onto it.
    ],
)
async def test_a_pattern_places_as_fsspec(
    flat_fs: SimulatedFileSystem, pattern: str, path2: str
) -> None:
    _tree(flat_fs, "bucket/")
    pattern, path2 = f"bucket/{pattern}", f"bucket/{path2}"
    paths = await AsyncFileSystem._expand_path(flat_fs, pattern, recursive=True)
    destinations = other_paths(paths, path2, exists=len(paths) == 1)
    expected = dict(zip(paths, destinations, strict=True))
    files = [p for p in paths if p in flat_fs._file_data]

    copied, errors = await _collect_with_errors(_cp(flat_fs, pattern, path2))
    assert errors == []
    assert sorted(copied) == sorted((p, expected[p]) for p in files)


async def test_a_list_keeps_the_trees_of_its_folders(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    copied, errors = await _collect_with_errors(
        _cp(flat_fs, ["data/2023", "data/2024/*", "data/README.md"], "copied")
    )
    assert errors == []
    # As cp -r places them, as tree_cp does.
    assert sorted(copied) == [
        ("data/2023/01/part-0.parquet", "copied/2023/01/part-0.parquet"),
        ("data/2023/02/part-0.parquet", "copied/2023/02/part-0.parquet"),
        ("data/2024/01/part-0.parquet", "copied/01/part-0.parquet"),
        ("data/README.md", "copied/README.md"),
    ]


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _cp(fs, "data", "copied/")
    copied, _ = await _collect_with_errors(op)
    assert len(copied) == 4
    # 2 pages, a copy per key: no info for "copied/", and no mkdir.
    assert fs.stats.calls == {"list": 2, "copy": 4}
    assert op.state == CpState(current=op.state.current, files_found=4, files_copied=4)


async def test_paths_it_refuses(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    with pytest.raises(ValueError, match="into itself"):
        await _collect_with_errors(_cp(flat_fs, "data", "data/inner"))
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(_cp(flat_fs, "missing", "copied"))


async def test_markers_are_not_copied() -> None:
    keys = ["data/", "data/empty/", "data/a.txt"]
    copied: list[tuple[str, str]] = []

    async def iexpand(prefix: str) -> Any:
        for name in keys:
            if name.startswith(prefix):
                yield {"name": name, "size": 0, "type": "file"}

    async def info(path: str) -> dict[str, Any]:
        raise FileNotFoundError(path)

    async def cp_file(source: str, destination: str) -> None:
        copied.append((source, destination))

    source, destination = _views(
        None,
        {
            "iexpand": iexpand,
            "info": info,
            "dst_info": info,
            "dst_makedirs": _nothing,
            "dst_rm_file": _nothing,
            "profile": _FLAT,
            "concurrency_limit": asyncio.Semaphore(4),
        },
    )
    op = _flat_cp(
        source, destination, "data", "copied", cp_file=cp_file, same_storage=True
    )
    assert await _collect_with_errors(op) == ([("data/a.txt", "copied/a.txt")], [])
    assert copied == [("data/a.txt", "copied/a.txt")]
    assert op.state.files_found == 1


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
    flat_fs: SimulatedFileSystem, path1: str, path2: str, maxdepth: int
) -> None:
    reference = SimulatedFileSystem(asynchronous=True)
    _shallow(flat_fs)
    _shallow(reference)
    path1, path2 = f"bucket/{path1}", f"bucket/{path2}"
    await AsyncFileSystem._copy(
        reference, path1, path2, recursive=True, maxdepth=maxdepth
    )

    _, errors = await _collect_with_errors(
        _cp(flat_fs, path1, path2, maxdepth=maxdepth)
    )
    assert errors == []
    assert {p: f.data for p, f in flat_fs._file_data.items()} == {
        p: f.data for p, f in reference._file_data.items()
    }


async def test_maxdepth_requests(flat_fs: SimulatedFileSystem) -> None:
    _shallow(flat_fs)
    op = _cp(flat_fs, "bucket/data", "bucket/copied/", maxdepth=2)
    copied, errors = await _collect_with_errors(op)
    assert errors == []
    assert sorted(source for source, _ in copied) == [
        "bucket/data/2023/x.txt",
        "bucket/data/a.txt",
        "bucket/data/b.txt",
    ]
    # One listing of every key under the prefix, and a copy per key kept.
    assert flat_fs.stats.calls == {"list": 1, "copy": 3}


async def test_maxdepth_below_1_is_refused(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="maxdepth"):
        _cp(flat_fs, "data", "copied", maxdepth=0)
