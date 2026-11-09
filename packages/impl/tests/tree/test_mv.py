"""Tests of `_tree_mv`, placing paths as fsspec's ``mv`` does."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem
from fsspec.utils import other_paths

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.mv import MvState, _tree_mv
from fsspec_alt_strategies_impl_testing import (
    SimulatedFileSystem,
    TreeView,
)

_PRIMITIVES = {"ils", "info", "rm_file", "rm_folder", "rm_tree", "cp_file", "mv_file"}
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


def _mv(fs: SimulatedFileSystem, path1: Any, path2: Any, **kwargs: Any) -> Any:
    """`_tree_mv` within ``fs``, whose primitives ``kwargs`` may replace."""
    source, destination = _views(fs, kwargs)
    return _tree_mv(
        source, destination, path1, path2, **{"same_storage": True, **kwargs}
    )


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
    moved, errors = await _collect_with_errors(_mv(any_fs, path1, path2))
    assert errors == []
    for source, destination in expected.items():
        if source in data:
            assert any_fs._file_data[destination].data == data[source]
            assert (source, destination) in moved
    assert not await any_fs._exists(path1.rstrip("/"))


async def test_a_pattern_moves_its_matches(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    moved, errors = await _collect_with_errors(_mv(any_fs, "data/202*", "moved"))
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
    moved, errors = await _collect_with_errors(_mv(any_fs, paths, "moved"))
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
    _, errors = await _collect_with_errors(_mv(any_fs, ["data/*/01"], "moved"))
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
    moved, _ = await _collect_with_errors(op)
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
        # fsspec's _mv_file: a copy, then a deletion.
        "copy": 4,
        "delete": 4 + 7,
    }
    assert await hierarchical_fs._exists("moved/empty")


async def test_a_storage_moving_a_file_in_one_request(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    op = _mv(hierarchical_fs, "data", "moved", mv_file=hierarchical_fs._rename)
    _, errors = await _collect_with_errors(op)
    assert errors == []
    # One request per file, where a copy and a deletion took two.
    assert hierarchical_fs.stats.calls["rename"] == 4
    assert "copy" not in hierarchical_fs.stats.calls
    assert hierarchical_fs.stats.calls["delete"] == 6  # The folders only.
    assert await hierarchical_fs._exists("moved/2023/01/part-0.parquet")


async def test_profile_saves_the_mkdirs(hierarchical_fs: SimulatedFileSystem) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/empty")
    profile = StorageProfile(parents_required=False)
    await _collect_with_errors(_mv(hierarchical_fs, "data", "moved", profile=profile))
    # Only the empty folder needs one: the copies create the others.
    assert hierarchical_fs.stats.calls["mkdir"] == 1
    assert await hierarchical_fs._exists("moved/empty")


async def test_implicit_folders_save_the_folder_requests(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    sent: list[str] = []

    async def rm_folder(path: str) -> None:
        sent.append(f"rmdir {path}")  # Requests a real object store may send.
        await flat_fs._rmdir(path)

    async def makedirs(path: str, exist_ok: bool = False) -> None:
        sent.append(f"mkdir {path}")
        await flat_fs._makedirs(path, exist_ok=exist_ok)

    unknown = _mv(flat_fs, "data", "moved", rm_folder=rm_folder, dst_makedirs=makedirs)
    moved_unknown, _ = await _collect_with_errors(unknown)
    assert sent  # Unknown, each folder is made, then deleted.

    sent.clear()
    profile = StorageProfile(implicit_folders=True)
    op = _mv(
        flat_fs,
        "moved",
        "data",
        rm_folder=rm_folder,
        dst_makedirs=makedirs,
        profile=profile,
    )
    moved, errors = await _collect_with_errors(op)
    assert (sent, errors) == ([], [])
    # The same outcome: the folders moved too, as one found already gone is.
    assert len(moved) == len(moved_unknown)
    assert op.state.folders_moved == unknown.state.folders_moved == 6
    assert sorted(p for p in flat_fs._file_data if p.startswith("data/")) == _FILES


def _files(fs: SimulatedFileSystem) -> dict[str, bytes]:
    return {p: f.data for p, f in sorted(fs._file_data.items())}


def _renaming(fs: SimulatedFileSystem) -> dict[str, Any]:
    """A storage's rename as mv_file, declared to move folders too."""
    return {"mv_file": fs._rename, "profile": StorageProfile(renames_folders=True)}


async def test_a_folder_moves_whole(hierarchical_fs: SimulatedFileSystem) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("data/empty")
    expected = {
        p.replace("data", "moved", 1) if p.startswith("data/") else p: data
        for p, data in _files(hierarchical_fs).items()
    }
    hierarchical_fs.reset_stats()
    op = _mv(hierarchical_fs, "data", "moved", **_renaming(hierarchical_fs))
    assert await _collect_with_errors(op) == ([("data", "moved")], [])
    # Where path2 goes, found free, then the rename: nothing is listed.
    assert hierarchical_fs.stats.calls == {"info": 1, "rename": 1}
    assert op.state == MvState(current=None, folders_moved=0, roots_renamed=1)
    assert _files(hierarchical_fs) == expected
    assert await hierarchical_fs._isdir("moved/empty")


async def test_each_folder_of_a_list_moves_whole(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    op = _mv(
        hierarchical_fs,
        ["data/2023", "data/README.md"],
        "archive",
        **_renaming(hierarchical_fs),
    )
    moved, errors = await _collect_with_errors(op)
    assert errors == []
    assert sorted(moved) == [
        ("data/2023", "archive/2023"),
        ("data/README.md", "archive/README.md"),
    ]
    # The folder whole; the file by its own mv_file, as ever.
    assert op.state.roots_renamed == 1
    assert hierarchical_fs.stats.calls["rename"] == 2
    assert await hierarchical_fs._exists("archive/2023/01/part-0.parquet")
    assert await hierarchical_fs._exists("data/2024/01/part-0.parquet")


@pytest.mark.parametrize(
    ("path1", "path2", "taken", "place"),
    [
        # The target exists: what goes merges into it.
        ("data", "archive", "archive/data/old.txt", "archive/data"),
        # The contents go onto path2.
        ("data/", "moved", None, "moved"),
    ],
)
async def test_a_merge_moves_file_by_file(
    hierarchical_fs: SimulatedFileSystem,
    path1: str,
    path2: str,
    taken: str | None,
    place: str,
) -> None:
    _tree(hierarchical_fs)
    if taken is not None:
        hierarchical_fs.add_file(taken, b"kept")
    op = _mv(hierarchical_fs, path1, path2, **_renaming(hierarchical_fs))
    _, errors = await _collect_with_errors(op)
    assert errors == []
    assert op.state.roots_renamed == 0
    assert hierarchical_fs.stats.calls["rename"] == len(_FILES)
    for source in _FILES:
        target = place + source[len("data") :]
        assert hierarchical_fs._file_data[target].data == source.encode()
    if taken is not None:
        assert hierarchical_fs._file_data[taken].data == b"kept"


async def test_a_folder_failing_to_move_whole_moves_file_by_file(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    # Wrongly declared: fsspec's _mv_file copies a file, and fails on a folder.
    profile = StorageProfile(renames_folders=True)
    op = _mv(flat_fs, "data", "moved", profile=profile)
    _, errors = await _collect_with_errors(op)
    assert errors == []
    assert op.state.roots_renamed == 0
    assert op.state.files_moved == len(_FILES)
    assert not await flat_fs._exists("data")


async def test_unknown_no_folder_is_moved_whole(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    op = _mv(hierarchical_fs, "data", "moved", mv_file=hierarchical_fs._rename)
    _, errors = await _collect_with_errors(op)
    assert errors == []
    assert op.state.roots_renamed is None
    assert hierarchical_fs.stats.calls["rename"] == len(_FILES)


def test_implicit_folders_cannot_require_parents() -> None:
    with pytest.raises(ValueError, match="parents_required"):
        StorageProfile(implicit_folders=True, parents_required=True)


async def test_a_storage_requiring_parents() -> None:
    fs = SimulatedFileSystem(
        hierarchical=True, parents_required=True, asynchronous=True
    )
    _tree(fs)
    _, errors = await _collect_with_errors(_mv(fs, "data", "new/deep/moved"))
    assert errors == []
    assert await fs._exists("new/deep/moved/2023/01/part-0.parquet")
    # Wrongly declared, the profile makes each copy fail, and keeps the source.
    wrong = StorageProfile(parents_required=False)
    _, errors = await _collect_with_errors(
        _mv(fs, "new/deep/moved", "elsewhere", profile=wrong)
    )
    assert {type(e) for e in errors} == {FileNotFoundError}
    assert await fs._exists("new/deep/moved/README.md")


async def test_a_storage_refusing_to_overwrite() -> None:
    fs = SimulatedFileSystem(hierarchical=True, overwrite=False, asynchronous=True)
    _tree(fs)
    fs.add_file("moved/data/README.md", b"old")
    moved, errors = await _collect_with_errors(_mv(fs, "data", "moved"))
    # The file in the way stays, and so does the one refused, as fsspec's mv
    # raises: the others move.
    assert [type(e) for e in errors] == [FileExistsError]
    assert fs._file_data["moved/data/README.md"].data == b"old"
    assert await fs._exists("data/README.md")
    assert sum(source in _FILES for source, _ in moved) == len(_FILES) - 1


class _Locked(SimulatedFileSystem):
    """Refuses to copy one file, as a storage without permission would."""

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        if self._strip_protocol(path1) == "data/2023/01/part-0.parquet":
            raise PermissionError(path1)
        await super()._cp_file(path1, path2, **kwargs)


async def test_a_failure_keeps_its_source_and_the_rest_moves() -> None:
    fs = _Locked(hierarchical=True, asynchronous=True)
    _tree(fs)
    moved, errors = await _collect_with_errors(_mv(fs, "data", "moved"))
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


async def test_a_move_failing_midway_is_reported() -> None:
    fs = _Undeletable(hierarchical=True, asynchronous=True)
    _tree(fs)
    _, errors = await _collect_with_errors(_mv(fs, "data", "moved"))
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == [
        "Moving 'data/README.md' to 'moved/README.md' failed."
    ]
    # fsspec's _mv_file copied it before failing to delete it.
    assert await fs._exists("data/README.md")
    assert await fs._exists("moved/README.md")


async def test_a_source_folder_failing_to_go_is_reported() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)

    async def rm_folder(path: str) -> None:
        if path == "data/2023/01":
            raise PermissionError(path)
        await fs._rmdir(path)

    op = _mv(fs, "data", "moved", rm_folder=rm_folder)
    moved, errors = await _collect_with_errors(op)
    assert [type(e) for e in errors] == [PermissionError]
    assert errors[0].__notes__ == ["Deleting the folder 'data/2023/01' failed."]
    # Its files moved; it and the folders above it stay.
    assert await fs._exists("moved/2023/01/part-0.parquet")
    assert await fs._exists("data/2023/01")
    assert ("data", "moved") not in moved
    assert not await fs._exists("data/2024")


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
        await _collect_with_errors(_mv(any_fs, "data", "data/inner"))
    with pytest.raises(FileNotFoundError):
        await _collect_with_errors(_mv(any_fs, "missing", "moved"))
    any_fs.reset_stats()
    assert await _collect_with_errors(_mv(any_fs, "data", "data")) == ([], [])
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


async def test_a_target_failing_its_info_moves_file_by_file(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    hierarchical_fs.add_dir("moved")  # So that data goes to moved/data.

    async def info(path: str, **kwargs: Any) -> Any:
        if path == "moved/data":
            raise PermissionError(path)
        return await hierarchical_fs._info(path, **kwargs)

    profile = StorageProfile(renames_folders=True)
    op = _mv(
        hierarchical_fs,
        "data",
        "moved",
        dst_info=info,
        mv_file=hierarchical_fs._rename,
        profile=profile,
    )
    _, errors = await _collect_with_errors(op)
    assert errors == []
    # Not known free, the root is not moved whole.
    assert op.state.roots_renamed == 0
    assert op.state.files_moved == len(_FILES)
    assert not await hierarchical_fs._exists("data")
    assert await hierarchical_fs._exists("moved/data/2023/01/part-0.parquet")


async def test_a_source_folder_found_already_gone_is_moved() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)

    async def rm_folder(path: str) -> None:
        await fs._rmdir(path)
        if path == "data/2023/01":
            raise FileNotFoundError(path)  # Deleted by another meanwhile.

    moved, errors = await _collect_with_errors(
        _mv(fs, "data", "moved", rm_folder=rm_folder)
    )
    assert errors == []
    assert ("data/2023/01", "moved/2023/01") in moved
    assert ("data", "moved") in moved
    assert not await fs._exists("data")


async def test_moves_to_another_storage(hierarchical_fs: SimulatedFileSystem) -> None:
    _tree(hierarchical_fs)
    dst = SimulatedFileSystem(hierarchical=True, asynchronous=True)

    async def mv_file(source: str, destination: str) -> None:
        await dst._pipe_file(destination, hierarchical_fs._file_data[source].data)
        await hierarchical_fs._rm_file(source)

    op = _mv(
        hierarchical_fs,
        "data",
        "moved",
        mv_file=mv_file,
        dst_info=dst._info,
        dst_makedirs=dst._makedirs,
        same_storage=False,
        # The destination's: it says nothing of how the source renames.
        profile=StorageProfile(renames_folders=True),
    )
    _, errors = await _collect_with_errors(op)
    assert errors == []
    assert sorted(dst._file_data) == sorted("moved" + p[len("data") :] for p in _FILES)
    assert not any(p.startswith("data/") for p in hierarchical_fs._file_data)
    assert not await hierarchical_fs._exists("data")
    assert op.state.roots_renamed is None


def _shallow(fs: SimulatedFileSystem) -> None:
    """A tree with files at each level, so that fsspec places by the root."""
    for path in ["a.txt", "b.txt", "2023/x.txt", "2023/01/y.txt"]:
        fs.add_file(f"bucket/data/{path}", path.encode())
    fs.add_file("bucket/archive/old.txt")


@pytest.mark.parametrize("maxdepth", [1, 2])
@pytest.mark.parametrize(
    ("path1", "path2"),
    [("data", "moved"), ("data", "archive"), ("data/*", "moved")],
)
@pytest.mark.parametrize("renames_folders", [False, True])
async def test_maxdepth_moves_what_fsspecs_copy_copies(
    hierarchical_fs: SimulatedFileSystem,
    path1: str,
    path2: str,
    maxdepth: int,
    renames_folders: bool,
) -> None:
    fs = hierarchical_fs
    reference = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _shallow(fs)
    _shallow(reference)
    path1, path2 = f"bucket/{path1}", f"bucket/{path2}"
    expanded = await AsyncFileSystem._expand_path(
        reference, path1, recursive=True, maxdepth=maxdepth
    )
    await AsyncFileSystem._copy(
        reference, path1, path2, recursive=True, maxdepth=maxdepth
    )
    # fsspec's mv then deletes the whole source tree, deeper files too: this
    # deletes only what was copied.
    for path in expanded:
        if path in reference._file_data:
            await reference._rm_file(path)

    kwargs: dict[str, Any] = {}
    if renames_folders:
        kwargs = {
            "mv_file": fs._rename,
            "profile": StorageProfile(renames_folders=True),
        }
    _, errors = await _collect_with_errors(
        _mv(fs, path1, path2, maxdepth=maxdepth, **kwargs)
    )
    assert errors == []
    assert _files(fs) == _files(reference)
    # A folder keeping a file deeper than maxdepth stays where it was.
    assert await fs._exists("bucket/data/2023/01/y.txt")


async def test_maxdepth_keeps_the_folders_not_emptied(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _shallow(hierarchical_fs)
    hierarchical_fs.add_dir("bucket/data/empty")
    op = _mv(hierarchical_fs, "bucket/data", "bucket/moved", maxdepth=2)
    moved, errors = await _collect_with_errors(op)
    assert errors == []
    assert sorted(source for source, _ in moved) == [
        "bucket/data/2023/x.txt",
        "bucket/data/a.txt",
        "bucket/data/b.txt",
    ]
    hierarchical_fs.stats.calls.pop("info")
    # No folder is deleted: 2023 keeps 01, data keeps 2023 and the empty one,
    # which is not made at the destination, as fsspec's copy makes none.
    assert hierarchical_fs.stats.calls == {
        "list": 3,
        "mkdir": 2,
        "copy": 3,
        "delete": 3,
    }
    assert await hierarchical_fs._exists("bucket/data/empty")
    assert not await hierarchical_fs._exists("bucket/moved/empty")


async def test_maxdepth_below_1_is_refused(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="maxdepth"):
        _mv(any_fs, "data", "moved", maxdepth=0)
