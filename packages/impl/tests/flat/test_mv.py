"""Tests of `_tree_mv` on a storage listing keys: moving each key, or copying it and deleting in batches."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.mv import _tree_mv
from fsspec_alt_strategies_impl_testing import FlatView, SimulatedFileSystem

# An object store's: its directories are implicit, none is made, and it renames
# no file, so a move copies, then deletes in batches.
_FLAT = StorageProfile(implicit_directories=True, renames_files=False)


async def _nothing(*args: Any, **kwargs: Any) -> None:
    pass


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_SOURCE = {"list_prefix", "info", "rm_files", "mv_file", "rm_dir"}
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
    else:
        return FlatView(fs, **source), FlatView(fs, **replaced, **profile)


def _mv(fs: Any, path1: Any, path2: Any, **kwargs: Any) -> Any:
    """`_tree_mv` within ``fs``, whose primitives ``kwargs`` may replace.

    ``cp_entry``, ``rm_files`` and ``batch_size`` given None move by
    ``mv_file``: the profile then says the storage renames.
    """
    kwargs = {"profile": _FLAT, **kwargs}
    batched = ("cp_entry", "rm_files", "batch_size")
    if all(k in kwargs and kwargs[k] is None for k in batched):
        for k in batched:
            del kwargs[k]
        kwargs["profile"] = replace(kwargs["profile"], renames_files=True)
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


def _keys(moved: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The keys moved, sorted: a move yields its directories too."""
    return sorted(m for m in moved if m[0] in _FILES)


def _tree(fs: SimulatedFileSystem) -> None:
    for path in _FILES:
        fs.add_file(path, path.encode())
    fs.add_file("other.txt")


async def test_moves_the_keys(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    moved, errors = await _collect_with_errors(_mv(flat_fs, "data", "moved"))
    assert errors == []
    assert _keys(moved) == [(p, "moved" + p[len("data") :]) for p in _FILES]
    assert sorted(flat_fs._file_data) == [
        "moved" + p[len("data") :] for p in _FILES
    ] + ["other.txt"]
    assert flat_fs._file_data["moved/README.md"].data == b"data/README.md"


async def test_requests() -> None:
    fs = SimulatedFileSystem(page_size=2, asynchronous=True)
    _tree(fs)
    op = _mv(fs, "data", "moved/", batch_size=3)
    await _collect_with_errors(op)
    # 2 pages, a copy per key, and the 4 sources deleted in 2 batches.
    assert fs.stats.calls == {"list": 2, "copy": 4, "delete": 2}
    assert (op.state.files_found, op.state.files_moved) == (4, 4)


async def test_a_pattern_moves_its_matches(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    moved, errors = await _collect_with_errors(_mv(flat_fs, "data/202*", "moved"))
    assert errors == []
    # Relative to the matches' common prefix, "data", as fsspec places them.
    assert sorted(dst for src, dst in moved if src in _FILES) == [
        "moved/2023/01/part-0.parquet",
        "moved/2023/02/part-0.parquet",
        "moved/2024/01/part-0.parquet",
    ]
    # Only the matches went: the README stays.
    assert [p for p in flat_fs._file_data if p.startswith("data/")] == [
        "data/README.md"
    ]


async def test_a_failed_batch_leaves_its_keys_in_both_places(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)

    async def rm_files(paths: list[str]) -> None:
        raise PermissionError("refused")

    moved, errors = await _collect_with_errors(
        _mv(flat_fs, "data", "moved", rm_files=rm_files)
    )
    # Copied, but not deleted: none is moved, and the directories are kept.
    assert moved == []
    assert [type(e) for e in errors] == [PermissionError]
    assert all(p in flat_fs._file_data for p in _FILES)
    assert "moved/README.md" in flat_fs._file_data


async def test_a_key_its_batch_refused_is_yielded_and_the_others_moved(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    refused = "data/2023/02/part-0.parquet"

    async def rm_files(paths: list[str]) -> dict[str, Exception]:
        await flat_fs._rm_files([p for p in paths if p != refused])
        return {refused: PermissionError("retention")} if refused in paths else {}

    op = _mv(flat_fs, "data", "moved", rm_files=rm_files)
    moved, errors = await _collect_with_errors(op)
    assert [m[0] for m in _keys(moved)] == sorted(p for p in _FILES if p != refused)
    assert [type(e) for e in errors] == [PermissionError]
    assert refused in flat_fs._file_data
    assert (op.state.files_moved, op.state.errors) == (len(_FILES) - 1, 1)


async def test_batch_size_must_be_positive(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    # Read once the handler is open, as the operation starts.
    with pytest.raises(ValueError, match="batch_size"):
        await _collect_with_errors(_mv(flat_fs, "data", "moved", batch_size=0))


_BY_MV_FILE = {"cp_entry": None, "rm_files": None, "batch_size": None}


async def test_mv_file_moves_each_key(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    sent: list[tuple[str, str]] = []

    async def mv_file(source: str, destination: str) -> None:
        sent.append((source, destination))  # One request on a renaming storage.
        await flat_fs._mv_file(source, destination)

    op = _mv(flat_fs, "data", "moved", mv_file=mv_file, **_BY_MV_FILE)
    moved, errors = await _collect_with_errors(op)
    assert errors == []
    expected = [(p, "moved" + p[len("data") :]) for p in _FILES]
    assert _keys(moved) == sorted(sent) == expected
    assert (op.state.files_found, op.state.files_moved) == (4, 4)
    assert flat_fs._file_data["moved/README.md"].data == b"data/README.md"
    assert not await flat_fs._exists("data")


@pytest.mark.parametrize(
    ("renames_files", "by_mv_file"), [(True, True), (False, False), (None, True)]
)
async def test_the_profile_chooses_between_both_ways(
    flat_fs: SimulatedFileSystem, renames_files: bool | None, by_mv_file: bool
) -> None:
    _tree(flat_fs)
    renamed: list[str] = []

    async def mv_file(source: str, destination: str) -> None:
        renamed.append(source)
        await flat_fs._mv_file(source, destination)

    profile = StorageProfile(renames_files=renames_files)
    op = _mv(flat_fs, "data", "moved", mv_file=mv_file, profile=profile)
    moved, errors = await _collect_with_errors(op)
    assert errors == []
    assert len(_keys(moved)) == len(_FILES)
    # Unknown, each key moves by mv_file, right on any storage. Declared to
    # rename none, a move copies, and deletes in batches.
    assert bool(renamed) == by_mv_file
    assert flat_fs.stats.calls["delete"] == (len(_FILES) if by_mv_file else 1)


async def test_moves_to_another_storage(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    dst = SimulatedFileSystem(hierarchical=True, asynchronous=True)

    async def cp_entry(
        entry: dict[str, Any],
        destination: str,
        *,
        missing: bool = False,
        parent: Any = None,
    ) -> None:
        source = entry["name"]
        await dst._pipe_file(destination, flat_fs._file_data[source].data)

    op = _mv(
        flat_fs,
        "data",
        "moved",
        cp_entry=cp_entry,
        dst_info=dst._info,
        dst_makedirs=dst._makedirs,
        dst_rm_file=dst._rm_file,
        same_storage=False,
        profile=StorageProfile(),
    )
    moved, errors = await _collect_with_errors(op)
    assert errors == []
    assert _keys(moved) == [(p, "moved" + p[len("data") :]) for p in _FILES]
    assert sorted(dst._file_data) == [d for _, d in _keys(moved)]
    assert sorted(flat_fs._file_data) == ["other.txt"]


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
@pytest.mark.parametrize("by", ["copies", "renames"])
async def test_maxdepth_moves_what_fsspecs_copy_copies(
    flat_fs: SimulatedFileSystem, path1: str, path2: str, maxdepth: int, by: str
) -> None:
    reference = SimulatedFileSystem(asynchronous=True)
    _shallow(flat_fs)
    _shallow(reference)
    path1, path2 = f"bucket/{path1}", f"bucket/{path2}"
    expanded = await AsyncFileSystem._expand_path(
        reference, path1, recursive=True, maxdepth=maxdepth
    )
    await AsyncFileSystem._copy(
        reference, path1, path2, recursive=True, maxdepth=maxdepth
    )
    # fsspec's mv then deletes the whole source prefix, deeper keys too: this
    # deletes only what was copied.
    await reference._rm_files([p for p in expanded if p in reference._file_data])

    kwargs: dict[str, Any] = {}
    if by == "renames":
        kwargs = {"mv_file": flat_fs._mv_file, "cp_entry": None, "rm_files": None}
        kwargs["batch_size"] = None
    _, errors = await _collect_with_errors(
        _mv(flat_fs, path1, path2, maxdepth=maxdepth, **kwargs)
    )
    assert errors == []
    assert {p: f.data for p, f in flat_fs._file_data.items()} == {
        p: f.data for p, f in reference._file_data.items()
    }
    assert await flat_fs._exists("bucket/data/2023/01/y.txt")


async def test_maxdepth_below_1_is_refused(flat_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="maxdepth"):
        _mv(flat_fs, "data", "moved", maxdepth=0)


async def test_mv_entry_is_given_the_listings_entries(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    profile = StorageProfile(implicit_directories=True, renames_files=True)
    op = _mv(flat_fs, "data", "moved", profile=profile)
    _, errors = await _collect_with_errors(op)
    assert errors == []
    # By the source's own _mv_file_entry, each key's entry as the listing gave it.
    assert sorted((e["name"], e["size"]) for e in flat_fs.entries) == [
        (p, len(p)) for p in _FILES
    ]


async def test_renames_by_the_mv_entry_given(flat_fs: SimulatedFileSystem) -> None:
    _tree(flat_fs)
    renamed: list[tuple[str, str]] = []

    async def mv_entry(
        entry: dict[str, Any],
        destination: str,
        *,
        missing: bool = False,
        parent: Any = None,
    ) -> None:
        renamed.append((entry["name"], destination))
        await flat_fs._mv_file(entry["name"], destination)

    view = FlatView(
        flat_fs,
        storage_profile=StorageProfile(implicit_directories=True, renames_files=True),
    )
    moved, errors = await _collect_with_errors(
        _tree_mv(view, view, "data", "moved", mv_entry=mv_entry)
    )
    assert errors == []
    assert sorted(renamed) == _keys(moved)
    assert len(renamed) == len(_FILES)


async def test_markers_are_directories_made_then_deleted() -> None:
    keys = {"data/": b"", "data/x/": b"", "data/x/y/": b"", "data/a.txt": b"a"}
    events: list[str] = []

    async def list_prefix(prefix: str) -> AsyncIterator[dict[str, Any]]:
        for name in sorted(keys):
            if name.startswith(prefix):
                yield {"name": name, "size": len(keys[name]), "type": "file"}

    async def info(path: str) -> dict[str, Any]:
        raise FileNotFoundError(path)

    async def mv_file(source: str, destination: str) -> None:
        events.append(f"mv {destination}")
        keys[destination] = keys.pop(source)

    async def makedirs(path: str, exist_ok: bool = False) -> None:
        events.append(f"mkdir {path}")

    async def rm_dir(path: str) -> None:
        events.append(f"rmdir {path}")

    op = _mv(
        None,
        "data",
        "moved",
        list_prefix=list_prefix,
        info=info,
        mv_file=mv_file,
        rm_dir=rm_dir,
        dst_info=info,
        dst_makedirs=makedirs,
        dst_rm_file=_nothing,
        concurrency_limit=asyncio.Semaphore(4),
        # Directories of their own, made before what goes in them.
        profile=StorageProfile(implicit_directories=False),
        **_BY_MV_FILE,
    )
    _, errors = await _collect_with_errors(op)
    assert errors == []
    # The empty directory, a marker, is made; each source directory deleted
    # after all in it moved.
    assert "mkdir moved/x/y" in events
    assert events.index("rmdir data/x/y") < events.index("rmdir data/x")
    assert events.index("rmdir data/x") < events.index("rmdir data")
    assert events.index("mv moved/a.txt") < events.index("rmdir data")
