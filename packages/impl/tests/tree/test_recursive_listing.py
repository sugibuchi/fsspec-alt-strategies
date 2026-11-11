"""Tests of the tree strategies on a storage with a recursive listing of its own.

Each root is then listed by one recursive ``_ils``, a page per request, split
into the parts a walk gives, so the strategies find what they find a directory
per request.
"""

from __future__ import annotations

from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.cp import _tree_cp
from fsspec_alt_strategies_impl.tree.du import _tree_du
from fsspec_alt_strategies_impl.tree.find import _tree_find
from fsspec_alt_strategies_impl.tree.glob import _tree_glob
from fsspec_alt_strategies_impl.tree.rm import _tree_rm
from fsspec_alt_strategies_impl.tree.walk import _chunks, _tree_walk
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView


def _view(fs: SimulatedFileSystem, **kwargs: Any) -> TreeView:
    return TreeView(fs, recursive_listing=True, **kwargs)


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * 10)
            fs.add_file(f"data/{year}/{month}/part-1.parquet", b"x" * 5)
    fs.add_file("data/README.md")
    fs.add_file("other/x")


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"withdirs": True}, {"maxdepth": 1}, {"maxdepth": 2, "withdirs": True}],
)
@pytest.mark.parametrize("path", ["data", "data/2024", "", "data/README.md", "missing"])
async def test_find_finds_as_fsspec(
    any_fs: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    _tree(any_fs)
    expected = await AsyncFileSystem._find(any_fs, path, **kwargs)

    found, errors = await _collect_with_errors(
        _tree_find(_view(any_fs), path, **kwargs)
    )

    assert errors == []
    assert sorted(e["name"].rstrip("/") for e in found) == sorted(expected)


async def test_find_lists_once(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    any_fs.reset_stats()

    await _collect_with_errors(_tree_find(_view(any_fs), "data"))

    # One recursive listing, where a walk lists each of the 7 directories.
    assert any_fs.stats.calls["list"] == 1


async def test_find_lists_a_page_per_request(any_fs: SimulatedFileSystem) -> None:
    any_fs.page_size = 3
    _tree(any_fs)
    any_fs.reset_stats()

    found, _ = await _collect_with_errors(_tree_find(_view(any_fs), "data"))

    assert len(found) == 9
    assert any_fs.stats.calls["list"] == -(-len(any_fs._listing("data")) // 3)


async def test_walk_gives_each_directory_as_fsspec(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    expected = {
        root: (sorted(dirs), sorted(files))
        async for root, dirs, files in AsyncFileSystem._walk(any_fs, "data")
    }

    for topdown in (True, False):
        walked, errors = await _collect_with_errors(
            _tree_walk(_view(any_fs), "data", topdown=topdown)
        )
        assert errors == []
        assert {
            root: (sorted(dirs), sorted(files)) for root, dirs, files in walked
        } == expected


async def test_du_sums_as_fsspec(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    expected = await AsyncFileSystem._du(any_fs, "data", total=True)

    op = _tree_du(_view(any_fs), "data")
    _, errors = await _collect_with_errors(op)

    assert errors == []
    assert op.state.size == expected


async def test_rm_deletes_all(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    any_fs.reset_stats()

    _, errors = await _collect_with_errors(_tree_rm(_view(any_fs), "data"))

    assert errors == []
    assert not await any_fs._exists("data")
    assert await any_fs._exists("other/x")
    assert any_fs.stats.calls["list"] == 1


async def test_rm_with_maxdepth_deletes_as_fsspec(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)

    _, errors = await _collect_with_errors(
        _tree_rm(_view(hierarchical_fs), "data", maxdepth=1)
    )

    # As fsspec's: the files in data, but no directory with anything in it.
    assert not await hierarchical_fs._exists("data/README.md")
    assert await hierarchical_fs._exists("data/2024/01/part-0.parquet")
    assert errors  # The directories at the limit were not empty.


async def test_cp_copies_all(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    view = _view(any_fs)

    _, errors = await _collect_with_errors(_tree_cp(view, view, "data", "copied"))

    assert errors == []
    assert sorted(await any_fs._find("copied")) == sorted(
        n.replace("data/", "copied/", 1) for n in await any_fs._find("data")
    )


async def test_a_pattern_still_lists_a_directory_per_request(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    any_fs.reset_stats()

    found, _ = await _collect_with_errors(
        _tree_glob(_view(any_fs), "data/2024/*/part-0.parquet")
    )

    # Pruned to the directories the pattern can match: no recursive listing.
    assert len(found) == 2
    assert any_fs.stats.calls["list"] > 1


@pytest.mark.parametrize(
    "kwargs", [{}, {"withdirs": False}, {"maxdepth": 2}, {"maxdepth": 3}]
)
@pytest.mark.parametrize(
    "pattern",
    [
        "data/**",
        "data/**/*.parquet",
        "data/*/**/part-0.parquet",
        "data/2024/**",
        "data/*/0?/**/*",
        "**/x",
        "data/**/02/*",
        "data/*/01/*.parquet",
    ],
)
async def test_glob_matches_as_fsspec(
    any_fs: SimulatedFileSystem, pattern: str, kwargs: dict[str, Any]
) -> None:
    _tree(any_fs)
    expected = await AsyncFileSystem._glob(any_fs, pattern, **kwargs)

    found, errors = await _collect_with_errors(
        _tree_glob(_view(any_fs), pattern, **kwargs)
    )

    assert errors == []
    assert sorted(e["name"].rstrip("/") for e in found) == sorted(expected)


@pytest.mark.parametrize(
    ("pattern", "lists"),
    [
        # At the first ** from the root: the root is listed whole.
        ("data/**/*.parquet", 1),
        # data, then each of its 2 directories, whole.
        ("data/*/**/part-0.parquet", 3),
        # data, its 2 directories, then each of their 4, whole.
        ("data/*/0?/**", 7),
    ],
)
async def test_glob_lists_whole_below_the_first_globstar(
    any_fs: SimulatedFileSystem, pattern: str, lists: int
) -> None:
    _tree(any_fs)
    any_fs.reset_stats()

    await _collect_with_errors(_tree_glob(_view(any_fs), pattern))

    assert any_fs.stats.calls["list"] == lists


async def test_glob_with_maxdepth_lists_a_directory_per_request(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    any_fs.reset_stats()

    await _collect_with_errors(_tree_glob(_view(any_fs), "data/**", maxdepth=2))

    # Its depth prunes below the **: data and its 2 directories.
    assert any_fs.stats.calls["list"] == 3


async def test_a_directory_comes_before_its_own_parts(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    seen: set[str] = {"data"}
    last: list[str] = []

    async for part in _chunks(_view(flat_fs), "data", chunk_size=1):
        assert part.directory in seen
        seen.update(
            e["name"].rstrip("/") for e in part.entries if e["type"] == "directory"
        )
        if part.last:
            last.append(part.directory)

    # Each directory ends once, though an object store lists none of them.
    assert sorted(last) == sorted(
        [
            "data",
            "data/2023",
            "data/2024",
            *(f"data/{y}/{m}" for y in ("2023", "2024") for m in ("01", "02")),
        ]
    )


async def test_entries_apart_from_their_directory_in_name_order(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    # In name order, "data/a-x" and "data/a.txt" come between "data/a" and
    # "data/a/f": "-" and "." sort before "/".
    for path in ("data/a/f", "data/a-x/g", "data/a.txt", "data/b/h"):
        hierarchical_fs.add_file(path)
    hierarchical_fs.add_dir("data/a/empty")
    hierarchical_fs.add_dir("data/a-empty")

    parts = [p async for p in _chunks(_view(hierarchical_fs), "data")]

    entries: dict[str, list[str]] = {}
    for part in parts:
        assert part.first == (part.directory not in entries)
        entries.setdefault(part.directory, []).extend(e["name"] for e in part.entries)
    assert entries == {
        "data": ["data/a", "data/a-empty", "data/a-x", "data/a.txt", "data/b"],
        "data/a": ["data/a/empty", "data/a/f"],
        "data/a/empty": [],
        "data/a-empty": [],
        "data/a-x": ["data/a-x/g"],
        "data/b": ["data/b/h"],
    }
    # Each directory ends once, all below it ending first.
    ended = [p.directory for p in parts if p.last]
    assert sorted(ended) == sorted(entries)
    for directory in ended:
        below = [d for d in ended if d.startswith(directory + "/")]
        assert all(ended.index(d) < ended.index(directory) for d in below)


def _reversed_ils(fs: SimulatedFileSystem) -> Any:
    """A directory's listing in reverse name order: a recursive one is depth first, siblings unsorted."""

    async def ils(path: str, **kwargs: Any) -> Any:
        listing = await fs._ls(path, detail=True, **kwargs)
        for entry in sorted(listing, key=lambda e: e["name"], reverse=True):
            yield entry

    return ils


async def test_depth_first_with_siblings_in_any_order(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    for path in ("data/a/f", "data/a-x/g", "data/a.txt", "data/b/h"):
        hierarchical_fs.add_file(path)
    hierarchical_fs.add_dir("data/a/empty")
    hierarchical_fs.add_dir("data/a-empty")
    view = TreeView(hierarchical_fs, ils=_reversed_ils(hierarchical_fs))

    parts = [p async for p in _chunks(view, "data")]

    entries: dict[str, list[str]] = {}
    for part in parts:
        assert part.first == (part.directory not in entries)
        entries.setdefault(part.directory, []).extend(e["name"] for e in part.entries)
    assert {d: sorted(names) for d, names in entries.items()} == {
        "data": ["data/a", "data/a-empty", "data/a-x", "data/a.txt", "data/b"],
        "data/a": ["data/a/empty", "data/a/f"],
        "data/a/empty": [],
        "data/a-empty": [],
        "data/a-x": ["data/a-x/g"],
        "data/b": ["data/b/h"],
    }
    # Each directory ends once, all below it ending first.
    ended = [p.directory for p in parts if p.last]
    assert sorted(ended) == sorted(entries)
    for directory in ended:
        below = [d for d in ended if d.startswith(directory + "/")]
        assert all(ended.index(d) < ended.index(directory) for d in below)

    found, errors = await _collect_with_errors(_tree_find(view, "data", withdirs=True))
    assert errors == []
    assert sorted(e["name"] for e in found) == sorted(
        await AsyncFileSystem._find(hierarchical_fs, "data", withdirs=True)
    )


async def test_a_directory_ends_before_the_listing_does(
    flat_fs: SimulatedFileSystem,
) -> None:
    _tree(flat_fs)
    parts = [p async for p in _chunks(_view(flat_fs), "data")]

    # Its last part comes as the listing passes it, before data/2024's entries.
    ended = next(
        i for i, p in enumerate(parts) if p.last and p.directory == "data/2023/01"
    )
    later = next(i for i, p in enumerate(parts) if p.directory.startswith("data/2024"))
    assert ended < later


async def test_a_missing_root_raises(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(FileNotFoundError):
        async for _ in _chunks(_view(any_fs), "missing"):
            pass


async def test_a_file_root_raises_not_a_directory(any_fs: SimulatedFileSystem) -> None:
    any_fs.add_file("a/x")
    with pytest.raises(NotADirectoryError):
        async for _ in _chunks(_view(any_fs), "a/x"):
            pass
