"""Tests of the tree strategies on a storage declaring ``lists_recursively``.

Each root is then listed by one recursive ``_ils``, split into the parts a walk
gives, so the strategies find what they find a directory per request.
"""

from __future__ import annotations

from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.cp import _tree_cp
from fsspec_alt_strategies_impl.tree.du import _tree_du
from fsspec_alt_strategies_impl.tree.find import _tree_find
from fsspec_alt_strategies_impl.tree.glob import _tree_glob
from fsspec_alt_strategies_impl.tree.rm import _tree_rm
from fsspec_alt_strategies_impl.tree.walk import _chunks, _tree_walk
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView

_RECURSIVE = StorageProfile(lists_recursively=True)


def _view(fs: SimulatedFileSystem, **kwargs: Any) -> TreeView:
    return TreeView(fs, storage_profile=_RECURSIVE, **kwargs)


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


async def test_a_missing_root_raises(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(FileNotFoundError):
        async for _ in _chunks(_view(any_fs), "missing"):
            pass


async def test_a_file_root_raises_not_a_directory(any_fs: SimulatedFileSystem) -> None:
    any_fs.add_file("a/x")
    with pytest.raises(NotADirectoryError):
        async for _ in _chunks(_view(any_fs), "a/x"):
            pass
