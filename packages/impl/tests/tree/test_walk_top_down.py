"""Tests of `_tree_walk` top down on `common.tree`'s traversal, which its caller prunes."""

from __future__ import annotations

from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies_impl.tree.walk import _tree_walk
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet")
    fs.add_file("data/README.md")


@pytest.mark.parametrize("maxdepth", [None, 1, 2])
async def test_yields_what_fsspecs_walk_yields_top_down(
    any_fs: SimulatedFileSystem, maxdepth: int | None
) -> None:
    _tree(any_fs)
    walked = [
        w
        async for w in _tree_walk(
            TreeView(any_fs), "data", max_running=3, maxdepth=maxdepth
        )
    ]
    expected = [
        w
        async for w in AsyncFileSystem._walk(
            any_fs, "data", detail=True, maxdepth=maxdepth
        )
    ]
    assert sorted(walked, key=lambda w: w[0]) == sorted(expected, key=lambda w: w[0])
    paths = [path for path, _, _ in walked]
    for path in paths[1:]:
        assert paths.index(path.rsplit("/", 1)[0]) < paths.index(path)


async def test_the_caller_prunes_by_changing_dirs(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    walked = []
    async for path, dirs, _ in _tree_walk(TreeView(any_fs), "data", max_running=3):
        walked.append(path)
        dirs.pop("2023", None)
        dirs.pop("02", None)
    assert sorted(walked) == ["data", "data/2024", "data/2024/01"]
    assert any_fs.stats.calls == {"list": 3}  # Nothing pruned is listed.


async def test_its_state_counts_the_directories_and_items(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)
    op = _tree_walk(TreeView(any_fs), "data", max_running=3)
    walked = [w async for w in op]
    assert op.state.directories_listed == len(walked) == 7
    assert op.state.found == sum(len(d) + len(f) for _, d, f in walked)
    assert op.state.errors == 0


async def test_cancelled_it_starts_no_listing(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    op = _tree_walk(TreeView(any_fs), "data", max_running=3)
    walked = []
    async for path, _, _ in op:
        walked.append(path)
        op.cancel()  # While the root waits: its subdirectories never start.
    assert walked == ["data"]
    assert op.cancelled()
    assert any_fs.stats.calls == {"list": 1}


async def test_a_directory_failing_yields_its_error(
    any_fs: SimulatedFileSystem,
) -> None:
    _tree(any_fs)

    async def ils(entry: dict[str, Any], **kwargs: Any) -> Any:
        if entry["name"].rstrip("/") == "data/2023":
            raise OSError("data/2023")
        async for e in view_ils(entry, **kwargs):
            yield e

    view = TreeView(any_fs)
    view_ils = view.ils
    view.ils = ils  # type: ignore[method-assign]
    walked = [w async for w in _tree_walk(view, "data", max_running=3)]
    errors = [w for w in walked if isinstance(w, Exception)]
    assert [str(e) for e in errors] == ["data/2023"]
    assert sorted(w[0] for w in walked if not isinstance(w, Exception)) == [
        "data",
        "data/2024",
        "data/2024/01",
        "data/2024/02",
    ]
