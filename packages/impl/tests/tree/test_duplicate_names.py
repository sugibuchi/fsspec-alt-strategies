"""Tests of the tree strategies on a listing naming one file twice, as Google Drive lists two files of one name."""

from __future__ import annotations

from typing import Any

from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.cp import _tree_cp
from fsspec_alt_strategies_impl.tree.mv import _tree_mv
from fsspec_alt_strategies_impl.tree.rm import _tree_rm
from fsspec_alt_strategies_impl_testing import SimulatedFileSystem, TreeView, dir_ils


def _fs() -> SimulatedFileSystem:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    for path in ("t/a/x", "t/a/y", "t/b/z"):
        fs.add_file(path, path.encode())
    return fs


def _twice(fs: SimulatedFileSystem) -> Any:
    """A listing giving t/a/x twice, each with an id of its own, as two items of one name."""
    listed = dir_ils(fs)

    async def ils(path: str, **kwargs: Any) -> Any:
        async for entry in listed(path, **kwargs):
            yield {**entry, "id": 1}
            if entry["name"] == "t/a/x":
                yield {**entry, "id": 2}

    return ils


async def test_a_copy_refuses_the_second_file_of_a_name() -> None:
    fs = _fs()
    view = TreeView(fs, ils=_twice(fs))
    copied, errors = await _collect_with_errors(_tree_cp(view, view, "t", "c"))
    # The first goes, the second is refused: nothing replaces it silently.
    assert sorted(copied) == [
        ("t/a/x", "c/a/x"),
        ("t/a/y", "c/a/y"),
        ("t/b/z", "c/b/z"),
    ]
    assert [type(e) for e in errors] == [FileExistsError]


async def test_a_move_refuses_it_and_keeps_its_directory() -> None:
    fs = _fs()
    moves: list[str] = []

    async def mv_file(path1: str, path2: str) -> None:
        moves.append(path1)
        await fs._mv_file(path1, path2)

    view = TreeView(fs, ils=_twice(fs), mv_file=mv_file)
    moved, errors = await _collect_with_errors(_tree_mv(view, view, "t", "m"))
    assert [type(e) for e in errors] == [FileExistsError]
    assert sorted(moves) == ["t/a/x", "t/a/y", "t/b/z"]
    # The second x stays at the source, and so does its directory, and t: only
    # t/b, all of it moved, is deleted.
    assert await fs._exists("t/a")
    assert not await fs._exists("t/b")
    assert ("t/b", "m/b") in moved
    assert ("t/a", "m/a") not in moved


async def test_rm_deletes_both_and_then_their_directory() -> None:
    fs = _fs()
    view = TreeView(fs, ils=_twice(fs))
    deleted, errors = await _collect_with_errors(_tree_rm(view, "t"))
    # Each entry is deleted, and counted: the directory is finished, then deleted.
    assert errors == []
    assert deleted.count("t/a/x") == 2
    assert deleted[-1] == "t"
    assert not await fs._exists("t")
