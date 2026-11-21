"""Tests of `common.glob`'s tasks on a dummy tree: the same actions, whichever traverses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies_impl.common.glob import (
    PathAction,
    TraverseSubtree,
    VisitDirectory,
    _Through,
)
from fsspec_alt_strategies_impl.common.patterns import _Search
from fsspec_alt_strategies_impl.common.tree import TreeTraversal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_FILES = ["r/1", "r/a/2", "r/a/b/3", "r/a/b-x/4", "r/c/5"]


def _ils(files: list[str], *, failing: str | None = None) -> Any:
    """A storage's ils: a directory's items, or all files below it, last first."""

    async def ils(entry: dict[str, Any], recursive: bool = False) -> AsyncIterator[Any]:
        directory = entry["name"]
        if directory == failing:
            raise OSError(directory)
        below = [f for f in files if f.startswith(directory + "/")]
        if recursive:
            for f in reversed(below):  # As an object store: files only.
                yield {"name": f, "type": "file", "size": 1}
            return
        names: dict[str, str] = {}
        for f in below:
            head = f[len(directory) + 1 :].split("/")[0]
            name = f"{directory}/{head}"
            names[name] = "file" if name == f else "directory"
        for name, kind in names.items():
            yield {"name": name, "type": kind, "size": 1}

    return ils


def _dir(name: str) -> dict[str, Any]:
    return {"name": name, "type": "directory"}


class _Count(PathAction):
    """Counts the files below each directory, and records the calls."""

    def __init__(self, emit_event: Any) -> None:
        super().__init__(emit_event)
        self.calls: list[tuple[str, str]] = []

    async def map(self, info: dict[str, Any]) -> Any:
        self.calls.append(("map", info["name"]))
        return 1 if info["type"] == "file" else 0

    async def reduce(self, info: dict[str, Any], value: Any, values: list[Any]) -> Any:
        self.calls.append(("reduce", info["name"]))
        await self._emit_event((info["name"], sum(values)))
        return sum(values)


async def _run(
    recursive: bool, root: str = "r", **kwargs: Any
) -> tuple[Any, _Count, list[Any]]:
    engine = TreeTraversal(max_running=2)
    action = _Count(engine._emit_event)
    ils = kwargs.pop("ils", _ils(_FILES))
    task = VisitDirectory(
        engine, _dir(root), action, ils, recursive=recursive, **kwargs
    )
    events = [e async for e in engine.events([task])]
    return task.result(), action, events


@pytest.mark.parametrize("recursive", [False, True])
async def test_each_path_mapped_before_and_reduced_after_its_subtree(
    recursive: bool,
) -> None:
    value, action, events = await _run(recursive)
    assert value == 5
    assert sorted(events) == [
        ("r", 5),
        ("r/a", 3),
        ("r/a/b", 1),
        ("r/a/b-x", 1),
        ("r/c", 1),
    ]
    calls = action.calls
    for name in ["r", "r/a", "r/a/b", "r/a/b-x", "r/c"]:
        start = calls.index(("map", name))
        end = calls.index(("reduce", name))
        below = [i for i, (_, n) in enumerate(calls) if n.startswith(name + "/")]
        assert all(start < i < end for i in below)


async def test_either_task_traverses_alike() -> None:
    _, by_directory, _ = await _run(False)
    engine = TreeTraversal(max_running=1)
    action = _Count(engine._emit_event)
    ils = _ils(_FILES)
    task = TraverseSubtree(engine, _dir("r"), action, ils(_dir("r"), recursive=True))
    [_ async for _ in engine.events([task])]
    assert task.result() == 5
    assert sorted(action.calls) == sorted(by_directory.calls)


@pytest.mark.parametrize("recursive", [False, True])
async def test_patterns_prune_and_choose(recursive: bool) -> None:
    patterns = [_Search("r/a/**", ends_with_sep=False, withdirs=False, maxdepth=None)]
    value, action, _ = await _run(recursive, patterns=patterns)
    # No pattern matches the root: it passes the values below it on, unreduced.
    assert isinstance(value, _Through)
    # Without withdirs, only the files are matched: r/c's is pruned.
    assert value.values == [1, 1, 1]
    mapped = sorted(n for c, n in action.calls if c == "map")
    assert mapped == ["r/a/2", "r/a/b-x/4", "r/a/b/3"]


@pytest.mark.parametrize("recursive", [False, True])
async def test_a_failure_below_skips_the_reduce_above(recursive: bool) -> None:
    engine = TreeTraversal(max_running=2)
    action = _Count(engine._emit_event)
    ils = _ils(_FILES, failing="r/a")  # Listed alone, or whole: it fails either way.
    task = VisitDirectory(engine, _dir("r"), action, ils, recursive=recursive)
    with pytest.raises(BaseExceptionGroup) as raised:
        [_ async for _ in engine.events([task])]
    assert [type(e) for e in raised.value.exceptions] == [OSError]
    reduced = {n for c, n in action.calls if c == "reduce"}
    assert reduced == {"r/c"}  # Its sibling runs on: r/a and r are not reduced.
