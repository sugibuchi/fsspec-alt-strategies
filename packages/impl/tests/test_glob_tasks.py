"""Tests of `common.glob`'s tasks on a dummy tree: the same actions, whichever traverses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies_impl.common.glob import (
    PathAction,
    TraverseSubtree,
    VisitDirectory,
)
from fsspec_alt_strategies_impl.common.handler import _PathHandler
from fsspec_alt_strategies_impl.common.patterns import _Search
from fsspec_alt_strategies_impl.common.tree import TreeTraversal

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_FILES = ["r/1", "r/a/2", "r/a/b/3", "r/a/b-x/4", "r/c/5"]


class _Handler:
    """A storage's handler on a dummy tree: a directory's items, or all files below it, last first.

    With ``slashes``, it lists as an object store may: each subdirectory as a
    prefix and as its marker, and the directory's own marker.
    """

    delimiter = "/"
    equals = _PathHandler.equals
    _split = _PathHandler._split

    def __init__(
        self, files: list[str], *, failing: str | None = None, slashes: bool = False
    ) -> None:
        self.files = files
        self.failing = failing
        self.slashes = slashes

    def join(self, directory: str, *names: str) -> str:
        return "/".join([directory, *names])

    async def ils(
        self, entry: dict[str, Any], recursive: bool = False
    ) -> AsyncIterator[Any]:
        directory = entry["name"].rstrip("/")
        if directory == self.failing:
            raise OSError(directory)
        below = [f for f in self.files if f.startswith(directory + "/")]
        if recursive:
            for f in reversed(below):  # As an object store: files only.
                yield {"name": f, "type": "file", "size": 1}
            return
        if self.slashes:
            yield {"name": directory + "/", "type": "directory", "size": 0}
        names: dict[str, str] = {}
        for f in below:
            head = f[len(directory) + 1 :].split("/")[0]
            name = f"{directory}/{head}"
            names[name] = "file" if name == f else "directory"
        for name, kind in names.items():
            yield {"name": name, "type": kind, "size": 1}
            if self.slashes and kind == "directory":
                yield {"name": name + "/", "type": "directory", "size": 0}


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
    handler = kwargs.pop("handler", _Handler(_FILES))
    task = VisitDirectory(
        engine, _dir(root), action, handler, recursive=recursive, **kwargs
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
    handler = _Handler(_FILES)
    listing = handler.ils(_dir("r"), recursive=True)
    task = TraverseSubtree(engine, _dir("r"), action, handler, listing)
    [_ async for _ in engine.events([task])]
    assert task.result() == 5
    assert sorted(action.calls) == sorted(by_directory.calls)


@pytest.mark.parametrize("recursive", [False, True])
async def test_patterns_only_prune(recursive: bool) -> None:
    patterns = [_Search("r/a/**", ends_with_sep=False, withdirs=False, maxdepth=None)]
    value, action, _ = await _run(recursive, patterns=patterns)
    # Every path visited is acted on: r/c is found, and mapped, but not entered.
    assert value == 4
    mapped = sorted(n for c, n in action.calls if c == "map")
    assert mapped == [
        "r",
        "r/1",
        "r/a",
        "r/a/2",
        "r/a/b",
        "r/a/b-x",
        "r/a/b-x/4",
        "r/a/b/3",
        "r/c",
    ]
    reduced = sorted(n for c, n in action.calls if c == "reduce")
    assert reduced == ["r", "r/a", "r/a/b", "r/a/b-x"]


@pytest.mark.parametrize("recursive", [False, True])
async def test_a_failure_below_skips_the_reduce_above(recursive: bool) -> None:
    engine = TreeTraversal(max_running=2)
    action = _Count(engine._emit_event)
    handler = _Handler(_FILES, failing="r/a")  # Alone, or whole: it fails either way.
    task = VisitDirectory(engine, _dir("r"), action, handler, recursive=recursive)
    with pytest.raises(BaseExceptionGroup) as raised:
        [_ async for _ in engine.events([task])]
    assert [type(e) for e in raised.value.exceptions] == [OSError]
    reduced = {n for c, n in action.calls if c == "reduce"}
    assert reduced == {"r/c"}  # Its sibling runs on: r/a and r are not reduced.


async def test_a_directory_listed_with_a_slash_is_traversed_once() -> None:
    _, _, events = await _run(False, handler=_Handler(_FILES, slashes=True))
    # The prefix and the marker are one directory, and the own marker nothing.
    assert sorted(events) == [
        ("r", 5),
        ("r/a", 3),
        ("r/a/b", 1),
        ("r/a/b-x", 1),
        ("r/c", 1),
    ]
