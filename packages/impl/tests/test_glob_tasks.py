"""Tests of `common.glob`'s tasks on a dummy tree: the same actions, whichever traverses."""

from __future__ import annotations

import asyncio
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
    segments = _PathHandler.segments

    def __init__(
        self, files: list[str], *, failing: str | None = None, slashes: bool = False
    ) -> None:
        self.files = files
        self.failing = failing
        self.slashes = slashes
        self.operation_limit = asyncio.Semaphore(2)

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
        self.matched: set[str] = set()

    async def map(self, entry: dict[str, Any], match: bool) -> Any:
        self.calls.append(("map", entry["name"]))
        if match:
            self.matched.add(entry["name"])
        return 1 if entry["type"] == "file" else 0

    async def reduce(
        self, entry: dict[str, Any], match: bool, value: Any, values: list[Any]
    ) -> Any:
        assert match == (entry["name"] in self.matched)
        self.calls.append(("reduce", entry["name"]))
        if entry["type"] == "file":
            assert values == []
        values = [v for v in values if v is not None]  # A listing failed: None.
        if entry["type"] != "file":
            await self._emit_event((entry["name"], sum(values)))
        return value + sum(values)


async def _run(
    use_one_listing: bool, root: str = "r", **kwargs: Any
) -> tuple[Any, _Count, list[Any]]:
    engine = TreeTraversal(max_running=2)
    action = _Count(engine._emit_event)
    handler = kwargs.pop("handler", _Handler(_FILES))
    task = VisitDirectory(
        engine, _dir(root), action, handler, use_one_listing=use_one_listing, **kwargs
    )
    events = [e async for e in engine.events([task])]
    return task.result(), action, events


@pytest.mark.parametrize("use_one_listing", [False, True])
async def test_each_path_mapped_before_and_reduced_after_its_subtree(
    use_one_listing: bool,
) -> None:
    value, action, events = await _run(use_one_listing)
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
    task = TraverseSubtree(engine, _dir("r"), action, _Handler(_FILES))
    [_ async for _ in engine.events([task])]
    assert task.result() == 5
    assert sorted(action.calls) == sorted(by_directory.calls)


async def test_a_subtree_streams_in_name_order_with_its_markers() -> None:
    # As an object store lists: by name, so "r/a/b-x/" before "r/a/b/", with
    # each directory's marker, the root's too, before what is below it.
    names = ["r/", "r/1", "r/a/", "r/a/2", "r/a/b-x/", "r/a/b-x/4", "r/a/b/"]
    names += ["r/a/b/3", "r/c/5"]

    class _InNameOrder(_Handler):
        async def ils(
            self, entry: dict[str, Any], recursive: bool = False
        ) -> AsyncIterator[Any]:
            assert recursive
            for name in names:
                kind = "directory" if name.endswith("/") else "file"
                yield {"name": name, "type": kind, "size": int(kind == "file")}

    _, by_directory, _ = await _run(False)
    engine = TreeTraversal(max_running=1)
    action = _Count(engine._emit_event)
    task = TraverseSubtree(engine, _dir("r"), action, _InNameOrder(_FILES))
    events = [e async for e in engine.events([task])]
    assert task.result() == 5
    # The entries are as listed: a directory listed keeps its slash.
    assert sorted((n.rstrip("/"), v) for n, v in events) == [
        ("r", 5),
        ("r/a", 3),
        ("r/a/b", 1),
        ("r/a/b-x", 1),
        ("r/c", 1),
    ]
    # Each directory once, whether listed with a slash or made: r/c is made.
    calls = sorted((c, n.rstrip("/")) for c, n in action.calls)
    assert calls == sorted(by_directory.calls)


@pytest.mark.parametrize("use_one_listing", [False, True])
async def test_patterns_prune_the_paths_none_matches(
    use_one_listing: bool,
) -> None:
    patterns = [_Search("r/a/**", ends_with_sep=False, withdirs=False, maxdepth=None)]
    value, action, _ = await _run(use_one_listing, patterns=patterns)
    # r/c is not descended into, and r/1 not matched: neither is visited.
    assert value == 3
    mapped = sorted(n for c, n in action.calls if c == "map")
    assert mapped == [
        "r",
        "r/a",
        "r/a/2",
        "r/a/b",
        "r/a/b-x",
        "r/a/b-x/4",
        "r/a/b/3",
    ]
    reduced = sorted(n for c, n in action.calls if c == "reduce")
    assert reduced == mapped
    # Every file visited matches: the directories are on the way.
    assert sorted(action.matched) == ["r/a/2", "r/a/b-x/4", "r/a/b/3"]


@pytest.mark.parametrize("use_one_listing", [False, True])
async def test_a_directory_matched_but_not_descended_into_is_visited_not_listed(
    use_one_listing: bool,
) -> None:
    patterns = [_Search("r/*", ends_with_sep=False, withdirs=True, maxdepth=None)]
    _, action, _ = await _run(use_one_listing, patterns=patterns)
    mapped = sorted(n for c, n in action.calls if c == "map")
    assert mapped == ["r", "r/1", "r/a", "r/c"]
    assert sorted(action.matched) == ["r/1", "r/a", "r/c"]


@pytest.mark.parametrize("use_one_listing", [False, True])
async def test_a_failure_below_skips_the_reduce_above(use_one_listing: bool) -> None:
    engine = TreeTraversal(max_running=2)
    action = _Count(engine._emit_event)
    handler = _Handler(_FILES, failing="r/a")  # Alone, or whole: it fails either way.
    task = VisitDirectory(
        engine,
        _dir("r"),
        action,
        handler,
        use_one_listing=use_one_listing,
        on_error="raise",
    )
    with pytest.raises(BaseExceptionGroup) as raised:
        [_ async for _ in engine.events([task])]
    assert [type(e) for e in raised.value.exceptions] == [OSError]
    reduced = {n for c, n in action.calls if c == "reduce"}
    # Its siblings run on: r/a and r are not reduced.
    assert reduced == {"r/1", "r/c", "r/c/5"}


@pytest.mark.parametrize("use_one_listing", [False, True])
async def test_a_listing_error_is_emitted_and_the_others_go_on(
    use_one_listing: bool,
) -> None:
    handler = _Handler(_FILES, failing="r/a")
    value, action, events = await _run(use_one_listing, handler=handler)
    errors = [(n["name"], e) for n, e in events if isinstance(e, Exception)]
    assert [(n, str(e)) for n, e in errors] == [("r/a", "r/a")]  # With its entry.
    # r/a is left out, and r reduced with the others' values.
    assert value == 2
    assert ("reduce", "r/a") not in action.calls


async def test_a_listing_error_omitted_or_called_back() -> None:
    called: list[Exception] = []
    for on_error in ("omit", called.append):
        handler = _Handler(_FILES, failing="r/a")
        _, _, events = await _run(False, handler=handler, on_error=on_error)
        assert not [e for _, e in events if isinstance(e, Exception)]
    assert [str(e) for e in called] == ["r/a"]


async def test_the_roots_listing_error_is_raised_whatever_on_error() -> None:
    with pytest.raises(OSError, match="r"):
        await _run(False, handler=_Handler(_FILES, failing="r"), on_error="omit")


@pytest.mark.parametrize("use_one_listing", [False, True])
async def test_maxdepth_bounds_the_directories_listed(use_one_listing: bool) -> None:
    _, action, events = await _run(use_one_listing, maxdepth=2)
    # r and its subdirectories are listed: r/a/b and r/a/b-x are visited, not listed.
    assert sorted(n for n, _ in events) == [
        "r",
        "r/a",
        "r/a/b",
        "r/a/b-x",
        "r/c",
    ]
    mapped = {n for c, n in action.calls if c == "map"}
    assert "r/a/b/3" not in mapped
    assert "r/a/2" in mapped


async def test_a_listing_holds_a_permit_per_step_only() -> None:
    # One permit: a listing holding it whole would starve the directories below.
    handler = _Handler(_FILES)
    handler.operation_limit = asyncio.Semaphore(1)
    value, _, _ = await _run(False, handler=handler)
    assert value == 5


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


@pytest.mark.parametrize("use_one_listing", [False, True])
async def test_the_items_are_mapped_in_parallel_after_their_directory(
    use_one_listing: bool,
) -> None:
    active = peak = 0
    mapped: set[str] = set()
    early: list[str] = []

    class _Slow(PathAction):
        async def map(self, entry: dict[str, Any], match: bool) -> Any:
            nonlocal active, peak
            parent = entry["name"].rpartition("/")[0]
            if parent and parent not in mapped:
                early.append(entry["name"])
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            mapped.add(entry["name"])

    files = [f"r/{i}" for i in range(6)]
    engine = TreeTraversal(max_running=4)
    task = VisitDirectory(
        engine,
        _dir("r"),
        _Slow(engine._emit_event),
        _Handler(files),
        use_one_listing=use_one_listing,
    )
    [_ async for _ in engine.events([task])]
    assert peak == 4
    assert early == []
