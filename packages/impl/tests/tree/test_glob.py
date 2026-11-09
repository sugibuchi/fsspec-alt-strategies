"""Tests of `_tree_glob`, checked against fsspec's own ``_glob``."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.common.patterns import _Prefix
from fsspec_alt_strategies_impl.tree.glob import GlobState, _tree_glob
from fsspec_alt_strategies_impl_testing import (
    SimulatedFileSystem,
    TreeView,
)

_PRIMITIVES = {"ils", "info", "rm_file", "rm_folder", "rm_tree", "cp_file", "mv_file"}
"""The keyword arguments that replace a primitive of the filesystem."""


def _glob(fs: SimulatedFileSystem, path: Any, **kwargs: Any) -> Any:
    """`_tree_glob` on a filesystem's own primitives, which ``kwargs`` may replace."""
    view = TreeView(fs, **{k: kwargs.pop(k) for k in _PRIMITIVES & kwargs.keys()})
    return _tree_glob(view, path, **kwargs)


async def _sorted_glob(
    fs: SimulatedFileSystem, path: str, **kwargs: Any
) -> tuple[list[dict[str, Any]], list[Exception]]:
    entries, errors = await _collect_with_errors(_glob(fs, path, **kwargs))
    return sorted(entries, key=lambda e: e["name"]), errors


_PATTERNS = [
    "data",
    "data/README.md",
    "missing/*",
    "data/README.md/*",
    "*",
    "da*",
    "data/*",
    "data/*/",
    "data/**",
    "data/**/",
    "data/**/*.parquet",
    "data/*/2024/*",
    "data/*/202[34]/0?/part-*",
    "data/sales/**/02/*",
    "data/**/2024/**",
    "data/**/**/README.md",
    "**/*.md",
    "data/sa*/2023",
]


def _tree(fs: SimulatedFileSystem) -> None:
    for table in ("sales", "stock", "users"):
        for year in ("2023", "2024"):
            for month in ("01", "02"):
                fs.add_file(f"data/{table}/{year}/{month}/part-0.parquet", b"x")
        fs.add_file(f"data/{table}/README.md")
    fs.add_file("data/README.md")
    fs.add_file("other/data.csv")


@pytest.mark.parametrize("kwargs", [{}, {"maxdepth": 2}, {"withdirs": False}])
@pytest.mark.parametrize("path", _PATTERNS)
async def test_answers_as_fsspec(
    any_fs: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    _tree(any_fs)
    entries, errors = await _sorted_glob(any_fs, path, **kwargs)
    assert errors == []
    expected = await AsyncFileSystem._glob(any_fs, path, detail=True, **kwargs)
    assert [e["name"].rstrip("/") for e in entries] == list(expected)
    assert entries == list(expected.values())


async def test_prunes_folders_the_pattern_cannot_match_in(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    op = _glob(hierarchical_fs, "data/*/2024/*/part-*")
    entries, _ = await _collect_with_errors(op)
    assert len(entries) == 6
    assert op.state == GlobState(
        current=op.state.current, folders_listed=1 + 3 + 3 + 6, matched=6
    )
    # data, the 3 tables, their 2024s: the 2023s are pruned.
    assert hierarchical_fs.stats.calls == {"list": 1 + 3 + 3 + 6}

    hierarchical_fs.reset_stats()
    await AsyncFileSystem._glob(hierarchical_fs, "data/*/2024/*/part-*")
    assert hierarchical_fs.stats.calls["list"] == 1 + 3 + 6 + 12


async def test_double_star_prunes_after_it(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    await _collect_with_errors(_glob(hierarchical_fs, "data/sales/**"))
    # Only data/sales and below: the literal root is data/sales.
    assert hierarchical_fs.stats.calls["list"] == 1 + 2 + 4


_PATTERN_LISTS = [
    ["data/*/2024/*", "data/*/2023/*"],
    ["data/*/2024/*/part-*", "data/sales/**"],  # One root inside the other.
    ["data/sales/*", "data/**/README.md", "other/*"],
    ["data/README.md", "data/*.md", "data/README.md"],
    ["*", "data/*/"],
    ["missing/*", "data/sa*/2023", "data/README.md/*"],
    ["data/sales/2024/**", "data/*/2024/01/*", "**/*.md"],
    ["data/stock", "data/stock/**"],
    [],
]


@pytest.mark.parametrize("kwargs", [{}, {"maxdepth": 2}, {"withdirs": False}])
@pytest.mark.parametrize("paths", _PATTERN_LISTS)
async def test_several_patterns_answer_as_fsspecs_globs_together(
    any_fs: SimulatedFileSystem, paths: list[str], kwargs: dict[str, Any]
) -> None:
    _tree(any_fs)
    entries, errors = await _collect_with_errors(_glob(any_fs, paths, **kwargs))
    assert errors == []
    expected: dict[str, Any] = {}
    for path in paths:
        expected |= await AsyncFileSystem._glob(any_fs, path, detail=True, **kwargs)
    names = [e["name"].rstrip("/") for e in entries]
    assert sorted(names) == sorted(expected)  # Each once.
    assert sorted(entries, key=lambda e: e["name"]) == [
        expected[n] for n in sorted(expected)
    ]


async def test_several_patterns_list_shared_folders_once(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    patterns = ["data/*/2024/*/*.parquet", "data/*/2024/*/*.csv"]
    entries, _ = await _collect_with_errors(_glob(hierarchical_fs, patterns))
    assert len(entries) == 6
    # data, the 3 tables, their 2024s: as for one pattern.
    assert hierarchical_fs.stats.calls == {"list": 1 + 3 + 3 + 6}

    hierarchical_fs.reset_stats()
    for pattern in patterns:
        await _collect_with_errors(_glob(hierarchical_fs, pattern))
    assert hierarchical_fs.stats.calls == {"list": 2 * (1 + 3 + 3 + 6)}


async def test_lists_folders_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(5):
        for j in range(5):
            fs.add_file(f"root/{i}/{j}/file")
    entries, _ = await _collect_with_errors(
        _glob(fs, "root/*/*/file", concurrency_limit=asyncio.Semaphore(4))
    )
    assert len(entries) == 25
    assert fs.stats.calls == {"list": 1 + 5 + 25}
    assert fs.stats.max_concurrency == 4


async def test_plain_paths_are_asked_for_concurrently() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    paths = [f"root/{i:02}.csv" for i in range(20)]
    for path in paths:
        fs.add_file(path)
    op = _glob(fs, [*paths, "root/missing.csv"], concurrency_limit=asyncio.Semaphore(4))
    entries, _ = await _collect_with_errors(op)
    assert sorted(e["name"] for e in entries) == paths
    assert fs.stats.calls == {"info": 21}
    assert fs.stats.max_concurrency == 4  # Not one after another.


async def test_cancel_asks_for_no_more_plain_paths() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    paths = [f"root/{i:02}.csv" for i in range(20)]
    for path in paths:
        fs.add_file(path)
    op = _glob(fs, paths, concurrency_limit=asyncio.Semaphore(4))
    async for _ in op:
        op.cancel()
    # The first 4, and those started before the first answer was yielded, at
    # most 4 more: none after the cancel.
    assert fs.stats.calls["info"] <= 8
    assert op.cancelled()


class _Forbidding(SimulatedFileSystem):
    """Refuses to list one folder, as a storage without permission would."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/sales":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


async def test_failed_listing_adds_its_error_and_the_search_goes_on() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    entries, errors = await _collect_with_errors(_glob(fs, "data/*/2024/*/*"))
    assert [type(e) for e in errors] == [PermissionError]
    assert len(entries) == 4
    assert not any(e["name"].startswith("data/sales/") for e in entries)


def test_errors_in_the_pattern_are_raised_at_once(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _glob(any_fs, "data/**", maxdepth=0)
    with pytest.raises(ValueError):
        _glob(any_fs, "data/a**/b")


async def test_closing_stops_the_search() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    async with _glob(
        fs, "root/*/file", concurrency_limit=asyncio.Semaphore(2)
    ) as matches:
        async for _ in matches:
            break  # The first match: other listings are now in flight.
    assert fs._in_flight == 0
    assert fs.stats.calls["list"] < 11
    assert matches.cancelled()


@pytest.mark.parametrize(
    ("segments", "relative", "alive"),
    [
        (["*", "2024", "*"], ["sales"], True),
        (["*", "2024", "*"], ["sales", "2023"], False),
        (["*", "2024", "*"], ["sales", "2024", "01"], False),
        (["**", "2024"], ["a", "b", "c"], True),
        (["a", "**", "**", "b"], ["a"], True),
        (["a", "**"], ["b"], False),
    ],
)
def test_prefix_alive(segments: list[str], relative: list[str], alive: bool) -> None:
    assert _Prefix(segments).alive(relative) is alive


async def test_cancel_starts_no_further_pattern_walk() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    fs.add_file("a/1.txt")
    fs.add_file("b/1.txt")
    op = _glob(fs, ["a/*.txt", "b/*.txt"])
    found = []
    async for entry in op:
        found.append(entry["name"])
        op.cancel()
    # The walk of a/ was under way; that of b/ is never started.
    assert found == ["a/1.txt"]
    assert fs.stats.calls == {"list": 1}
    assert op.cancelled()
