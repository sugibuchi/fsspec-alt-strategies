"""Tests of `flat_glob`, checked against fsspec's own ``_glob``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.control import collect
from fsspec_alt_strategies.flat.glob import GlobState, flat_glob
from fsspec_alt_strategies_testing import SimulatedFileSystem

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _glob(fs: SimulatedFileSystem, path: Any, **kwargs: Any) -> Any:
    """`flat_glob` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {
        "concurrency_limit": asyncio.Semaphore(32),
        "ils": fs._ils,
        "info": fs._info,
    }
    return flat_glob(path, **{**operations, **kwargs})


def _tree(fs: SimulatedFileSystem) -> None:
    for table in ("sales", "stock", "users"):
        for year in ("2023", "2024"):
            for month in ("01", "02"):
                fs.add_file(f"data/{table}/{year}/{month}/part-0.parquet", b"x")
        fs.add_file(f"data/{table}/README.md")
    fs.add_file("data/README.md")
    fs.add_file("data/2024-01.csv")
    fs.add_file("data/2024-02.csv")
    fs.add_file("other/data.csv")


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
    "data/2024-*.csv",
]


@pytest.mark.parametrize("kwargs", [{}, {"maxdepth": 2}, {"withdirs": False}])
@pytest.mark.parametrize("path", _PATTERNS)
async def test_answers_as_fsspec(
    flat_fs: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    _tree(flat_fs)
    entries, errors = await collect(_glob(flat_fs, path, **kwargs))
    assert errors == []
    expected = await AsyncFileSystem._glob(flat_fs, path, detail=True, **kwargs)
    assert sorted(e["name"] for e in entries) == list(expected)
    assert sorted(entries, key=lambda e: e["name"]) == list(expected.values())


@pytest.mark.parametrize(
    "paths",
    [
        ["data/*/2024/*", "data/*/2023/*"],
        ["data/sales/*", "data/**/README.md", "other/*"],
        ["data/README.md", "data/*.md", "data/README.md"],
        ["data/stock", "data/stock/**"],
        [],
    ],
)
async def test_several_patterns_answer_as_fsspecs_globs_together(
    flat_fs: SimulatedFileSystem, paths: list[str]
) -> None:
    _tree(flat_fs)
    entries, _ = await collect(_glob(flat_fs, paths))
    expected: dict[str, Any] = {}
    for path in paths:
        expected |= await AsyncFileSystem._glob(flat_fs, path, detail=True)
    assert sorted(e["name"] for e in entries) == sorted(expected)  # Each once.


def _recorded(fs: SimulatedFileSystem, prefixes: list[str]) -> Any:
    """The filesystem's ``ils``, noting each prefix it lists."""

    def ils(prefix: str) -> AsyncIterator[dict[str, Any]]:
        prefixes.append(prefix)
        return fs._ils(prefix)

    return ils


async def test_lists_only_the_keys_starting_as_the_pattern() -> None:
    fs = SimulatedFileSystem(page_size=1, asynchronous=True)
    _tree(fs)
    prefixes: list[str] = []
    op = _glob(fs, "data/2024-*.csv", ils=_recorded(fs, prefixes))
    entries, _ = await collect(op)
    assert len(entries) == 2
    assert prefixes == ["data/2024-"]
    # A page per key, 2 keys: fsspec's lists all 6 entries of data, the three
    # tables and the README too.
    assert fs.stats.calls == {"list": 2}
    assert op.state == GlobState(current=op.state.current, matched=2)

    fs.reset_stats()
    await AsyncFileSystem._glob(fs, "data/2024-*.csv")
    assert fs.stats.calls["list"] == 6


async def test_patterns_share_a_listing() -> None:
    fs = SimulatedFileSystem(asynchronous=True)
    _tree(fs)
    prefixes: list[str] = []
    patterns = ["data/sales/2024/*", "data/sales/*", "data/s*/README.md", "other/*"]
    await collect(_glob(fs, patterns, ils=_recorded(fs, prefixes)))
    # data/sales/ is in data/s: one listing serves the three.
    assert prefixes == ["data/s", "other/"]


def test_errors_in_the_pattern_are_raised_at_once(
    flat_fs: SimulatedFileSystem,
) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _glob(flat_fs, "data/**", maxdepth=0)
    with pytest.raises(ValueError):
        _glob(flat_fs, "data/a**/b")
