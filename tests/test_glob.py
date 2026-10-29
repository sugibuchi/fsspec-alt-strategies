"""Tests of `tree_glob`, checked against fsspec's own ``_glob``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.control import Stop
from fsspec_alt_strategies.glob import GlobState, _Pattern, tree_glob

if TYPE_CHECKING:
    from fsspec_alt_strategies.testing import SimulatedFileSystem

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
    result = await tree_glob(any_fs, path, **kwargs)
    assert result.complete
    expected = await AsyncFileSystem._glob(any_fs, path, **kwargs)
    assert list(result.entries) == expected


async def test_prunes_folders_the_pattern_cannot_match_in(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    result = await tree_glob(hierarchical_fs, "data/*/2024/*/part-*")
    assert len(result.entries) == 6
    # data, the 3 tables, their 2024s: the 2023s are pruned.
    assert hierarchical_fs.stats.calls == {"list": 1 + 3 + 3 + 6}
    assert result.folders_pruned == 3

    hierarchical_fs.reset_stats()
    await AsyncFileSystem._glob(hierarchical_fs, "data/*/2024/*/part-*")
    assert hierarchical_fs.stats.calls["list"] == 1 + 3 + 6 + 12


async def test_double_star_prunes_after_it(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    result = await tree_glob(hierarchical_fs, "data/sales/**")
    # Only data/sales and below: the literal root is data/sales.
    assert result.folders_pruned == 0
    assert hierarchical_fs.stats.calls["list"] == 1 + 2 + 4


async def test_stop_returns_what_was_matched(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    states: list[GlobState] = []

    def first_only(state: GlobState) -> Stop:
        states.append(state)
        return Stop()

    result = await tree_glob(any_fs, "data/*", callback=first_only)
    assert not result.complete
    assert states[0].folders_listed == 1


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
def test_pattern_alive(segments: list[str], relative: list[str], alive: bool) -> None:
    assert _Pattern(segments).alive(relative) is alive
