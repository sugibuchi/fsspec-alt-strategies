"""Tests of `tree_glob` and `itree_glob`, checked against fsspec's own ``_glob``."""

from __future__ import annotations

from contextlib import aclosing
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.glob import _Prefix, itree_glob, tree_glob
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
    entries, errors = await tree_glob(any_fs, path, **kwargs)
    assert errors == []
    expected = await AsyncFileSystem._glob(any_fs, path, detail=True, **kwargs)
    assert [e["name"].rstrip("/") for e in entries] == list(expected)
    assert entries == list(expected.values())

    matches = [m async for m in itree_glob(any_fs, path, **kwargs)]
    assert sorted(matches, key=lambda e: e["name"]) == entries


async def test_prunes_folders_the_pattern_cannot_match_in(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    entries, _ = await tree_glob(hierarchical_fs, "data/*/2024/*/part-*")
    assert len(entries) == 6
    # data, the 3 tables, their 2024s: the 2023s are pruned.
    assert hierarchical_fs.stats.calls == {"list": 1 + 3 + 3 + 6}

    hierarchical_fs.reset_stats()
    await AsyncFileSystem._glob(hierarchical_fs, "data/*/2024/*/part-*")
    assert hierarchical_fs.stats.calls["list"] == 1 + 3 + 6 + 12


async def test_double_star_prunes_after_it(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    await tree_glob(hierarchical_fs, "data/sales/**")
    # Only data/sales and below: the literal root is data/sales.
    assert hierarchical_fs.stats.calls["list"] == 1 + 2 + 4


async def test_iterator_makes_the_same_requests(
    hierarchical_fs: SimulatedFileSystem,
) -> None:
    _tree(hierarchical_fs)
    await tree_glob(hierarchical_fs, "data/*/2024/*/part-*")
    calls = dict(hierarchical_fs.stats.calls)
    hierarchical_fs.reset_stats()
    async for _ in itree_glob(hierarchical_fs, "data/*/2024/*/part-*"):
        pass
    assert hierarchical_fs.stats.calls == calls


async def test_scans_subtrees_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(5):
        for j in range(5):
            fs.add_file(f"root/{i}/{j}/file")
    entries, _ = await tree_glob(fs, "root/*/*/file", batch_size=4)
    assert len(entries) == 25
    assert fs.stats.calls == {"list": 1 + 5 + 25}
    assert fs.stats.max_concurrency == 4


class _Forbidding(SimulatedFileSystem):
    """Refuses to list one folder, as a storage without permission would."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/sales":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


async def test_failed_listing_adds_its_error_and_the_search_goes_on() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    entries, errors = await tree_glob(fs, "data/*/2024/*/*")
    assert [type(e) for e in errors] == [PermissionError]
    assert len(entries) == 4
    assert not any(e["name"].startswith("data/sales/") for e in entries)

    matches = [m async for m in itree_glob(fs, "data/*/2024/*/*")]
    assert sum(isinstance(m, PermissionError) for m in matches) == 1


async def test_breaking_out_stops_the_search() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    async with aclosing(itree_glob(fs, "root/*/file", batch_size=2)) as matches:
        async for _ in matches:
            break  # The first match: other listings are now in flight.
    assert fs._in_flight == 0
    assert fs.stats.calls["list"] < 11


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
