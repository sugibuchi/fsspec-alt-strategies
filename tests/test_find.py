"""Tests of `find` and `ifind`, checked against fsspec's own ``_find``."""

from __future__ import annotations

from contextlib import aclosing
from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.control import OperationStopped, Stop
from fsspec_alt_strategies.find import FindState, find, ifind
from fsspec_alt_strategies.testing import SimulatedFileSystem


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * 10)
    fs.add_file("data/README.md")


def _names(entries: list[dict[str, Any]]) -> list[str]:
    return [e["name"].rstrip("/") for e in entries]


async def _fsspec_find(fs: SimulatedFileSystem, path: str, **kwargs: Any) -> Any:
    # fsspec's own walk, as the reference answer.
    return await AsyncFileSystem._find(fs, path, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"withdirs": True}, {"maxdepth": 1}, {"maxdepth": 2, "withdirs": True}],
)
@pytest.mark.parametrize("path", ["data", "data/2024", "data/README.md", "missing"])
async def test_answers_as_fsspec(
    any_fs: SimulatedFileSystem, path: str, kwargs: dict[str, Any]
) -> None:
    _tree(any_fs)
    entries, errors = await find(any_fs, path, **kwargs)
    assert errors == []
    expected = await _fsspec_find(any_fs, path, detail=True, **kwargs)
    assert _names(entries) == list(expected)
    assert entries == list(expected.values())

    found = [e async for e in ifind(any_fs, path, **kwargs)]
    assert sorted(_names(found)) == list(expected)


async def test_entries_are_the_listings(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    entries, _ = await find(any_fs, "data")
    assert entries[0]["size"] == 10
    # One listing per folder, and no _info.
    assert any_fs.stats.calls == {"list": 7}


async def test_the_whole_filesystem(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    entries, _ = await find(any_fs, "", withdirs=True)
    assert _names(entries) == await _fsspec_find(any_fs, "", withdirs=True)


async def test_lists_folders_concurrently() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(20):
        fs.add_file(f"root/{i:02}/file")
    entries, _ = await find(fs, "root", batch_size=5)
    assert len(entries) == 20
    assert fs.stats.max_concurrency == 5


class _Forbidding(SimulatedFileSystem):
    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


async def test_a_failed_folder_is_reported() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    entries, errors = await find(fs, "data")
    assert [type(e) for e in errors] == [PermissionError]
    assert _names(entries) == [
        "data/2024/01/part-0.parquet",
        "data/2024/02/part-0.parquet",
        "data/README.md",
    ]
    found = [e async for e in ifind(fs, "data")]
    assert sum(isinstance(e, PermissionError) for e in found) == 1


async def test_stop_returns_what_was_found(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    states: list[FindState] = []

    def first_only(state: FindState) -> Stop:
        states.append(state)
        return Stop()

    entries, _ = await find(any_fs, "data", callback=first_only)
    assert _names(entries) == ["data/README.md"]
    assert (states[0].folders_listed, states[0].found) == (1, 1)
    stop = Stop(raise_error=True)
    with pytest.raises(OperationStopped) as info:
        await find(any_fs, "data", callback=lambda _: stop)
    entries, _ = info.value.result
    assert _names(entries) == ["data/README.md"]
    assert info.value.stop is stop


async def test_breaking_out_stops_the_search() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    async with aclosing(ifind(fs, "root", batch_size=2)) as found:
        async for _ in found:
            break  # The first file: other listings are now in flight.
    assert fs._in_flight == 0
    assert fs.stats.calls["list"] < 11
