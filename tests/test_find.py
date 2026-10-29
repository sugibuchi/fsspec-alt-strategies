"""Tests of `tree_find`, checked against fsspec's own ``_find``."""

from __future__ import annotations

from typing import Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.control import OperationStopped, Stop
from fsspec_alt_strategies.find import FindState, tree_find
from fsspec_alt_strategies.testing import SimulatedFileSystem


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet", b"x" * 10)
    fs.add_file("data/README.md")


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
    result = await tree_find(any_fs, path, **kwargs)
    assert result.complete
    assert list(result.entries) == await _fsspec_find(any_fs, path, **kwargs)


async def test_entries_are_the_listings(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    result = await tree_find(any_fs, "data")
    assert result.entries["data/2024/01/part-0.parquet"]["size"] == 10
    # One listing per folder, and no _info.
    assert any_fs.stats.calls == {"list": 7}


async def test_the_whole_filesystem(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    result = await tree_find(any_fs, "", withdirs=True)
    assert list(result.entries) == await _fsspec_find(any_fs, "", withdirs=True)


async def test_lists_folders_concurrently() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(20):
        fs.add_file(f"root/{i:02}/file")
    result = await tree_find(fs, "root", batch_size=5)
    assert len(result.entries) == 20
    assert fs.stats.max_concurrency == 5


class _Forbidding(SimulatedFileSystem):
    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


async def test_a_failed_folder_is_reported() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    result = await tree_find(fs, "data")
    assert list(result.errors) == ["data/2023"]
    assert list(result.entries) == [
        "data/2024/01/part-0.parquet",
        "data/2024/02/part-0.parquet",
        "data/README.md",
    ]


async def test_stop_returns_what_was_found(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    states: list[FindState] = []

    def first_only(state: FindState) -> Stop:
        states.append(state)
        return Stop()

    result = await tree_find(any_fs, "data", callback=first_only)
    assert not result.complete
    assert list(result.entries) == ["data/README.md"]
    assert (states[0].folders_listed, states[0].found) == (1, 1)
    with pytest.raises(OperationStopped) as info:
        await tree_find(any_fs, "data", callback=lambda _: Stop(raise_error=True))
    assert list(info.value.result.entries) == ["data/README.md"]
