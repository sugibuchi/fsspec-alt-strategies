"""Tests of `walk`, the engine under the tree strategies."""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies.testing import SimulatedFileSystem
from fsspec_alt_strategies.walk import Listing, walk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet")
    fs.add_file("data/README.md")


async def _all(listings: AsyncIterator[Listing]) -> dict[str, Listing]:
    return {listing.folder: listing async for listing in listings}


async def test_walks_every_folder_once(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    listings = await _all(walk(any_fs, "data"))
    assert sorted(listings) == [
        "data",
        "data/2023",
        "data/2023/01",
        "data/2023/02",
        "data/2024",
        "data/2024/01",
        "data/2024/02",
    ]
    assert any_fs.stats.calls == {"list": 7}
    assert [listing.depth for listing in listings.values()].count(3) == 4
    names = {e["name"] for e in listings["data"].entries}
    assert names == {"data/2023", "data/2024", "data/README.md"}


async def test_first_listing_is_the_root(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    async with aclosing(walk(any_fs, "simulated:///data/")) as listings:
        first = await anext(listings)
    assert (first.folder, first.depth) == ("data", 1)


async def test_maxdepth(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    listings = await _all(walk(any_fs, "data", maxdepth=2))
    assert sorted(listings) == ["data", "data/2023", "data/2024"]
    with pytest.raises(ValueError, match="at least 1"):
        await _all(walk(any_fs, "data", maxdepth=0))


async def test_descend_prunes(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)

    def only_2024(entry: dict[str, Any]) -> bool:
        return entry["name"].startswith("data/2024")

    listings = await _all(walk(any_fs, "data", descend=only_2024))
    assert sorted(listings) == ["data", "data/2024", "data/2024/01", "data/2024/02"]
    assert any_fs.stats.calls == {"list": 4}


async def test_root_errors_are_raised(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await _all(walk(any_fs, "missing"))
    with pytest.raises(NotADirectoryError):
        await _all(walk(any_fs, "data/README.md"))


class _Forbidding(SimulatedFileSystem):
    """Refuses to list one folder, as a storage without permission would."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


async def test_failed_listing_is_yielded_and_the_walk_goes_on() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    listings = await _all(walk(fs, "data"))
    assert isinstance(listings["data/2023"].error, PermissionError)
    assert listings["data/2023"].entries == []
    assert "data/2023/01" not in listings
    assert "data/2024/02" in listings


async def test_lists_folders_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(20):
        fs.add_file(f"root/{i:02}/file")
    await _all(walk(fs, "root", batch_size=4))
    assert fs.stats.calls == {"list": 21}
    assert fs.stats.max_concurrency == 4


async def test_closing_cancels_listings_in_flight() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    async with aclosing(walk(fs, "root", batch_size=10)) as listings:
        async for _ in listings:
            break  # The root's listing: its ten subfolders are now in flight.
    # None is left running, and those cancelled before they started were never sent.
    assert fs._in_flight == 0
    assert fs.stats.calls["list"] < 11
