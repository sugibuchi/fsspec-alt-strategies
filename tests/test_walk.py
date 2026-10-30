"""Tests of `iwalk`, and of ``_listings``, the engine under the tree strategies."""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies.testing import SimulatedFileSystem
from fsspec_alt_strategies.walk import _listings, iwalk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet")
    fs.add_file("data/README.md")


async def _all(items: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in items]


async def test_yields_every_entry_once(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    entries = await _all(iwalk(any_fs, "data"))
    assert sorted(e["name"].rstrip("/") for e in entries) == [
        "data/2023",
        "data/2023/01",
        "data/2023/01/part-0.parquet",
        "data/2023/02",
        "data/2023/02/part-0.parquet",
        "data/2024",
        "data/2024/01",
        "data/2024/01/part-0.parquet",
        "data/2024/02",
        "data/2024/02/part-0.parquet",
        "data/README.md",
    ]
    assert any_fs.stats.calls == {"list": 7}


def _order(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {e["name"].rstrip("/"): i for i, e in enumerate(entries)}


@pytest.mark.parametrize("latency", [0.0, 0.002])
async def test_topdown_orders(latency: float) -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=latency, asynchronous=True)
    _tree(fs)
    down = _order(await _all(iwalk(fs, "data", batch_size=3)))
    up = _order(await _all(iwalk(fs, "data", topdown=False, batch_size=3)))
    for name in down:
        parent = name.rsplit("/", 1)[0]
        if parent != "data":
            # Its folder's entry is in the listing of the folder above it.
            assert down[parent] < down[name]
            assert up[parent] > up[name]
    assert sorted(up) == sorted(down)


async def test_maxdepth(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    entries = await _all(iwalk(any_fs, "data", maxdepth=2))
    assert len(entries) == 3 + 4
    assert any_fs.stats.calls == {"list": 3}
    with pytest.raises(ValueError, match="at least 1"):
        await _all(iwalk(any_fs, "data", maxdepth=0))


async def test_descend_prunes(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)

    def only_2024(entry: dict[str, Any]) -> bool:
        return entry["name"].startswith("data/2024")

    await _all(iwalk(any_fs, "data", descend=only_2024))
    assert any_fs.stats.calls == {"list": 4}


async def test_root_errors_are_raised(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await _all(iwalk(any_fs, "missing"))
    with pytest.raises(NotADirectoryError):
        await _all(iwalk(any_fs, "data/README.md"))


class _Forbidding(SimulatedFileSystem):
    """Refuses to list one folder, as a storage without permission would."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


@pytest.mark.parametrize("topdown", [True, False])
async def test_failed_listing_is_yielded_and_the_walk_goes_on(topdown: bool) -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    items = await _all(iwalk(fs, "data", topdown=topdown))
    errors = [i for i in items if isinstance(i, Exception)]
    assert [type(e) for e in errors] == [PermissionError]
    names = {i["name"] for i in items if not isinstance(i, Exception)}
    assert "data/2023" in names
    assert not any(n.startswith("data/2023/") for n in names)
    assert "data/2024/02/part-0.parquet" in names


class _Recording(_Forbidding):
    """Records each folder whose listing reached the storage."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        self.requested.append(self._strip_protocol(path))
        return await super()._ls(path, detail=detail, **kwargs)


@pytest.mark.parametrize("topdown", [True, False])
async def test_raise_on_error_stops_at_once(topdown: bool) -> None:
    fs = _Recording(hierarchical=True, asynchronous=True)
    fs.requested = []
    _tree(fs)
    with pytest.raises(PermissionError):
        await _all(
            iwalk(fs, "data", topdown=topdown, raise_on_error=True, batch_size=1)
        )
    # data/2023 fails, and data/2024, queued behind it, is never listed.
    assert fs.requested == ["data", "data/2023"]
    assert fs._in_flight == 0


async def test_lists_folders_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(20):
        fs.add_file(f"root/{i:02}/file")
    await _all(iwalk(fs, "root", batch_size=4))
    assert fs.stats.calls == {"list": 21}
    assert fs.stats.max_concurrency == 4


async def test_closing_cancels_listings_in_flight() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    async with aclosing(iwalk(fs, "root", batch_size=10)) as entries:
        async for _ in entries:
            break  # The root's first entry: its ten subfolders are now in flight.
    # None is left running, and those cancelled before they started were never sent.
    assert fs._in_flight == 0
    assert fs.stats.calls["list"] < 11


async def test_listings_carry_folder_and_depth(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    async with aclosing(_listings(any_fs, "simulated:///data/")) as listings:
        first = await anext(listings)
    assert (first.folder, first.depth) == ("data", 1)
    listings = {lst.folder: lst async for lst in _listings(any_fs, "data")}
    assert [lst.depth for lst in listings.values()].count(3) == 4


async def test_bottom_up_listings_come_after_their_subfolders() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.002, asynchronous=True)
    _tree(fs)
    folders = [
        lst.folder async for lst in _listings(fs, "data", topdown=False, batch_size=2)
    ]
    assert folders[-1] == "data"
    for i, folder in enumerate(folders):
        assert all(not f.startswith(folder + "/") for f in folders[i + 1 :])
