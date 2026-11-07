"""Tests of `tree_walk`, and of ``_listings``, the engine under the tree strategies."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

import pytest

from fsspec_alt_strategies.tree.walk import WalkState, _listings, tree_walk
from fsspec_alt_strategies_testing import SimulatedFileSystem


def _walk(fs: SimulatedFileSystem, path: Any, **kwargs: Any) -> Any:
    """`tree_walk` on a filesystem's own operations, which ``kwargs`` may replace."""
    operations = {"concurrency_limit": asyncio.Semaphore(32), "ls": fs._ls}
    return tree_walk(path, **{**operations, **kwargs})


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
    entries = await _all(_walk(any_fs, "data"))
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
    down = _order(await _all(_walk(fs, "data", concurrency_limit=asyncio.Semaphore(3))))
    up = _order(
        await _all(
            _walk(fs, "data", topdown=False, concurrency_limit=asyncio.Semaphore(3))
        )
    )
    for name in down:
        parent = name.rsplit("/", 1)[0]
        if parent != "data":
            # Its folder's entry is in the listing of the folder above it.
            assert down[parent] < down[name]
            assert up[parent] > up[name]
    assert sorted(up) == sorted(down)


async def test_maxdepth(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    entries = await _all(_walk(any_fs, "data", maxdepth=2))
    assert len(entries) == 3 + 4
    assert any_fs.stats.calls == {"list": 3}
    with pytest.raises(ValueError, match="at least 1"):
        await _all(_walk(any_fs, "data", maxdepth=0))


async def test_descend_prunes(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)

    def only_2024(entry: dict[str, Any]) -> bool:
        return entry["name"].startswith("data/2024")

    await _all(_walk(any_fs, "data", descend=only_2024))
    assert any_fs.stats.calls == {"list": 4}


async def test_root_errors_are_raised(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await _all(_walk(any_fs, "missing"))
    with pytest.raises(NotADirectoryError):
        await _all(_walk(any_fs, "data/README.md"))


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
    items = await _all(_walk(fs, "data", topdown=topdown))
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
            _walk(
                fs,
                "data",
                topdown=topdown,
                raise_on_error=True,
                concurrency_limit=asyncio.Semaphore(1),
            )
        )
    # data/2023 fails, and data/2024, queued behind it, is never listed.
    assert fs.requested == ["data", "data/2023"]
    assert fs._in_flight == 0


async def test_lists_folders_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(20):
        fs.add_file(f"root/{i:02}/file")
    await _all(_walk(fs, "root", concurrency_limit=asyncio.Semaphore(4)))
    assert fs.stats.calls == {"list": 21}
    assert fs.stats.max_concurrency == 4


async def test_the_concurrency_limit_defaults_to_its_own() -> None:
    # Not fsspec's batch size, which is sized from the limit of open files.
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(100):
        fs.add_file(f"root/{i:03}/file")
    await _all(_walk(fs, "root"))
    assert fs.stats.max_concurrency == 32


async def _all_permits_back(limit: asyncio.Semaphore, permits: int) -> None:
    for _ in range(permits):
        await asyncio.wait_for(limit.acquire(), 0.1)


async def test_a_shared_limit_spans_operations() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(10):
        fs.add_file(f"a/{i}/file")
        fs.add_file(f"b/{i}/file")
    limit = asyncio.Semaphore(3)
    await asyncio.gather(
        _all(_walk(fs, "a", concurrency_limit=limit)),
        _all(_walk(fs, "b", concurrency_limit=limit)),
    )
    # Each alone would list 10 folders at once: together, still 3 at a time.
    assert fs.stats.max_concurrency == 3
    await _all_permits_back(limit, 3)


async def test_waits_for_a_permit_another_gives_back() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    _tree(fs)
    limit = asyncio.Semaphore(1)
    await limit.acquire()  # Held by another file, or operation.
    walk = asyncio.ensure_future(_all(_walk(fs, "data", concurrency_limit=limit)))
    await asyncio.sleep(0.01)
    assert fs.stats.calls == {}
    limit.release()
    assert len(await walk) == 11
    await _all_permits_back(limit, 1)


async def test_closing_gives_every_permit_back() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    limit = asyncio.Semaphore(4)
    async with _walk(fs, "root", concurrency_limit=limit) as entries:
        async for _ in entries:
            break  # Listings in flight, and a waiter for the next permit.
    await _all_permits_back(limit, 4)


async def test_closing_cancels_listings_in_flight() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    async with _walk(fs, "root", concurrency_limit=asyncio.Semaphore(10)) as entries:
        async for _ in entries:
            break  # The root's first entry: its ten subfolders are now in flight.
    # None is left running, and those cancelled before they started were never sent.
    assert fs._in_flight == 0
    assert fs.stats.calls["list"] < 11


async def test_cancel_yields_the_listings_in_flight() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/a")
        fs.add_file(f"root/{i}/b")
    op = _walk(fs, "root", concurrency_limit=asyncio.Semaphore(4))
    names = []
    async for entry in op:
        names.append(entry["name"])
        if len(names) == 10:
            # The root's ten entries: four subfolders are in flight.
            op.cancel()
    assert len(names) == 10 + 4 * 2
    assert fs.stats.calls["list"] == 1 + 4
    assert op.cancelled()
    assert op.state == WalkState(
        current=op.state.current, folders_listed=5, found=10 + 4 * 2
    )


async def test_failed_listing_notes_its_folder() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    errors = [i for i in await _all(_walk(fs, "data")) if isinstance(i, Exception)]
    assert errors[0].__notes__ == ["Listing 'data/2023' failed."]


async def test_listings_carry_folder_and_depth(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    # A trailing "/" is trimmed, as fsspec's _strip_protocol does.
    async with aclosing(
        _listings(any_fs._ls, "data/", concurrency_limit=asyncio.Semaphore(32))
    ) as listings:
        first = await anext(listings)
    assert (first.folder, first.depth) == ("data", 1)
    listings = {
        lst.folder: lst
        async for lst in _listings(
            any_fs._ls, "data", concurrency_limit=asyncio.Semaphore(32)
        )
    }
    assert [lst.depth for lst in listings.values()].count(3) == 4


async def test_bottom_up_listings_come_after_their_subfolders() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.002, asynchronous=True)
    _tree(fs)
    folders = [
        lst.folder
        async for lst in _listings(
            fs._ls, "data", topdown=False, concurrency_limit=asyncio.Semaphore(2)
        )
    ]
    assert folders[-1] == "data"
    for i, folder in enumerate(folders):
        assert all(not f.startswith(folder + "/") for f in folders[i + 1 :])
