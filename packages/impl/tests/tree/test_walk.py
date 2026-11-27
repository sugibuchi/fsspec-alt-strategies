"""Tests of `_tree_walk`, top down and bottom up."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.asyn import AsyncFileSystem

from fsspec_alt_strategies.profile import StorageProfile
from fsspec_alt_strategies_impl.common.operation import _collect_with_errors
from fsspec_alt_strategies_impl.tree.walk import WalkState, _tree_walk
from fsspec_alt_strategies_impl_testing import (
    SimulatedFileSystem,
    TreeView,
    ils_by_directory,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_PRIMITIVES = {"ils", "info", "rm_file", "rm_dir", "rm_tree", "cp_file", "mv_file"}
"""The keyword arguments that replace a primitive of the filesystem."""

_SETTINGS = {"profile": "storage_profile", "concurrency_limit": "operation_limit"}
"""The keyword arguments that set the filesystem's profile and limit, by its attribute."""


class _ByDirectory(TreeView):
    """A view whose ``ils_recursive`` lists a directory at a time, depth first, by ``ils_by_directory``."""

    def ils_recursive(self, entry: Any, prefix: bool = False) -> Any:
        return ils_by_directory(self, entry)


def _walk(fs: SimulatedFileSystem, path: Any, **kwargs: Any) -> Any:
    """`_tree_walk` on a filesystem's own primitives, which ``kwargs`` may replace."""
    view = _ByDirectory(
        fs,
        **{k: kwargs.pop(k) for k in _PRIMITIVES & kwargs.keys()},
        **{_SETTINGS[k]: kwargs.pop(k) for k in _SETTINGS.keys() & kwargs.keys()},
    )
    return _tree_walk(view, path, **kwargs)


def _tree(fs: SimulatedFileSystem) -> None:
    for year in ("2023", "2024"):
        for month in ("01", "02"):
            fs.add_file(f"data/{year}/{month}/part-0.parquet")
    fs.add_file("data/README.md")


async def _all(items: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in items]


@pytest.mark.parametrize("kwargs", [{}, {"maxdepth": 2}, {"topdown": False}])
async def test_yields_what_fsspecs_walk_yields(
    any_fs: SimulatedFileSystem, kwargs: dict[str, Any]
) -> None:
    _tree(any_fs)
    directories = await _all(_walk(any_fs, "data", **kwargs))
    expected = await _all(AsyncFileSystem._walk(any_fs, "data", detail=True, **kwargs))
    assert sorted(directories, key=lambda f: f[0]) == sorted(
        expected, key=lambda f: f[0]
    )


async def test_yields_each_dir_once(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    directories = await _all(_walk(any_fs, "data"))
    assert sorted(path for path, _, _ in directories) == [
        "data",
        "data/2023",
        "data/2023/01",
        "data/2023/02",
        "data/2024",
        "data/2024/01",
        "data/2024/02",
    ]
    root = next(f for f in directories if f[0] == "data")
    assert (sorted(root[1]), sorted(root[2])) == (["2023", "2024"], ["README.md"])
    assert any_fs.stats.calls == {"list": 7}


@pytest.mark.parametrize("latency", [0.0, 0.002])
async def test_topdown_orders(latency: float) -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=latency, asynchronous=True)
    _tree(fs)
    limit = asyncio.Semaphore(3)
    down = [p for p, _, _ in await _all(_walk(fs, "data", concurrency_limit=limit))]
    up = [
        p
        for p, _, _ in await _all(
            _walk(fs, "data", topdown=False, concurrency_limit=limit)
        )
    ]
    for path in down:
        if path != "data":
            parent = path.rsplit("/", 1)[0]
            assert down.index(parent) < down.index(path)
            assert up.index(parent) > up.index(path)
    assert sorted(up) == sorted(down)


async def test_maxdepth(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    directories = await _all(_walk(any_fs, "data", maxdepth=2))
    assert sorted(p for p, _, _ in directories) == ["data", "data/2023", "data/2024"]
    assert any_fs.stats.calls == {"list": 3}
    with pytest.raises(ValueError, match="at least 1"):
        await _all(_walk(any_fs, "data", maxdepth=0))


async def test_root_errors_are_raised(any_fs: SimulatedFileSystem) -> None:
    _tree(any_fs)
    with pytest.raises(FileNotFoundError):
        await _all(_walk(any_fs, "missing"))
    with pytest.raises(NotADirectoryError):
        await _all(_walk(any_fs, "data/README.md"))


class _Forbidding(SimulatedFileSystem):
    """Refuses to list one directory, as a storage without permission would."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        if self._strip_protocol(path) == "data/2023":
            raise PermissionError(path)
        return await super()._ls(path, detail=detail, **kwargs)


async def test_failed_listing_is_yielded_and_the_walk_goes_on() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    items = await _all(_walk(fs, "data"))
    errors = [i for i in items if isinstance(i, Exception)]
    assert [type(e) for e in errors] == [PermissionError]
    paths = {i[0] for i in items if not isinstance(i, Exception)}
    # Its error instead of its directory, and nothing below it.
    assert not any(p.startswith("data/2023") for p in paths)
    assert {"data", "data/2024", "data/2024/02"} <= paths


async def test_bottom_up_a_failed_listing_is_yielded_and_the_rest_walked() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    items = await _all(_walk(fs, "data", topdown=False))
    # The default listing, a directory per request: data/2023 fails as its own,
    # and its error comes as it does.
    errors = [i for i in items if isinstance(i, Exception)]
    assert [type(e) for e in errors] == [PermissionError]
    walked = [i for i in items if not isinstance(i, Exception)]
    paths = [p for p, _, _ in walked]
    assert sorted(paths) == ["data", "data/2024", "data/2024/01", "data/2024/02"]
    assert paths[-1] == "data"
    # Its parent's listing found it: it is still in its parent's dirs.
    assert sorted(walked[-1][1]) == ["2023", "2024"]


class _Recording(_Forbidding):
    """Records each directory whose listing reached the storage."""

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        self.requested.append(self._strip_protocol(path))
        return await super()._ls(path, detail=detail, **kwargs)


@pytest.mark.parametrize("topdown", [True, False])
async def test_on_error_raise_stops_at_once(topdown: bool) -> None:
    fs = _Recording(hierarchical=True, asynchronous=True)
    fs.requested = []
    _tree(fs)
    with pytest.raises(PermissionError):
        await _all(
            _walk(
                fs,
                "data",
                topdown=topdown,
                on_error="raise",
                concurrency_limit=asyncio.Semaphore(1),
                # Its listing, a directory per request, is depth first.
                profile=StorageProfile(recursive_ls=True),
            )
        )
    if topdown:
        # Its siblings started with it, their pages bound by the permit: none
        # below them is listed.
        assert "data/2023" in fs.requested
        assert set(fs.requested) <= {"data", "data/2023", "data/2024"}
    else:
        # The listing is read a directory at a time, depth first:
        # data/2023 fails, and data/2024, waiting behind it, is never listed.
        assert fs.requested == ["data", "data/2023"]
    assert fs._in_flight == 0


async def test_lists_dirs_concurrently_within_batch_size() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.005, asynchronous=True)
    for i in range(20):
        fs.add_file(f"root/{i:02}/file")
    await _all(_walk(fs, "root", concurrency_limit=asyncio.Semaphore(4)))
    assert fs.stats.calls == {"list": 21}
    assert fs.stats.max_concurrency == 4


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
    # Each alone would list 10 directories at once: together, still 3 at a time.
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
    assert len(await walk) == 7
    await _all_permits_back(limit, 1)


async def test_closing_gives_every_permit_back() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    limit = asyncio.Semaphore(4)
    async with _walk(fs, "root", concurrency_limit=limit) as directories:
        async for _ in directories:
            break  # Listings in flight, and a waiter for the next permit.
    await _all_permits_back(limit, 4)


async def test_closing_cancels_listings_in_flight() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.05, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/file")
    async with _walk(
        fs, "root", concurrency_limit=asyncio.Semaphore(10)
    ) as directories:
        async for _ in directories:
            break  # The root: its ten subdirectories are now in flight.
    # None is left running, and those cancelled before they started were never sent.
    assert fs._in_flight == 0
    assert fs.stats.calls["list"] < 11


async def test_cancel_yields_the_directories_started() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.01, asynchronous=True)
    for i in range(10):
        fs.add_file(f"root/{i}/a/f")
    op = _walk(fs, "root", concurrency_limit=asyncio.Semaphore(4))
    paths = []
    async for path, _, _ in op:
        paths.append(path)
        if path != "root":
            # The root's ten subdirectories have started, their pages bound by
            # the permits: they are walked, but none below them starts.
            op.cancel()
    assert len(paths) == 1 + 10
    assert fs.stats.calls["list"] == 1 + 10
    assert op.cancelled()
    assert op.state == WalkState(
        current=op.state.current, directories_listed=11, found=10 + 10
    )


@pytest.mark.parametrize("topdown", [True, False])
async def test_the_state_names_a_failed_dir(topdown: bool) -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    op = _walk(fs, "data", topdown=topdown)
    async for item in op:
        if isinstance(item, Exception):
            assert op.state.current == "data/2023"
            break
    else:
        pytest.fail("no error")


async def test_on_error_omits_or_calls() -> None:
    fs = _Forbidding(hierarchical=True, asynchronous=True)
    _tree(fs)
    omitted = await _all(_walk(fs, "data", on_error="omit"))
    assert not any(isinstance(i, Exception) for i in omitted)
    called: list[Exception] = []
    items = await _all(_walk(fs, "data", on_error=called.append))
    assert not any(isinstance(i, Exception) for i in items)
    assert [type(e) for e in called] == [PermissionError]
    # Either way, the directories that did list are all there.
    assert sorted(i[0] for i in items) == sorted(i[0] for i in omitted)


def test_on_error_is_checked_at_once(any_fs: SimulatedFileSystem) -> None:
    with pytest.raises(ValueError, match="on_error"):
        _walk(any_fs, "data", on_error="ignore")


async def test_bottom_up_directories_come_after_their_subdirs() -> None:
    fs = SimulatedFileSystem(hierarchical=True, latency=0.002, asynchronous=True)
    _tree(fs)
    walk = _walk(fs, "data", topdown=False, concurrency_limit=asyncio.Semaphore(2))
    directories = [directory async for directory, _, _ in walk]
    assert directories[-1] == "data"
    for i, directory in enumerate(directories):
        assert all(not f.startswith(directory + "/") for f in directories[i + 1 :])


async def test_an_empty_dir_is_yielded_empty() -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    fs.add_dir("data")
    for topdown in (True, False):
        assert await _all(_walk(fs, "data", topdown=topdown)) == [("data", {}, {})]


@pytest.mark.parametrize("maxdepth", [None, 3])
async def test_bottom_up_each_directory_comes_once_after_all_below_it(
    any_fs: SimulatedFileSystem, maxdepth: int | None
) -> None:
    # Recursively listed without maxdepth, as declared, a directory per request with it.
    _tree(any_fs)
    profile = StorageProfile(recursive_ls=True)
    view = TreeView(any_fs, recursive_listing=maxdepth is None, storage_profile=profile)
    walk = _tree_walk(view, "data", topdown=False, maxdepth=maxdepth)
    order = [directory async for directory, _, _ in walk]
    assert len(order) == len(set(order))
    assert order[-1] == "data"
    for i, directory in enumerate(order):
        assert not any(d.startswith(directory + "/") for d in order[i + 1 :])


def _streamed_pages(fs: SimulatedFileSystem, page: int) -> Any:
    """A directory's listing a page per request, each yielded as it arrives, as real storages stream them."""

    async def ils(path: str, **kwargs: Any) -> Any:
        entries = await fs._ls(path, detail=True)
        for start in range(0, len(entries), page):
            await asyncio.sleep(0.005)
            for entry in entries[start : start + page]:
                yield entry

    return ils


async def test_top_down_a_directory_comes_before_those_below_it_while_it_streams() -> (
    None
):
    # big/a is found in big's first part, and listed while big's goes on:
    # its listing ends first, but it waits for big's.
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True)
    fs.add_file("big/a/x")
    fs.add_file("big/a/b/y")
    for i in range(3000):
        fs.add_file(f"big/f{i:04}")
    view = TreeView(fs, ils=_streamed_pages(fs, 500))
    walked, errors = await _collect_with_errors(_tree_walk(view, "big", topdown=True))
    assert errors == []
    assert [d for d, _, _ in walked] == ["big", "big/a", "big/a/b"]


@pytest.mark.parametrize("how", ["end", "stop", "close"])
async def test_every_permit_is_given_back(how: str) -> None:
    fs = SimulatedFileSystem(hierarchical=True, asynchronous=True, latency=0.002)
    _tree(fs)
    limit = asyncio.Semaphore(3)
    op = _walk(fs, "data", maxdepth=4, concurrency_limit=limit)
    async for _ in op:
        if how == "stop":
            op.cancel()
        elif how == "close":
            break
    await op.aclose()
    assert limit._value == 3
