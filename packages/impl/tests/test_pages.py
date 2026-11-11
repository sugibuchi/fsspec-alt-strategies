"""Tests of `_pages_ahead`, a paginated listing asking for its next page while one is used."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import Any

import pytest

from fsspec_alt_strategies_impl.common.pages import _pages_ahead


class _Pages:
    """Pages of entries, each answered after ``delay``, recording when each is asked and done."""

    def __init__(self, pages: list[list[int]], delay: float = 0.01) -> None:
        self.pages = pages
        self.delay = delay
        self.events: list[str] = []
        self.closed = False
        self.fail_at: int | None = None

    async def __call__(self) -> Any:
        try:
            for i, page in enumerate(self.pages):
                self.events.append(f"ask {i}")
                await asyncio.sleep(self.delay)
                if i == self.fail_at:
                    raise OSError(f"page {i}")
                self.events.append(f"got {i}")
                yield page
        finally:
            self.closed = True


async def test_yields_every_entry_in_order() -> None:
    pages = _Pages([[1, 2], [], [3], [4, 5, 6]])
    assert [e async for e in _pages_ahead(pages())] == [1, 2, 3, 4, 5, 6]
    assert pages.closed


async def test_asks_for_the_next_page_while_one_is_used() -> None:
    pages = _Pages([[0], [1], [2]], delay=0.02)
    async for entry in _pages_ahead(pages()):
        pages.events.append(f"using {entry}")
        await asyncio.sleep(0.03)  # The caller's work on the page.
        pages.events.append(f"used {entry}")
    # Page 1 is asked for, and answered, while page 0 is used.
    assert pages.events.index("ask 1") < pages.events.index("used 0")
    assert pages.events.index("got 1") < pages.events.index("used 0")
    # One page ahead, never two: page 2 waits until page 1 is handed out.
    assert pages.events.index("ask 2") > pages.events.index("using 1")


async def test_a_page_failing_raises_after_the_entries_before_it() -> None:
    pages = _Pages([[1], [2], [3]])
    pages.fail_at = 2
    taken: list[int] = []
    with pytest.raises(OSError, match="page 2"):
        # A loop, not a comprehension: what came before the error is kept.
        async for entry in _pages_ahead(pages()):
            taken.append(entry)  # noqa: PERF401
    assert taken == [1, 2]


async def test_closing_early_cancels_the_page_asked_for() -> None:
    pages = _Pages([[1], [2], [3]], delay=0.05)
    async with aclosing(_pages_ahead(pages())) as entries:
        assert await anext(entries) == 1
        await asyncio.sleep(0.01)  # Page 1 is being asked for.
    # Cancelled, never answered, and the pages closed.
    assert "ask 1" in pages.events
    assert "got 1" not in pages.events
    assert pages.closed


async def test_the_callers_cancellation_is_not_swallowed() -> None:
    pages = _Pages([[1], [2]], delay=0.5)

    async def take_all() -> list[int]:
        return [e async for e in _pages_ahead(pages())]

    task = asyncio.ensure_future(take_all())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pages.closed
