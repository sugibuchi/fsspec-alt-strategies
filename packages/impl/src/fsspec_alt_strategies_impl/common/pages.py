# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A paginated listing's entries, with the next page asked for while one is used.

A storage lists in pages, each asked for with the token the one before it ends
with. An SDK's pager asks for the next page only once the last is used, so each
page costs the storage's time to answer, the client's to parse it, and the
caller's to use it, one after another. Measured on Azure's ListBlobs, a page
of 5000 blobs took about 0.4 s to answer and 0.55 s to parse.

`_pages_ahead` asks for the next page as soon as one is handed out. While the
storage answers, the caller uses the page before, so a page costs the larger of
the two, not their sum. A filesystem's ``_ils`` lists with it, e.g. its
recursive listing.

This module depends on nothing else in the package.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable

__all__: list[str] = []

_T = TypeVar("_T")

_END = object()
"""What `_next` returns once the pages are all taken."""


async def _pages_ahead(
    pages: AsyncIterator[Iterable[_T]],
) -> AsyncGenerator[_T]:
    """Each entry of each page, the next page asked for while one is used.

    ``pages`` is an async iterator of pages, each an iterable of entries, e.g.
    a generator over an SDK's pager. One page more than the caller has taken is
    asked for: a request at a time, as a pager sends them, so the listing holds
    one request in flight, as before, and one page more in memory.

    A page's error is raised when its entries are due, after the entries of
    the pages before it. Closed early, the page asked for is cancelled, and
    ``pages`` closed.
    """
    iterator = aiter(pages)
    ahead: asyncio.Task[Any] = asyncio.ensure_future(_next(iterator))
    try:
        while True:
            page = await ahead
            if page is _END:
                return
            # Asked for before this page is used: its answer comes meanwhile.
            ahead = asyncio.ensure_future(_next(iterator))
            for entry in page:
                yield entry
    finally:
        ahead.cancel()
        # Its own outcome, cancelled or an error no entry was due for, is
        # dropped. The caller's cancellation, meanwhile, still raises.
        await asyncio.gather(ahead, return_exceptions=True)
        if (aclose := getattr(iterator, "aclose", None)) is not None:
            await aclose()


async def _next(iterator: AsyncIterator[Any]) -> Any:
    """The next page, or `_END`: a task cannot end by StopAsyncIteration."""
    try:
        return await anext(iterator)
    except StopAsyncIteration:
        return _END
