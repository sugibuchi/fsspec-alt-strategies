# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The listing under the flat strategies: a prefix's keys, and the folders they imply.

A flat storage lists the keys under a prefix, at any depth, but no folders. A
folder exists as the common prefix of keys, or as an empty marker object whose
key ends with "/". fsspec's ``_find`` and ``_glob`` answer with folders too, so
the strategies derive them from the keys, as `_folders` does.

The listing, ``ils``, holds no permit of the concurrency limit: its pages come
one after another, and the strategy never sees them. Only the other requests,
such as an ``info``, hold one.

This module depends on `walk` of `tree`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..tree.walk import _limit, _trim

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncGenerator, AsyncIterator, Callable
    from typing import TypeAlias

    from ..tree.walk import _Info

    _Ils: TypeAlias = Callable[[str], AsyncIterator[dict[str, Any]]]
    """Lists every key starting with a prefix, called as ``ils(prefix)``."""


def _folder(name: str) -> dict[str, Any]:
    """A folder's entry, as fsspec's filesystems derive one for an object store."""
    return {"name": name, "size": 0, "type": "directory"}


def _folders(name: str) -> list[str]:
    """The folders a key implies, the shallowest first.

    They are the key's parents, and the key itself if it is a folder's marker,
    ending with "/".
    """
    parts = name.rstrip("/").split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts) + name.endswith("/"))]


def _limited(info: _Info, limit: asyncio.Semaphore) -> _Info:
    """``info``, holding a permit of ``limit`` while it runs."""

    async def limited(path: str) -> dict[str, Any]:
        async with limit:
            return await info(path)

    return limited


async def _close(entries: AsyncIterator[dict[str, Any]]) -> None:
    """Close a listing left early, if it can be: an async generator can."""
    if (aclose := getattr(entries, "aclose", None)) is not None:
        await aclose()


async def _scan(
    ils: _Ils,
    info: _Info,
    path: str,
    *,
    maxdepth: int | None = None,
    withdirs: bool = False,
    concurrency_limit: asyncio.Semaphore | None = None,
    stopping: Callable[[], object] | None = None,
) -> AsyncGenerator[dict[str, Any]]:
    """The entries under ``path``, as fsspec's ``_find`` finds them.

    It yields each file under ``path`` as the listing gives it, and with
    ``withdirs`` each folder once, derived from the keys. ``path`` itself is
    treated as fsspec's ``_find`` treats it:

    - missing, it yields nothing;
    - a file, it yields the file's ``info`` entry;
    - a folder, it yields the folder's own ``info`` entry with ``withdirs``,
      unless stopping.

    Once ``stopping`` returns true, it takes no more from the listing. An error
    of the listing ends it, as it ends the only listing there is.
    """
    root = _trim(path)
    prefix = f"{root}/" if root else ""
    info = _limited(info, _limit(concurrency_limit))
    under = False
    seen: set[str] = set()
    entries = ils(prefix)
    try:
        async for entry in entries:
            under = True
            name = entry["name"]
            if withdirs:
                for folder in _folders(name[len(prefix) :]):
                    full = prefix + folder
                    if full not in seen and _within(folder, maxdepth):
                        seen.add(full)
                        yield _folder(full)
            if not name.endswith("/") and _within(name[len(prefix) :], maxdepth):
                yield entry
            if stopping is not None and stopping():
                return
    finally:
        await _close(entries)
    if not root or (under and not withdirs):
        return
    try:
        entry = await info(root)
    except FileNotFoundError:
        return
    # A folder is found itself only with withdirs. A file is found itself only
    # when nothing is under it, as fsspec's _find tells them apart.
    is_folder = entry["type"] == "directory"
    if (is_folder and withdirs) or (not is_folder and not under):
        yield entry


def _within(relative: str, maxdepth: int | None) -> bool:
    """Whether a path, relative to the root, is at most ``maxdepth`` deep."""
    return maxdepth is None or relative.count("/") + 1 <= maxdepth
