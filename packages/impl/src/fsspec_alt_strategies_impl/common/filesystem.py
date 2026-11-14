# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What the filesystem classes of the families share, e.g. `TreeFileSystem`'s.

`_BaseFileSystem` moves data by its handler: its ``_cat_file``,
``_pipe_file``, ``_get_file`` and ``_put_file`` call the handler's, which
move it through the handler's files unless the handler does otherwise.
Its ``_cp_file`` and ``_mv_file`` ask a path's ``_info``, then copy or move by
the families' ``_cp_file_entry`` and ``_mv_file_entry``. fsspec's methods on
many paths, such as ``_cat``, call those on one path ``io_concurrency`` at a
time. Its ``_copy`` and ``_mv`` start the family's copy and move, which each
family's class names.

The functions below it shape the answers and errors of fsspec's methods, as
the families' classes give them.

This module depends on `operation` and `handler`.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from glob import has_magic
from typing import TYPE_CHECKING, Any, Self

from fsspec.asyn import AsyncFileSystem, _run_coros_in_chunks, sync
from fsspec.callbacks import DEFAULT_CALLBACK

from fsspec_alt_strategies.monitor import Call, Monitored

from .handler import _awaited
from .operation import _collect_with_monitors

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable, Sequence

    from fsspec.callbacks import Callback

    from .handler import StorageHandler


_MiB = 1024 * 1024


class _BaseFileSystem(AsyncFileSystem, Monitored):
    """An async filesystem on a handler, moving data through its files.

    What the storage does is the handler's alone: its profile, which the
    strategies read from it, its limits, ``operation_limit`` and
    ``io_limit``, and its data methods. The filesystem exposes none.
    """

    handler: StorageHandler
    """The storage's handler, whose primitives the filesystem calls."""

    def __init__(
        self,
        handler: StorageHandler,
        *,
        asynchronous: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        use_listings_cache: bool = True,
        listings_expiry_time: float | None = None,
        max_paths: int | None = None,
    ) -> None:
        # fsspec's own options, named: fsspec's constructors take any keyword
        # and ignore those they do not know, so a misspelled option would
        # change nothing, silently. fsspec's batch_size is not among them:
        # the handler's io_concurrency bounds the paths handled at a time.
        super().__init__(
            asynchronous=asynchronous,
            loop=loop,
            use_listings_cache=use_listings_cache,
            listings_expiry_time=listings_expiry_time,
            max_paths=max_paths,
        )

        self.handler = handler

    def unstrip_protocol(self, name: str) -> str:
        """A path of the storage as a URL: the handler's."""
        return self.handler.unstrip_protocol(name)

    def _unwrapped(self) -> AsyncFileSystem:
        """The filesystem a monitor is told of: this one, or the one a wrapper wraps."""
        wrapped = self.handler.wrapped
        return wrapped if wrapped is not None else self

    # Files, by the handler's open_file: the same file for open_async and the sync open.

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        """Open a file whose methods are awaited, as fsspec's ``open_async`` does, by the handler's `open_file`.

        A file of the package's is given fsspec's async API by an
        `AsyncFileAdapter`. One whose methods are awaited already, e.g. a
        wrapped filesystem's, is returned as it is. A file opened to write
        drops the listings kept, as it opens: the file knows no filesystem.
        """
        if kwargs.pop("compression", None):
            raise ValueError("Compression is not supported by open_async")
        _check_mode(mode)
        if mode != "rb":
            self._drop_listings()
        entry = self.handler.entry(path)
        return _awaited(await self.handler.open_file(entry, mode, **kwargs))

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,
        cache_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Open a file for the sync ``open``, by the handler's `open_sync_file`, on the filesystem's loop.

        Within a transaction, a file written is committed with it: no reader
        sees it before. A file opened to write drops the listings kept, as
        `open_async` does.
        """
        _check_mode(mode)
        if mode != "rb":
            self._drop_listings()
        return self.handler.open_sync_file(
            self.handler.entry(path),
            mode,
            loop=self.loop,
            autocommit=autocommit,
            block_size=block_size,
            cache_options=cache_options,
            **kwargs,
        )

    # fsspec's dircache: listings kept by _ls, for fsspec's own methods and callers.

    async def _ls(
        self, path: str, detail: bool = True, refresh: bool = False, **kwargs: Any
    ) -> Any:
        """List a directory, as fsspec's ``_ls`` does: all of ``_ils``'s entries, or their names.

        As in fsspec's implementations, the listing is kept in ``dircache``
        once listed whole, and a later ``_ls`` of the path answers from it,
        unless ``refresh=True``. ``use_listings_cache=False`` keeps none. A
        listing with other keyword arguments, e.g. a version, is not kept.

        Only ``_ls`` uses the listings kept. The strategies list with ``_ils``,
        which neither reads nor fills them, so a listing kept never changes
        what they do. ``_info`` asks the storage, never them. Any change made
        through the filesystem drops them all, by `_drop_listings`. A change made
        by another client is not seen until then, or until ``refresh=True`` or
        ``invalidate_cache()``.
        """
        key = self._strip_protocol(path)
        entries = None if refresh or kwargs else self.dircache.get(key)
        if entries is None:
            entries = [entry async for entry in self._ils(path, **kwargs)]  # type: ignore[attr-defined]
            if not kwargs:
                self.dircache[key] = entries
        return list(entries) if detail else [e["name"] for e in entries]

    def _drop_listings(self) -> None:
        """Drop all the listings kept: each method changing the storage calls it once it ends, failed or not."""
        self.dircache.clear()

    def invalidate_cache(self, path: str | None = None) -> None:
        """Drop the listings kept of ``path``, of what is below it, and of its parents: all, without one."""
        if path is None:
            self.dircache.clear()
        else:
            path = self._strip_protocol(path)
            for key in list(self.dircache):
                below, above = key.startswith(f"{path}/"), path.startswith(f"{key}/")
                if key == path or below or above or not key:
                    self.dircache.pop(key, None)
        super().invalidate_cache(path)
        self.handler.invalidate_cache(path)  # E.g. a wrapped filesystem's own.

    def close(self) -> None:
        """Release what the filesystem holds: its handler's, by the handler's sync `close`.

        fsspec never calls it. A router closes its inner filesystems with it,
        and ``with`` closes the filesystem it opened. It may be called more
        than once: each releases what is held then.
        """
        self.handler.close()

    # fsspec's sync methods it leaves to each filesystem: the handler's, awaited on
    # the filesystem's loop. A handler without them raises NotImplementedError.

    def created(self, path: str) -> Any:
        """When an item was created, as fsspec's ``created`` answers: the handler's `created`."""
        return sync(self.loop, self.handler.created, self.handler.entry(path))

    def modified(self, path: str) -> Any:
        """When an item was last modified, as fsspec's ``modified`` answers: the handler's `modified`."""
        return sync(self.loop, self.handler.modified, self.handler.entry(path))

    def sign(self, path: str, expiration: int = 100, **kwargs: Any) -> str:
        """A signed URL of a file, as fsspec's ``sign`` answers: the handler's `sign`."""
        entry = self.handler.entry(path)
        return sync(self.loop, self.handler.sign, entry, expiration, **kwargs)  # type: ignore[no-any-return]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def _single_file(self, path: str | Sequence[str]) -> dict[str, Any] | None:
        """The entry of the one existing file ``path`` names, with no pattern, else None.

        fsspec's ``mv`` of a file that is not recursive copies it, then deletes
        it, never calling ``_mv_file``. The families move such a file with
        their recursive strategy instead, which moves it as a recursive move
        would: by the storage's rename, where the class uses one. A copy goes
        the same way. The entry goes to the strategy, which then need not list
        the file to tell what it is.
        """
        if not isinstance(path, str) or has_magic(path) or path.endswith("/"):
            return None

        try:
            entry = await self._info(path)
        except FileNotFoundError:
            return None

        return entry if entry["type"] != "directory" else None

    # The family's copy and move, e.g. _tree_cp and _tree_mv: each family's class sets them.
    _cp_strategy: Callable[..., Any]
    _mv_strategy: Callable[..., Any]

    async def _copy(
        self,
        path1: str | list[str],
        path2: str | list[str],
        recursive: bool = False,
        on_error: str | None = None,
        maxdepth: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Copy files, as fsspec's ``copy`` does: with the family's copy when recursive.

        A single file goes by the family's copy too, recursive or not, which
        needs not ask it again, as fsspec's does.
        """
        try:
            entry = None
            if isinstance(path2, str) and not recursive:
                entry = await self._single_file(path1)
            if not isinstance(path2, str) or (not recursive and entry is None):
                await super()._copy(
                    path1, path2, recursive, on_error, maxdepth, batch_size, **kwargs
                )
                return
            if entry is not None:
                # As fsspec's copy: a path given, not recursive, raises if missing.
                on_error = on_error or "raise"

            _, errors = await _collect_with_monitors(
                self._cp_strategy(
                    self.handler,
                    self.handler,
                    _strip_only_protocol(self, path1),
                    _strip_only_protocol(self, path2),
                    maxdepth=maxdepth,
                    root_entry=entry,
                ),
                Call(self._unwrapped(), "copy", path1, path2),
                kwargs.get("callback"),
            )
            _raise(_copy_errors(errors, on_error), "Copying")
        finally:
            self._drop_listings()

    async def _mv(
        self,
        path1: str | list[str],
        path2: str,
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Move files, as fsspec's ``mv`` does: with the family's move when recursive.

        A single file goes by the family's move too, recursive or not, so that
        it moves by ``_mv_file_entry``, the storage's rename where it has one.
        """
        try:
            if path1 == path2:
                return

            entry = None
            if not recursive:
                entry = await self._single_file(path1)
                if entry is None:
                    # fsspec's own: a copy, then the deletion of its sources.
                    await self._copy(
                        path1, path2, recursive, "raise", maxdepth, **kwargs
                    )
                    await self._rm(path1, recursive=recursive)
                    return

            _, errors = await _collect_with_monitors(
                self._mv_strategy(
                    self.handler,
                    self.handler,
                    _strip_only_protocol(self, path1),
                    _strip_only_protocol(self, path2),
                    maxdepth=maxdepth,
                    root_entry=entry,
                ),
                Call(self._unwrapped(), "mv", path1, path2),
                kwargs.get("callback"),
            )
            _raise(errors, "Moving")
        finally:
            self._drop_listings()

    # fsspec makes sync methods only from the async ones it knows: _mv and _rmdir are
    # not. Each awaits the instance's own, so that a subclass's override is the one run.

    def mv(self, path1: str | list[str], path2: str, **kwargs: Any) -> None:
        return sync(self.loop, self._mv, path1, path2, **kwargs)

    def rmdir(self, path: str) -> None:
        return sync(self.loop, self._rmdir, path)

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        """Copy a file, as fsspec's ``_cp_file`` does: by ``_cp_file_entry``, with its path alone.

        The strategies pass on the entry their listing found. fsspec's own
        methods, and other callers, know only the path: the handler takes an
        entry whose keys are unknown, asking the storage only where it needs
        them.
        """
        try:
            entry = self.handler.entry(path1)
            await self._cp_file_entry(entry, path2, **kwargs)  # type: ignore[attr-defined]
        finally:
            self._drop_listings()

    async def _mv_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        """Move a file, as fsspec's ``_mv_file`` does: by ``_mv_file_entry``, with its ``_info``.

        As `_cp_file`, for a move.
        """
        try:
            entry = self.handler.entry(path1)
            await self._mv_file_entry(entry, path2, **kwargs)  # type: ignore[attr-defined]
        finally:
            self._drop_listings()

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        """Read a file, or a range of it, by the handler's `cat_file`.

        The handler holds the permits of ``io_limit``, as its files or its
        own requests take them. fsspec's methods on many paths call it at
        most ``io_concurrency`` at a time.
        """
        entry = self.handler.entry(path)
        return await self.handler.cat_file(entry, start=start, end=end, **kwargs)

    async def _pipe_file(
        self, path: str, value: bytes, mode: str = "overwrite", **kwargs: Any
    ) -> None:
        """Write a file whole, by the handler's `pipe_file`: see `_cat_file`."""
        try:
            entry = self.handler.entry(path)
            await self.handler.pipe_file(entry, value, mode=mode, **kwargs)
        finally:
            self._drop_listings()

    async def _get_file(
        self,
        rpath: str,
        lpath: str,
        callback: Callback = DEFAULT_CALLBACK,
        **kwargs: Any,
    ) -> None:
        """Copy a file to a local path, by the handler's `get_file`: see `_cat_file`."""
        if os.path.isdir(lpath):  # noqa: ASYNC240, local files are short next to requests.
            return  # A directory, which fsspec's get makes itself.
        entry = self.handler.entry(rpath)
        await self.handler.get_file(entry, lpath, callback=callback, **kwargs)

    async def _put_file(
        self,
        lpath: str,
        rpath: str,
        mode: str = "overwrite",
        callback: Callback = DEFAULT_CALLBACK,
        **kwargs: Any,
    ) -> None:
        """Copy a local file to the storage, by the handler's `put_file`: see `_cat_file`."""
        try:
            if os.path.isdir(lpath):  # noqa: ASYNC240, local files are short next to requests.
                return  # A directory: a storage needs none for the files put in it.
            entry = self.handler.entry(rpath)
            await self.handler.put_file(
                lpath, entry, mode=mode, callback=callback, **kwargs
            )
        finally:
            self._drop_listings()

    # fsspec's methods on many paths call the method on one path for each, up to
    # batch_size at a time: here by default, io_concurrency. So they bound the
    # paths in flight, while the handler's permits bound its requests.

    def _io_batch_size(self, batch_size: int | None) -> int | None:
        """The paths handled at a time: the call's, else ``io_concurrency``.

        ``io_concurrency`` given as a semaphore tells no number: fsspec's own
        default then holds, and only the permits bound the requests.
        """
        if batch_size is not None:
            return batch_size
        if isinstance(self.handler.io_concurrency, int):
            return self.handler.io_concurrency
        else:
            return None

    async def _cat(
        self,
        path: Any,
        recursive: bool = False,
        on_error: str = "raise",
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        return await super()._cat(
            path,
            recursive=recursive,
            on_error=on_error,
            batch_size=self._io_batch_size(batch_size),
            **kwargs,
        )

    async def _cat_ranges(
        self,
        paths: list[str],
        starts: Any,
        ends: Any,
        max_gap: int | None = None,
        batch_size: int | None = None,
        on_error: str = "return",
        **kwargs: Any,
    ) -> list[Any]:
        """Read byte ranges of files, as fsspec's ``cat_ranges`` does, in fewer requests.

        fsspec's sends one ``_cat_file`` per range. This merges the ranges of a
        file whose gaps are at most ``max_gap``, or else ``ranges_max_gap``,
        fetching the gaps too. A merged range, overlaps and repeats included,
        is at most ``ranges_max_block``.
        A merged range from twice ``ranges_split_size`` is fetched in parts of
        at least that, up to ``ranges_spread`` of them, at once. Each range is
        then cut from what was fetched, in the order asked. A range counted
        from the end, or to it, is fetched alone: merging it would ask the
        file's size first.

        It answers as fsspec's: the same bytes, a range beyond the end short or
        empty, and a failure in place of each range it covers with
        ``on_error="return"``, or raised otherwise.
        """
        starts, ends = _range_bounds(paths, starts, ends)
        gap = self.ranges_max_gap if max_gap is None else max_gap
        fetches, plan = _merge_ranges(paths, starts, ends, gap, self.ranges_max_block)
        fetched = await self._fetch_ranges(fetches, batch_size, kwargs)
        out = [_cut(fetched[index], offset, length) for index, offset, length in plan]

        if on_error != "return":
            for result in out:
                if isinstance(result, BaseException):
                    raise result
        return out

    async def _fetch_ranges(
        self,
        fetches: list[tuple[str, int | None, int | None]],
        batch_size: int | None,
        kwargs: dict[str, Any],
    ) -> list[bytearray | BaseException]:
        """Each range's bytes, or its first failed part's error.

        The parts of all the ranges run together, as many at a time as
        `_io_batch_size` says.
        """
        parts = [
            (index, part)
            for index, (_, start, end) in enumerate(fetches)
            for part in self._ranges_parts(start, end)
        ]
        results = await _run_coros_in_chunks(
            [
                self._cat_file(fetches[index][0], start=start, end=end, **kwargs)
                for index, (start, end) in parts
            ],
            batch_size=self._io_batch_size(batch_size),
            nofiles=True,
            return_exceptions=True,
        )
        fetched: list[bytearray | BaseException] = [bytearray() for _ in fetches]
        for (index, _), result in zip(parts, results, strict=True):
            data = fetched[index]
            if isinstance(result, BaseException) and not isinstance(
                data, BaseException
            ):
                fetched[index] = result
            elif isinstance(data, bytearray):
                data += result
        return fetched

    ranges_max_gap: int = _MiB
    """The longest gap ``cat_ranges`` fetches to merge two ranges of a file.

    Measured on Azure, a request took about 19 ms, and a MiB more in it about
    16 ms: fetching a gap of up to about 1 MiB costs less than a request.
    """

    ranges_max_block: int = 64 * _MiB
    """The longest range ``cat_ranges`` merges ranges into."""

    ranges_split_size: int = 4 * _MiB
    """The smallest part ``cat_ranges`` fetches a merged range in, as the files' ``split_size``."""

    ranges_spread: int = 8
    """The most parts ``cat_ranges`` fetches a merged range in at once."""

    def _ranges_parts(
        self, start: int | None, end: int | None
    ) -> list[tuple[int | None, int | None]]:
        """The parts a range is fetched in: even, of at least ``ranges_split_size``."""
        if start is None or end is None or start < 0 or end < 0:
            return [(start, end)]
        size = end - start
        parts = max(1, min(self.ranges_spread, size // self.ranges_split_size))
        step = math.ceil(size / parts) if size > 0 else 1
        return [(s, min(s + step, end)) for s in range(start, end, step)] or [
            (start, end)
        ]

    async def _pipe(
        self, path: Any, value: Any = None, batch_size: int | None = None, **kwargs: Any
    ) -> Any:
        return await super()._pipe(
            path, value, batch_size=self._io_batch_size(batch_size), **kwargs
        )

    async def _get(self, rpath: Any, lpath: Any, **kwargs: Any) -> Any:
        kwargs["batch_size"] = self._io_batch_size(kwargs.get("batch_size"))
        return await super()._get(rpath, lpath, **kwargs)

    async def _put(self, lpath: Any, rpath: Any, **kwargs: Any) -> Any:
        kwargs["batch_size"] = self._io_batch_size(kwargs.get("batch_size"))
        return await super()._put(lpath, rpath, **kwargs)


def _range_bounds(
    paths: list[str], starts: Any, ends: Any
) -> tuple[list[int | None], list[int | None]]:
    """``starts`` and ``ends`` as lists, one per path, as fsspec's ``cat_ranges`` takes them.

    A single value is every range's. It raises as fsspec's does: TypeError for
    paths not a list, ValueError for counts that differ.
    """
    if not isinstance(paths, list):
        raise TypeError(f"paths must be a list: {type(paths).__name__}")
    count = len(paths)
    starts = list(starts) if isinstance(starts, Iterable) else [starts] * count
    ends = list(ends) if isinstance(ends, Iterable) else [ends] * count
    if len(starts) != count or len(ends) != count:
        raise ValueError("paths, starts and ends must be as many")
    return starts, ends


def _cut(
    data: bytearray | BaseException, offset: int, length: int | None
) -> bytes | BaseException:
    """A range asked for, cut from what its fetch returned, or the fetch's failure."""
    if isinstance(data, BaseException):
        return data
    if length is None:
        return bytes(data)
    else:
        return bytes(data[offset : offset + max(0, length)])


def _merge_ranges(
    paths: list[str],
    starts: list[int | None],
    ends: list[int | None],
    max_gap: int,
    max_block: int,
) -> tuple[list[tuple[str, int | None, int | None]], list[tuple[int, int, int | None]]]:
    """The ranges to fetch, and for each range asked, where to cut it from: ``(fetch, offset, length)``.

    The ranges of a file with known ends, sorted, are merged while the gap is at
    most ``max_gap`` and the merged range at most ``max_block``. A range
    counted from the end, or to it, is fetched alone, its length None: whole.
    """
    fetches: list[tuple[str, int | None, int | None]] = []
    plan: list[tuple[int, int, int | None] | None] = [None] * len(paths)
    known: dict[str, list[int]] = {}
    for i, (path, start, end) in enumerate(zip(paths, starts, ends, strict=True)):
        if (start is None or start >= 0) and end is not None and end >= 0:
            known.setdefault(path, []).append(i)
        else:
            plan[i] = (len(fetches), 0, None)
            fetches.append((path, start, end))

    for path, indices in known.items():
        indices.sort(key=lambda i: (starts[i] or 0, ends[i]))
        members: list[int] = []
        low = high = 0
        for index in [*indices, None]:
            begin = finish = 0
            if index is not None:
                begin, known_end = starts[index] or 0, ends[index]
                assert known_end is not None
                finish = max(known_end, begin)
                if (
                    members
                    and begin - high <= max_gap
                    and max(finish, high) - low <= max_block
                ):
                    high = max(high, finish)
                    members.append(index)
                    continue
            if members:
                for m in members:
                    first = starts[m] or 0
                    plan[m] = (
                        len(fetches),
                        first - low,
                        max(ends[m] or 0, first) - first,
                    )
                fetches.append((path, low, high))
            if index is not None:
                members, low, high = [index], begin, finish

    return fetches, plan  # type: ignore[return-value]


def _strip_only_protocol(fs: Any, path: str | Sequence[str]) -> Any:
    """A path, or paths, of ``fs`` without its protocol, keeping a trailing "/".

    The strategies take paths without a protocol. A trailing "/" means
    something to some, e.g. a glob of directories only, or a copy into a directory.

    A path with no pattern is verified by ``fs``'s handler as given, so that
    what a URL names beyond the path, e.g. an account, is checked before it
    is stripped. A pattern is stripped only: no handler verifies a pattern,
    and the strategy verifies the directory it starts from.
    """
    if not isinstance(path, str):
        return [_strip_only_protocol(fs, p) for p in path]
    plain = fs._strip_protocol(path) if has_magic(path) else fs.handler.verified(path)
    return plain + "/" if path.endswith("/") and not plain.endswith("/") else plain


def _check_mode(mode: str) -> None:
    """Refuse a mode `StorageHandler.open_file` does not take, before the handler is asked."""
    if mode not in ("rb", "wb", "xb"):
        raise ValueError(f"mode must be 'rb', 'wb' or 'xb': {mode!r}")


def _answer(entries: list[dict[str, Any]], detail: bool) -> Any:
    """What fsspec's ``_find`` and ``_glob`` answer: the paths sorted, or their entries."""
    found = {e["name"].rstrip("/"): e for e in entries}
    names = sorted(found)
    return {n: found[n] for n in names} if detail else names


def _du_answer(size: int, entries: list[dict[str, Any]], total: bool) -> Any:
    """What fsspec's ``_du`` answers: the total size, or each path's."""
    if total:
        return size
    else:
        return {e["name"]: e.get("size") or 0 for e in entries}


def _copy_errors(errors: list[Exception], on_error: str | None) -> list[Exception]:
    """The errors of a recursive copy, as fsspec's raises them.

    A source gone meanwhile is no error, unless ``on_error`` is "raise".
    """
    if on_error == "raise":
        return errors
    else:
        return [e for e in errors if not isinstance(e, FileNotFoundError)]


def _raise(errors: list[Exception], doing: str) -> None:
    """Raise the errors an operation yielded: one alone, several in a group."""
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup(f"{doing} failed for {len(errors)} paths", errors)


def _failed_dir(
    directory: str,
    error: Exception,
    on_error: str | Callable[[Exception], Any],
    detail: bool,
) -> tuple[str, Any, Any]:
    """What fsspec's ``walk`` yields for a directory that failed to list, the root too."""
    if on_error == "raise":
        raise error
    if callable(on_error):
        on_error(error)
    return (directory, {}, {}) if detail else (directory, [], [])
