# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What the filesystem classes of the families share, e.g. `TreeFileSystem`'s.

`_BaseFileSystem` holds the two concurrency limits, and moves data through
file objects: its ``_cat_file``, ``_pipe_file``, ``_get_file`` and
``_put_file`` open the file with ``open_async``, which a subclass implements.
Its ``_cp_file`` and ``_mv_file`` ask a path's ``_info``, then copy or move by
the families' ``_cp_file_entry`` and ``_mv_file_entry``. fsspec's methods on
many paths, such as ``_cat``, call those on one path ``io_concurrency`` at a
time. Its ``_copy`` and ``_mv`` start the family's copy and move, which each
family's class names.

The functions below it shape the answers and errors of fsspec's methods, as
the families' classes give them.

This module depends on `operation`.
"""

from __future__ import annotations

import asyncio
import io
import math
import os
from abc import ABCMeta
from collections.abc import Iterable
from glob import has_magic
from typing import TYPE_CHECKING, Any, Self

from fsspec.asyn import AsyncFileSystem, _run_coros_in_chunks, sync
from fsspec.callbacks import DEFAULT_CALLBACK

from fsspec_alt_strategies.monitor import Call, Monitored
from fsspec_alt_strategies.profile import UNKNOWN, StorageProfile

from .operation import _collect_with_monitors

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fsspec import AbstractFileSystem
    from fsspec.callbacks import Callback


_MiB = 1024 * 1024


class _AbstractFileSystemMeta(type(AsyncFileSystem), ABCMeta):  # type: ignore[misc]
    """fsspec's metaclass, which caches instances, with ABCMeta's abstract methods.

    A filesystem class has fsspec's metaclass already. ABCMeta must join it, so
    that a subclass missing an abstract method fails at once, when made.
    """


class _BaseFileSystem(AsyncFileSystem, Monitored, metaclass=_AbstractFileSystemMeta):
    """An async filesystem with two concurrency limits, moving data through files.

    ``operation_limit`` bounds the requests of the strategies, such as
    listings, deletions and copies. ``io_limit`` bounds the reads and writes of
    data: ``open_async`` gives it to the files it opens, as their
    ``concurrency_limit``. Each is an `asyncio.Semaphore`: the one given as
    the concurrency, so that several filesystems share it, or else a new one,
    of the number given.

    ``storage_profile`` tells the strategies what the storage does. It is one
    for the whole filesystem, and always set: one whose storage is unknown says
    so, with ``UNKNOWN``. Where a request finds it, e.g. whether an
    account has a hierarchical namespace, a subclass sets what it knows and
    overrides ``_load_storage_profile()`` too. The strategies and the
    subclass's primitives get it with ``await self._get_storage_profile()``,
    which loads it once, sets it as ``storage_profile``, and returns it.
    Neither takes a path, so that the profile cannot differ by path. A storage
    whose behaviour does, e.g. per bucket, is several filesystems under a
    router: one per bucket, and the router for the whole namespace.

    Each family's class declares the abstract methods its subclasses implement,
    so that a reader finds them all in one place.
    """

    storage_profile: StorageProfile = UNKNOWN
    """What the storage does, for the strategies: unknown, right on any storage, by default.

    A subclass sets what it knows. One overriding ``_load_storage_profile``
    too has it replaced by the profile `_get_storage_profile` loads.
    """

    transfer_size: int = 8 * _MiB
    """The bytes read or written at a time by ``_get_file`` and ``_put_file``."""

    def __init__(
        self,
        *args: Any,
        operation_concurrency: int | asyncio.Semaphore,
        io_concurrency: int | asyncio.Semaphore,
        **kwargs: Any,
    ) -> None:
        # A semaphore given is shared. A number makes a new one.
        if isinstance(operation_concurrency, asyncio.Semaphore):
            operation_limit = operation_concurrency
        elif operation_concurrency < 1:
            raise ValueError(
                f"operation_concurrency must be at least 1: {operation_concurrency}"
            )
        else:
            operation_limit = asyncio.Semaphore(operation_concurrency)

        if isinstance(io_concurrency, asyncio.Semaphore):
            io_limit = io_concurrency
        elif io_concurrency < 1:
            raise ValueError(f"io_concurrency must be at least 1: {io_concurrency}")
        else:
            io_limit = asyncio.Semaphore(io_concurrency)

        super().__init__(*args, **kwargs)

        # Named apart from a subclass's own members.
        self._storage_profile_loaded = False
        self._storage_profile_lock = asyncio.Lock()

        self.operation_concurrency = operation_concurrency
        self.io_concurrency = io_concurrency
        self.operation_limit = operation_limit
        self.io_limit = io_limit

    async def _load_storage_profile(self) -> StorageProfile:
        """Find the profile by a request: here, ``storage_profile`` as declared.

        A subclass overrides it where a request finds the profile, e.g.
        whether an account has a hierarchical namespace. It takes no path: the
        profile is one for the whole filesystem. A storage whose behaviour
        differs by path, e.g. per bucket, is several filesystems under a
        router, each loading its own.

        `_get_storage_profile` sets what it returns as ``storage_profile``,
        then never calls it again. One that cannot tell raises: the next
        operation needing the profile tries again. Returning ``UNKNOWN``
        instead sets that for good, which is right on any storage.
        """
        return self.storage_profile

    async def _get_storage_profile(self) -> StorageProfile:
        """``storage_profile``, loaded first by `_load_storage_profile`.

        Once loaded, ``storage_profile`` is returned as it is. A first call
        loads it while the others wait, then find it loaded. A load that
        raises is tried again by the next call.

        The strategies await it when they start. The subclass's primitives may
        await it too, e.g. a ``_rmdir`` depending on whether directories are
        real.
        """
        # Once loaded, no call waits for the lock.
        if self._storage_profile_loaded:
            return self.storage_profile

        async with self._storage_profile_lock:
            if not self._storage_profile_loaded:
                self.storage_profile = await self._load_storage_profile()
                self._storage_profile_loaded = True
        return self.storage_profile

    def _unwrapped(self) -> AsyncFileSystem:
        """The filesystem a monitor is told of: this one, or the one a wrapper wraps."""
        return self

    def close(self) -> None:
        """Release what the filesystem holds: here, nothing.

        fsspec never calls it. A router closes its inner filesystems with it,
        and ``with`` closes the filesystem it opened. A subclass, or a mixin
        such as `SyncClientSupport`, releases its own, e.g. a client or
        threads, and calls this one. It may be called more than once: each
        releases what is held then.
        """

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
            entry = await self._info(self._strip_protocol(path))
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
                self,
                self,
                _strip_only_protocol(self, path1),
                _strip_only_protocol(self, path2),
                maxdepth=maxdepth,
                root_entry=entry,
            ),
            Call(self._unwrapped(), "copy", path1, path2),
            kwargs.get("callback"),
        )
        _raise(_copy_errors(errors, on_error), "Copying")

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
        if path1 == path2:
            return

        entry = None
        if not recursive:
            entry = await self._single_file(path1)
            if entry is None:
                # fsspec's own: a copy, then the deletion of its sources.
                await self._copy(path1, path2, recursive, "raise", maxdepth, **kwargs)
                await self._rm(path1, recursive=recursive)
                return

        _, errors = await _collect_with_monitors(
            self._mv_strategy(
                self,
                self,
                _strip_only_protocol(self, path1),
                _strip_only_protocol(self, path2),
                maxdepth=maxdepth,
                root_entry=entry,
            ),
            Call(self._unwrapped(), "mv", path1, path2),
            kwargs.get("callback"),
        )
        _raise(errors, "Moving")

    # fsspec makes sync methods only from the async ones it knows: _mv and _rmdir are
    # not. Each awaits the instance's own, so that a subclass's override is the one run.

    def mv(self, path1: str | list[str], path2: str, **kwargs: Any) -> None:
        return sync(self.loop, self._mv, path1, path2, **kwargs)

    def rmdir(self, path: str) -> None:
        return sync(self.loop, self._rmdir, path)

    async def _cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        """Copy a file, as fsspec's ``_cp_file`` does: by ``_cp_file_entry``, with its ``_info``.

        The strategies call ``_cp_file_entry`` with the entry their listing found.
        fsspec's own methods, and other callers, know only the path: its entry
        is asked first. A subclass whose copy needs nothing of the entry may
        override this too, to save that request.
        """
        entry = await self._info(path1)
        await self._cp_file_entry(entry, self._strip_protocol(path2), **kwargs)  # type: ignore[attr-defined]

    async def _mv_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        """Move a file, as fsspec's ``_mv_file`` does: by ``_mv_file_entry``, with its ``_info``.

        As `_cp_file`, for a move.
        """
        entry = await self._info(path1)
        await self._mv_file_entry(entry, self._strip_protocol(path2), **kwargs)  # type: ignore[attr-defined]

    async def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> bytes:
        """Read a file, or a range of it, through a file of ``open_async``.

        The file's requests each hold a permit of ``io_limit``. A subclass
        overriding this, or ``_pipe_file``, ``_get_file`` or ``_put_file``, e.g.
        with one download, holds a permit for each request it sends itself. It
        holds none while it calls the filesystem's files, which take their own:
        held around them, it would wait for itself. fsspec's methods on many
        paths call it at most ``io_concurrency`` at a time, whatever it does.
        """
        f = await self.open_async(path, "rb", **kwargs)
        async with f:
            position = 0
            if start is not None:
                position = start if start >= 0 else max(0, f.size + start)
                await _skip(f, position, self.transfer_size)

            if end is not None:
                if end < 0:
                    end = f.size + end
                return await f.read(max(0, end - position))
            else:
                return await f.read()

    async def _pipe_file(
        self, path: str, value: bytes, mode: str = "overwrite", **kwargs: Any
    ) -> None:
        """Write a file whole, through a file of ``open_async``: see `_cat_file` for ``io_limit``."""
        f = await self.open_async(path, "xb" if mode == "create" else "wb", **kwargs)
        async with f:
            await f.write(value)

    async def _get_file(
        self,
        rpath: str,
        lpath: str,
        callback: Callback = DEFAULT_CALLBACK,
        **kwargs: Any,
    ) -> None:
        """Copy a file to a local path, through a file of ``open_async``: see `_cat_file` for ``io_limit``."""
        if os.path.isdir(lpath):  # noqa: ASYNC240, as the local open below.
            return  # A directory, which fsspec's get makes itself.

        f = await self.open_async(rpath, "rb", **kwargs)
        async with f:
            # The bytes, as fsspec's own filesystems tell them to the callback.
            callback.set_size(f.size)

            # Local writes are short next to a storage's requests, as in fsspec's
            # own filesystems.
            with open(lpath, "wb") as local:  # noqa: ASYNC230
                while chunk := await f.read(self.transfer_size):
                    local.write(chunk)
                    callback.relative_update(len(chunk))

    async def _put_file(
        self,
        lpath: str,
        rpath: str,
        mode: str = "overwrite",
        callback: Callback = DEFAULT_CALLBACK,
        **kwargs: Any,
    ) -> None:
        """Copy a local file to the storage, through a file of ``open_async``: see `_cat_file` for ``io_limit``."""
        if os.path.isdir(lpath):  # noqa: ASYNC240, as the local open below.
            return  # A directory: a storage needs none for the files put in it.

        f = await self.open_async(rpath, "xb" if mode == "create" else "wb", **kwargs)
        async with f:
            with open(lpath, "rb") as local:  # noqa: ASYNC230
                callback.set_size(os.fstat(local.fileno()).st_size)
                while chunk := local.read(self.transfer_size):
                    await f.write(chunk)
                    callback.relative_update(len(chunk))

    # fsspec's methods on many paths call the method on one path for each, up to
    # batch_size at a time: here by default, io_concurrency. So they bound a
    # subclass's _cat_file holding no permit, e.g. one download, without
    # taking a permit around each call, which would wait for itself as
    # _cat_file says.

    def _io_batch_size(self, batch_size: int | None) -> int | None:
        """The paths handled at a time: the call's, else the instance's, else ``io_concurrency``.

        ``io_concurrency`` given as a semaphore tells no number: fsspec's own
        default then holds, and only the permits bound the requests.
        """
        if batch_size is not None:
            return batch_size
        if self.batch_size is not None:
            return self.batch_size  # type: ignore[no-any-return]
        if isinstance(self.io_concurrency, int):
            return self.io_concurrency
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
        for i in [*indices, None]:
            if i is not None:
                start, end = starts[i] or 0, ends[i]
                assert end is not None
                end = max(end, start)
                if (
                    members
                    and start - high <= max_gap
                    and max(end, high) - low <= max_block
                ):
                    high = max(high, end)
                    members.append(i)
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
            if i is not None:
                members, low, high = [i], start, end

    return fetches, plan  # type: ignore[return-value]


def _strip_only_protocol(fs: AbstractFileSystem, path: str | Sequence[str]) -> Any:
    """A path, or paths, of ``fs`` without its protocol, keeping a trailing "/".

    The strategies take paths without a protocol. A trailing "/" means
    something to some, e.g. a glob of directories only, or a copy into a directory.
    """
    if not isinstance(path, str):
        return [_strip_only_protocol(fs, p) for p in path]
    plain = fs._strip_protocol(path)
    return plain + "/" if path.endswith("/") and not plain.endswith("/") else plain


async def _skip(f: Any, start: int, chunk_size: int) -> None:
    """Move a file just opened to ``start``: by seeking, or else by reading.

    A storage SDK's handle may stream its data, unable to seek. Its bytes up
    to ``start`` are then read and dropped, a chunk at a time, so that memory
    stays bounded.
    """
    try:
        if f.seekable():
            f.seek(start)
            return
    except (io.UnsupportedOperation, OSError):
        pass

    skipped = 0
    while skipped < start:
        chunk = await f.read(min(chunk_size, start - skipped))
        if not chunk:
            return  # Past the end: the read after answers nothing.
        skipped += len(chunk)


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
