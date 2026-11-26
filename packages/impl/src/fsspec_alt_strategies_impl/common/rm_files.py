# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The deletion of files the caller found, a handler's ``rm_files``: by ``rm_file`` each, or by one request.

`_rm_each_file` is the default of every handler: each file by the
handler's ``rm_file``, as many at once as its ``operation_limit`` allows. A
storage deleting several files by one request, e.g. S3's DeleteObjects or
Azure's Blob Batch, overrides ``rm_files`` with `rm_by_request`, given that
request. The recursive deletions and moves send their files to ``rm_files`` in
batches of the handler's ``rm_batch_size``.

A missing file is deleted, as the storage is as asked: no error.

This module depends on `handler` and `operation`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, TypeVar

from fsspec_alt_strategies.states import RmState

from .operation import Operation

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping

    from .handler import Entry, StorageHandler

__all__ = ["rm_by_request"]

_T = TypeVar("_T")


def _rm_each_file(
    handler: StorageHandler, entries: list[Entry]
) -> Operation[str, RmState]:
    """Delete files, each by the handler's ``rm_file``, as many at once as its ``operation_limit`` allows.

    It yields each file's path once deleted, a missing one too, and the error
    of each file that failed, noting its path. Each deletion holds a permit
    of the limit while it runs.

    It wins on a storage deleting one file per request, as most do. A storage
    deleting several by one request wins with `rm_by_request` instead.

    Once cancelled, it starts no new deletion, and yields those in flight.
    Closed, it waits for those in flight without yielding them.

    Requests: one ``rm_file`` per file.

    The operation yields each path deleted, or an error, and its state is an
    `RmState`.
    """

    async def delete(entry: Entry) -> None:
        # A file missing is deleted: the storage is as asked.
        with contextlib.suppress(FileNotFoundError):
            await handler.rm_file(entry)

    async def run(op: Operation[str, RmState]) -> AsyncGenerator[str | Exception]:
        op.state = RmState(files_found=len(entries))
        sent = _bounded(handler.operation_limit, entries, delete, op.cancelling)
        async for entry, error in sent:
            path = entry["name"]
            op.state = _counted(op.state, path, failed=error is not None)
            if error is None:
                yield path
            else:
                error.add_note(f"Deleting {path!r} failed.")
                yield error

    return Operation(run, RmState())


async def _bounded(
    limit: asyncio.Semaphore,
    items: list[_T],
    send: Callable[[_T], Awaitable[None]],
    cancelling: Callable[[], object],
) -> AsyncGenerator[tuple[_T, Exception | None]]:
    """Send each item under a permit of ``limit``, yielding it with its error, or None, as each ends.

    Once ``cancelling``, an item not started yet is dropped, never yielded.
    Closed early, those not started are dropped, and those running awaited: a
    request sent is never left behind.
    """
    started: set[asyncio.Task[Any]] = set()

    async def one(item: _T) -> bool:
        async with limit:
            if cancelling():
                return False
            started.add(asyncio.current_task())  # type: ignore[arg-type]
            await send(item)
            return True

    tasks = {asyncio.ensure_future(one(item)): item for item in items}
    try:
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                try:
                    if not task.result():
                        continue  # Cancelled before its turn: nothing done.
                except Exception as e:
                    yield tasks[task], e
                else:
                    yield tasks[task], None
    finally:
        for task in tasks:
            if task not in started:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def rm_by_request(
    handler: StorageHandler,
    entries: list[Entry],
    request: Callable[[list[Entry]], Awaitable[Mapping[str, Exception] | None]],
) -> Operation[str, RmState]:
    """Delete files by one request deleting them all, e.g. S3's DeleteObjects, for a handler's ``rm_files``.

    ``request(entries)`` deletes the files, at most the handler's
    ``rm_batch_size``, and returns those it could not delete, by path, each
    with its error, a missing file being deleted. It raises only if it failed
    whole: the operation then raises it, as nothing was deleted. It holds a
    permit of the handler's ``operation_limit`` while it runs.

    It yields each path deleted, and the error of each refused, noting its
    path. Its one request runs as the operation starts: cancelling it then
    changes nothing.

    Requests: one ``request``.

    The operation yields each path deleted, or an error, and its state is an
    `RmState`.

    Examples
    --------
    >>> class MyHandler(TreeFileOperationHandler):
    ...     rm_batch_size = 1000
    ...
    ...     def rm_files(self, entries):
    ...         return rm_by_request(self, entries, self._delete_objects)
    """

    async def run(op: Operation[str, RmState]) -> AsyncGenerator[str | Exception]:
        op.state = RmState(files_found=len(entries))
        async with handler.operation_limit:
            refused = await request(entries) or {}

        for entry in entries:
            path = entry["name"]
            error = refused.get(path)
            op.state = _counted(op.state, path, failed=error is not None)
            if error is None:
                yield path
            else:
                error.add_note(f"Deleting {path!r} failed.")
                yield error

    return Operation(run, RmState())


def _counted(state: RmState, path: str, *, failed: bool) -> RmState:
    """The state once a file is deleted, or failed to be."""
    return RmState(
        errors=state.errors + failed,
        current=path,
        files_found=state.files_found,
        files_deleted=state.files_deleted + (not failed),
    )
