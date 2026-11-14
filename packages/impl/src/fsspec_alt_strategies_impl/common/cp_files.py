# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The copy and move of files within one storage, a handler's ``cp_files`` and ``mv_files``: each, or by one request.

`_cp_each_file` and `_mv_each_file` are the defaults of both families'
handlers: each file by the handler's ``cp_file`` or ``mv_file``, as many at
once as its ``operation_limit`` allows. A storage copying or moving several
files by one request, e.g. the Hugging Face Hub's commit, overrides
``cp_files`` or ``mv_files`` with `_cp_by_request` or `_mv_by_request`, given
that request. The recursive copies and moves send their files to them in
batches of the handler's ``cp_batch_size`` or ``mv_batch_size``.

Each error yielded notes its paths, and has its source's path as its
``source`` attribute, so that the caller tells which file it is of.

This module depends on `handler` and `operation`.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import CpState, MvState

from .operation import Operation

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping

    from .handler import FileTransfer, StorageHandler

__all__: list[str] = []


def _cp_each_file(
    handler: StorageHandler, transfers: list[FileTransfer]
) -> Operation[tuple[str, str], CpState]:
    """Copy files, each by the handler's ``cp_file``, as many at once as its ``operation_limit`` allows.

    It yields ``(source, destination)`` for each file copied, and the error of
    each that failed, noting its paths. Each copy holds a permit of the limit
    while it runs.

    It wins on a storage copying one file per request, as most do. A storage
    copying several by one request wins with `_cp_by_request` instead.

    Once cancelled, it starts no new copy, and yields those in flight. Closed,
    it waits for those in flight without yielding them.

    Requests: one ``cp_file`` per file.

    The operation yields ``(source, destination)``, or an error, and its state
    is a `CpState`.
    """

    def send(t: FileTransfer) -> Awaitable[None]:
        return handler.cp_file(t.entry, t.path2, missing=t.missing, parent=t.parent)

    return Operation(
        lambda op: _each(handler, transfers, send, "Copying", _copied, op),
        CpState(),
    )


def _mv_each_file(
    handler: StorageHandler, transfers: list[FileTransfer]
) -> Operation[tuple[str, str], MvState]:
    """Move files, each by the handler's ``mv_file``, as many at once as its ``operation_limit`` allows.

    As `_cp_each_file`, for a move. A storage moving several files by one
    request wins with `_mv_by_request` instead.

    Requests: one ``mv_file`` per file.

    The operation yields ``(source, destination)``, or an error, and its state
    is an `MvState`.
    """

    def send(t: FileTransfer) -> Awaitable[None]:
        return handler.mv_file(t.entry, t.path2, missing=t.missing, parent=t.parent)

    return Operation(
        lambda op: _each(handler, transfers, send, "Moving", _moved, op),
        MvState(),
    )


async def _each(
    handler: StorageHandler,
    transfers: list[FileTransfer],
    send: Callable[[FileTransfer], Awaitable[None]],
    doing: str,
    counted: Callable[[Any, str, bool], Any],
    op: Operation[tuple[str, str], Any],
) -> AsyncGenerator[tuple[str, str] | Exception]:
    """Send each transfer under a permit, yielding its outcome as it comes."""
    op.state = replace(op.state, files_found=len(transfers))
    started: set[asyncio.Task[Any]] = set()

    async def one(transfer: FileTransfer) -> bool:
        async with handler.operation_limit:
            if op.cancelling():
                return False
            started.add(asyncio.current_task())  # type: ignore[arg-type]
            await send(transfer)
            return True

    tasks = {asyncio.ensure_future(one(t)): t for t in transfers}
    try:
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                transfer = tasks[task]
                source = transfer.entry["name"]
                try:
                    if not task.result():
                        continue  # Cancelled before its turn: nothing done.
                except Exception as e:
                    op.state = counted(op.state, source, True)
                    yield _noted(e, doing, transfer)
                else:
                    op.state = counted(op.state, source, False)
                    yield source, transfer.path2
    finally:
        # Closed early: the transfers not started are dropped, those running awaited.
        for task in tasks:
            if task not in started:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _cp_by_request(
    handler: StorageHandler,
    transfers: list[FileTransfer],
    request: Callable[[list[FileTransfer]], Awaitable[Mapping[str, Exception] | None]],
) -> Operation[tuple[str, str], CpState]:
    """Copy files by one request copying them all, e.g. a commit, for a handler's ``cp_files``.

    ``request(transfers)`` copies the files, at most the handler's
    ``cp_batch_size``, and returns those it could not copy, by source path,
    each with its error. A destination refusing to overwrite is a
    FileExistsError, which the strategies retry by replacing the file on its
    own. It raises only if it failed whole: the operation then raises it, as
    nothing was copied. It holds a permit of the handler's
    ``operation_limit`` while it runs.

    It yields ``(source, destination)`` per file copied, and the error of each
    refused, noting its paths. Its one request runs as the operation starts:
    cancelling it then changes nothing.

    Requests: one ``request``.

    The operation yields ``(source, destination)``, or an error, and its state
    is a `CpState`.

    Examples
    --------
    >>> class MyHandler(TreeFileOperationHandler):
    ...     cp_batch_size = 100
    ...
    ...     def cp_files(self, transfers):
    ...         return _cp_by_request(self, transfers, self._commit_copies)
    """
    return Operation(
        lambda op: _by_request(handler, transfers, request, "Copying", _copied, op),
        CpState(),
    )


def _mv_by_request(
    handler: StorageHandler,
    transfers: list[FileTransfer],
    request: Callable[[list[FileTransfer]], Awaitable[Mapping[str, Exception] | None]],
) -> Operation[tuple[str, str], MvState]:
    """Move files by one request moving them all, e.g. a commit, for a handler's ``mv_files``.

    As `_cp_by_request`, for a move, at most the handler's ``mv_batch_size``.
    A file it could not move stays where it was.

    Requests: one ``request``.

    The operation yields ``(source, destination)``, or an error, and its state
    is an `MvState`.
    """
    return Operation(
        lambda op: _by_request(handler, transfers, request, "Moving", _moved, op),
        MvState(),
    )


async def _by_request(
    handler: StorageHandler,
    transfers: list[FileTransfer],
    request: Callable[[list[FileTransfer]], Awaitable[Mapping[str, Exception] | None]],
    doing: str,
    counted: Callable[[Any, str, bool], Any],
    op: Operation[tuple[str, str], Any],
) -> AsyncGenerator[tuple[str, str] | Exception]:
    """Send the transfers by one request, then yield each one's outcome."""
    op.state = replace(op.state, files_found=len(transfers))
    async with handler.operation_limit:
        refused = await request(transfers) or {}

    for transfer in transfers:
        source = transfer.entry["name"]
        error = refused.get(source)
        op.state = counted(op.state, source, error is not None)
        if error is None:
            yield source, transfer.path2
        else:
            yield _noted(error, doing, transfer)


def _noted(error: Exception, doing: str, transfer: FileTransfer) -> Exception:
    """A transfer's error, noting its paths, with its source's path as its ``source``."""
    source = transfer.entry["name"]
    error.add_note(f"{doing} {source!r} to {transfer.path2!r} failed.")
    error.source = source  # type: ignore[attr-defined]
    return error


def _copied(state: CpState, path: str, failed: bool) -> CpState:
    """The state once a file is copied, or failed to be."""
    return CpState(
        errors=state.errors + failed,
        current=path,
        files_found=state.files_found,
        files_copied=state.files_copied + (not failed),
    )


def _moved(state: MvState, path: str, failed: bool) -> MvState:
    """The state once a file is moved, or failed to be."""
    return MvState(
        errors=state.errors + failed,
        current=path,
        files_found=state.files_found,
        files_moved=state.files_moved + (not failed),
    )
