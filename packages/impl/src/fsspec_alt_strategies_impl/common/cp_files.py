# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""The copy and move of files within one storage, a handler's ``cp_files`` and ``mv_files``: each, or by one request.

`_cp_each_file` and `_mv_each_file` are the defaults of every handler: each
file by the handler's ``cp_file`` or ``mv_file``, as many at once as its
``operation_limit`` allows. A storage copying or moving several files by one
request, e.g. the Hugging Face Hub's commit, overrides ``cp_files`` or
``mv_files`` with `cp_by_request` or `mv_by_request`, given that request. The
recursive copies and moves send their files to them in batches of the
handler's ``cp_batch_size`` or ``mv_batch_size``.

Each error is yielded as the storage raised it. The state's ``current`` then
names its source, so that the caller tells which file it is of.

This module depends on `handler`, `operation` and `rm_files`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fsspec_alt_strategies.states import CpState, MvState

from .operation import Operation
from .rm_files import _bounded

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping

    from .handler import FileTransfer, StorageHandler

    _Request = Callable[[list[FileTransfer]], Awaitable[Mapping[str, Exception] | None]]

__all__ = ["cp_by_request", "mv_by_request"]


def _cp_each_file(
    handler: StorageHandler, transfers: list[FileTransfer]
) -> Operation[tuple[str, str], CpState]:
    """Copy files, each by the handler's ``cp_file``, as many at once as its ``operation_limit`` allows.

    It wins on a storage copying one file per request, as most do. A storage
    copying several by one request wins with `cp_by_request` instead.

    Each copy holds a permit of the limit while it runs. Once cancelled, it
    starts no new copy, and yields those in flight. Closed, it waits for those
    in flight without yielding them.

    Requests: one ``cp_file`` per file.

    The operation yields ``(source, destination)`` per file copied, or an
    error, and its state is a `CpState`.
    """
    return Operation(
        lambda op: _each(handler, transfers, handler.cp_file, "files_copied", op),
        CpState(files_found=len(transfers)),
    )


def _mv_each_file(
    handler: StorageHandler, transfers: list[FileTransfer]
) -> Operation[tuple[str, str], MvState]:
    """Move files, each by the handler's ``mv_file``: as `_cp_each_file`, for a move.

    Requests: one ``mv_file`` per file.

    The operation yields ``(source, destination)`` per file moved, or an
    error, and its state is an `MvState`.
    """
    return Operation(
        lambda op: _each(handler, transfers, handler.mv_file, "files_moved", op),
        MvState(files_found=len(transfers)),
    )


def cp_by_request(
    handler: StorageHandler, transfers: list[FileTransfer], request: _Request
) -> Operation[tuple[str, str], CpState]:
    """Copy files by one request copying them all, e.g. a commit, for a handler's ``cp_files``.

    ``request(transfers)`` copies the files, at most the handler's
    ``cp_batch_size``, and returns those it could not copy, by source path,
    each with its error. A destination refusing to overwrite is a
    FileExistsError, which the strategies retry by replacing the file on its
    own. It raises only if it failed whole: the operation then raises it, as
    nothing was copied. It holds a permit of the handler's
    ``operation_limit`` while it runs. Its one request runs as the operation
    starts: cancelling it then changes nothing.

    Requests: one ``request``.

    The operation yields ``(source, destination)`` per file copied, or an
    error, and its state is a `CpState`.

    Examples
    --------
    >>> class MyHandler(TreeFileOperationHandler):
    ...     cp_batch_size = 100
    ...
    ...     def cp_files(self, transfers):
    ...         return cp_by_request(self, transfers, self._commit_copies)
    """
    return Operation(
        lambda op: _by_request(handler, transfers, request, "files_copied", op),
        CpState(files_found=len(transfers)),
    )


def mv_by_request(
    handler: StorageHandler, transfers: list[FileTransfer], request: _Request
) -> Operation[tuple[str, str], MvState]:
    """Move files by one request moving them all, e.g. a commit, for a handler's ``mv_files``.

    As `cp_by_request`, for a move, at most the handler's ``mv_batch_size``.
    A file it could not move stays where it was.

    Requests: one ``request``.

    The operation yields ``(source, destination)`` per file moved, or an
    error, and its state is an `MvState`.
    """
    return Operation(
        lambda op: _by_request(handler, transfers, request, "files_moved", op),
        MvState(files_found=len(transfers)),
    )


async def _each(
    handler: StorageHandler,
    transfers: list[FileTransfer],
    primitive: Callable[..., Awaitable[None]],
    done: str,
    op: Operation[tuple[str, str], Any],
) -> AsyncGenerator[tuple[str, str] | Exception]:
    """Send each transfer by ``primitive`` under a permit, yielding its outcome as it ends."""

    def send(t: FileTransfer) -> Awaitable[None]:
        return primitive(t.entry, t.path2, missing=t.missing, parent=t.parent)

    async for transfer, error in _bounded(
        handler.operation_limit, transfers, send, op.cancelling
    ):
        yield _outcome(op, done, transfer, error)


async def _by_request(
    handler: StorageHandler,
    transfers: list[FileTransfer],
    request: _Request,
    done: str,
    op: Operation[tuple[str, str], Any],
) -> AsyncGenerator[tuple[str, str] | Exception]:
    """Send the transfers by one request, then yield each one's outcome."""
    async with handler.operation_limit:
        refused = await request(transfers) or {}
    for transfer in transfers:
        yield _outcome(op, done, transfer, refused.get(transfer.entry["name"]))


def _outcome(
    op: Operation[tuple[str, str], Any],
    done: str,
    transfer: FileTransfer,
    error: Exception | None,
) -> tuple[str, str] | Exception:
    """A transfer's outcome, counted in the state: ``done`` names the count of files gone, ``current`` its source."""
    state = op.state
    source = transfer.entry["name"]
    if error is None:
        op.state = replace(state, current=source, **{done: getattr(state, done) + 1})
        return source, transfer.path2
    else:
        op.state = replace(state, current=source, errors=state.errors + 1)
        return error
