# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""How a copy or move sends each file: within one storage, or between two, its data going through the caller.

`_transfer_of` and `_moves_of` say how a copy or a move sends each file, a
`_Transfer`: the source's ``cp_file`` or ``mv_file`` within one storage,
unless the caller gives its own.

`_across` is how a copy or move between two handlers sends each file, unless
its caller gives another way, e.g. a cloud's copy from a URL. It reads by the
source's ``open_async_file``, and writes by the destination's, a block at a
time. So each handler keeps the primitives of its own storage only, and the
transfer between two is the strategy's.

A file's transfer leaves no part of a file at its destination:

- On a storage whose profile declares ``atomic_writes``, a write is seen only
  once complete, and a failed one leaves the file that was there: it writes in
  place.
- Elsewhere, it writes to a temporary name beside the destination, then moves
  it there by the destination's ``mv_file``. It deletes the temporary file if
  either fails.

It checks the bytes written against the source's size, the reader's where it
tells one, else the entry's. A file of another size raises OSError before its
write completes, so that the destination keeps nothing of it.

This module depends on `handler`.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from typing import Protocol

    from .handler import Entry, StorageHandler, StoragePath

    class _Transfer(Protocol):
        """How one file goes from its source entry to a destination path.

        It is the source's ``cp_file`` or ``mv_file`` within one storage.
        Between two, a copy's is `transfer._across`, and a move's that copy,
        then the source's ``rm_file``, unless the caller gives its own. The entry is the
        source's, as its listing or ``info`` gave it, its ``"name"`` the source
        path. ``missing`` says nothing is at the destination, and ``parent``
        is the destination's parent's entry, as the primitives take them: one
        ignoring them is right all the same.
        """

        def __call__(
            self,
            entry: dict[str, Any],
            path2: str,
            *,
            missing: bool = False,
            parent: dict[str, Any] | None = None,
        ) -> Awaitable[None]: ...


__all__: list[str] = []


def _transfer_of(
    src: StorageHandler,
    dst: StorageHandler,
    cp_entry: _Transfer | None,
    same_storage: bool | None,
) -> tuple[_Transfer, bool]:
    """How a copy sends each file, and whether it stays in one storage.

    ``same_storage`` is whether ``dst`` is ``src``, unless given. Unless
    ``cp_entry`` is given, a file goes by the source's own ``cp_file`` within
    one storage, and by `transfer._across` between two, its data through the
    caller.
    """
    if same_storage is None:
        same_storage = src is dst
    if cp_entry is None:
        # A primitive takes a StoragePath, which the strategies pass: a _Transfer.
        cp_entry = cast("_Transfer", src.cp_file) if same_storage else _across(src, dst)
    return cp_entry, same_storage


def _moves_of(
    src: StorageHandler,
    dst: StorageHandler,
    cp_entry: _Transfer | None,
    mv_entry: _Transfer | None,
    same_storage: bool | None,
) -> tuple[_Transfer, _Transfer, bool]:
    """How a move sends each file, by copy and by move, and whether it stays in one storage.

    ``cp_entry`` is as `_transfer_of` makes it. ``mv_entry``, unless given, is
    the source's ``mv_file`` within one storage. Between two, it is
    ``cp_entry``, then the source's ``rm_file``: a file whose copy failed is
    kept. So a caller's copy, e.g. a cloud's copy from a URL, moves too.
    """
    cp_entry, same_storage = _transfer_of(src, dst, cp_entry, same_storage)
    if mv_entry is None:
        if same_storage:
            mv_entry = cast("_Transfer", src.mv_file)
        else:
            copy = cp_entry

            async def move(
                entry: dict[str, Any],
                path2: str,
                *,
                missing: bool = False,
                parent: dict[str, Any] | None = None,
            ) -> None:
                await copy(entry, path2, missing=missing, parent=parent)
                await src.rm_file(entry)

            mv_entry = move

    return cp_entry, mv_entry, same_storage


def _across(src: StorageHandler, dst: StorageHandler) -> _Transfer:
    """How a copy between two handlers sends each file: streamed, leaving no part of a file.

    A move between two copies by it, then deletes the source, as `_moves_of` makes it.
    """

    async def transfer(
        entry: Entry,
        path2: str,
        *,
        missing: bool = False,
        parent: Entry | None = None,
    ) -> None:
        # The strategies place each file by the destination's path methods: its
        # path is verified already.
        destination: StoragePath = path2  # type: ignore[assignment]
        await _cross_cp_file(
            src, entry, dst, destination, missing=missing, parent=parent
        )

    return transfer


async def _cross_cp_file(
    src: StorageHandler,
    entry: Entry,
    dst: StorageHandler,
    path: StoragePath,
    *,
    missing: bool = False,
    parent: Entry | None = None,
) -> None:
    """Copy the file of ``entry``, of ``src``, to ``path`` of ``dst``, leaving no part of a file.

    ``missing`` is the destination's `StorageHandler.cp_file`'s: it goes to the
    move of the temporary file into place. ``parent``, ``path``'s parent's
    entry, goes to the write, and to that move: the temporary file is beside
    ``path``, in the same parent.
    """
    await dst.open()
    if dst.profile.atomic_writes:
        await _write(src, entry, dst, path, parent)
        return

    temporary = dst.join(
        dst.parent(path), f".{dst.name(path)}.{secrets.token_hex(8)}.part"
    )
    try:
        size = await _write(src, entry, dst, temporary, parent)
        await dst.mv_file(
            {"name": temporary, "type": "file", "size": size},
            path,
            missing=missing,
            parent=parent,
        )
    except BaseException as e:
        try:
            await dst.rm_file({"name": temporary, "type": "file"})
        except FileNotFoundError:
            pass  # Never written, or moved before the failure.
        except Exception as cleanup:
            # Its type only: a storage's message may carry credentials.
            e.add_note(
                f"Deleting the temporary file {temporary!r} failed too: "
                f"{type(cleanup).__name__}."
            )
        raise


async def _write(
    src: StorageHandler,
    entry: Entry,
    dst: StorageHandler,
    path: StoragePath,
    parent: Entry | None = None,
) -> int:
    """Stream the source's file to ``path``, a block at a time, returning its size.

    The write's entry holds its size and its parent's entry, where known.

    A size other than the source's raises OSError inside the writer, which then
    completes nothing.
    """
    opened = await src.open_async_file(entry, "rb")
    expected = getattr(opened, "size", None)
    if not isinstance(expected, int):
        expected = entry.get("size")

    async with opened as reader:
        # A path the strategies placed, verified, with the size it will have,
        # where known, as the writer may need it up front.
        target: Entry = {"name": path}
        if isinstance(expected, int):
            target["size"] = expected
        if parent is not None:
            target["parent"] = parent
        async with await dst.open_async_file(target, "wb") as writer:
            written = 0
            while block := await reader.read(src.transfer_size):
                await writer.write(block)
                written += len(block)
            if isinstance(expected, int) and written != expected:
                raise OSError(
                    f"Copying {entry['name']!r} to {path!r} read {written} bytes of"
                    f" {expected}: the file changed, or a read was cut short."
                )
    return written
