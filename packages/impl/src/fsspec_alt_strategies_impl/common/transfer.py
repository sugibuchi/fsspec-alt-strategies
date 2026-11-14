# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""A file's copy or move between two storages, its data going through the caller.

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

This module depends on `handler` of `common`.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .handler import Entry, StorageHandler, StoragePath
    from .jobs import _Transfer

__all__: list[str] = []

_MiB = 1024 * 1024

_BLOCK_SIZE = 8 * _MiB
"""The bytes read, then written, at a time: a transfer holds one block in memory."""


def _across(src: StorageHandler, dst: StorageHandler) -> _Transfer:
    """How a copy between two handlers sends each file: streamed, leaving no part of a file.

    A move between two copies by it, then deletes the source, as `jobs._moves_of` makes it.
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
            while block := await reader.read(_BLOCK_SIZE):
                await writer.write(block)
                written += len(block)
            if isinstance(expected, int) and written != expected:
                raise OSError(
                    f"Copying {entry['name']!r} to {path!r} read {written} bytes of"
                    f" {expected}: the file changed, or a read was cut short."
                )
    return written
