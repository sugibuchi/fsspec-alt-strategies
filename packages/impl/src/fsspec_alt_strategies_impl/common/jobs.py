# SPDX-FileCopyrightText: 2026 Tsuyoshi SUGIBUCHI
# SPDX-License-Identifier: Apache-2.0
"""What the strategies changing a storage share: how each file goes.

`_transfer_of` and `_moves_of` say how a copy or a move sends each file.
`_check_batch_size` refuses a handler's batch size below 1.

This module depends on `handler` and `transfer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .transfer import _across

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from typing import Protocol

    from .handler import StorageHandler

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


def _check_batch_size(batch_size: int, name: str) -> None:
    """Refuse a handler's batch size below 1, e.g. its ``rm_batch_size``, as no batch could be sent."""
    if batch_size < 1:
        raise ValueError(f"{name} must be at least 1: {batch_size}")
